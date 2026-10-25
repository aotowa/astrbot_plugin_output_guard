"""输出审查的纯文本规则引擎。

本模块刻意不导入 AstrBot，方便脱离框架单独测试。
`main.py` 负责把 AstrBot 的消息链、配置和这里的结果对接起来。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# 合法取值
ACTIONS = ("block", "replace", "flag")
MATCH_TYPES = ("keyword", "regex")


@dataclass
class RuleHit:
    """一次规则命中记录。"""

    rule: str
    action: str
    matched: str


@dataclass
class ReviewResult:
    """规则审查的汇总结果。

    `text` 是应用替换规则之后的文本；`blocked` 表示应直接丢弃；
    `flagged` 表示内容可疑、建议转人工（是否真正转人工由配置决定）。
    """

    text: str = ""
    blocked: bool = False
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)
    hits: list[RuleHit] = field(default_factory=list)


def _short(text: str, limit: int = 40) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def split_patterns(raw: object, match_type: str = "keyword") -> list[str]:
    """把配置里的"匹配内容"拆成多个独立条目。

    分隔规则：
    - 换行：总是分隔，一行一个（推荐写法）；
    - ``|``：仅关键字模式作为分隔符，方便单行填写；
      正则模式不拆 ``|``，因为它在正则里表示"或"。
    空格不会被当作分隔符，所以关键字内部可以带空格。
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items = [str(item) for item in raw]
    else:
        text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
        if match_type == "keyword":
            text = text.replace("|", "\n").replace("｜", "\n")
        items = text.split("\n")

    result: list[str] = []
    for item in items:
        value = item.strip()
        if value and value not in result:
            result.append(value)
    return result


class Rule:
    """单条审查规则，可以包含多个匹配内容。"""

    def __init__(
        self,
        name: str = "",
        pattern: object = "",
        action: str = "flag",
        match_type: str = "keyword",
        replacement: str = "",
        case_sensitive: bool = False,
        enabled: bool = True,
    ) -> None:
        self.name = str(name or "").strip() or "未命名规则"
        action = str(action or "flag").strip().lower()
        self.action = action if action in ACTIONS else "flag"
        match_type = str(match_type or "keyword").strip().lower()
        self.match_type = match_type if match_type in MATCH_TYPES else "keyword"
        self.replacement = "" if replacement is None else str(replacement)
        self.case_sensitive = bool(case_sensitive)
        self.enabled = bool(enabled)

        self.patterns = split_patterns(pattern, self.match_type)
        self.errors: list[str] = []
        self._compiled: list[re.Pattern[str]] = []
        if self.match_type == "regex":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            for value in self.patterns:
                try:
                    self._compiled.append(re.compile(value, flags))
                except re.error as exc:
                    # 用户填错正则时只跳过这一条，不影响同规则的其他内容
                    self.errors.append(f"正则「{value}」无效：{exc}")

    @property
    def error(self) -> str | None:
        """兼容旧代码的聚合错误信息。"""
        return "；".join(self.errors) if self.errors else None

    # ------------------------------------------------------------------ 匹配
    def matches(self, text: str) -> list[str]:
        """返回所有命中片段；没有命中时返回空列表。"""
        if not text or not self.patterns:
            return []

        found: list[str] = []
        if self.match_type == "regex":
            for regex in self._compiled:
                found.extend(
                    match.group(0)
                    for match in regex.finditer(text)
                    if match.group(0)
                )
            return found

        for needle in self.patterns:
            target = needle if self.case_sensitive else needle.lower()
            haystack = text if self.case_sensitive else text.lower()
            start = 0
            while True:
                index = haystack.find(target, start)
                if index < 0:
                    break
                found.append(text[index : index + len(needle)])
                start = index + len(needle)
        return found

    def apply_replace(self, text: str) -> str:
        """只执行替换动作，供按消息段重写文本时使用。"""
        if not text or not self.patterns:
            return text

        if self.match_type == "regex":
            for regex in self._compiled:
                text = regex.sub(self.replacement, text)
            return text

        for needle in self.patterns:
            if self.case_sensitive:
                text = text.replace(needle, self.replacement)
            else:
                # 用 lambda 避免替换串里的反斜杠被当作反向引用
                text = re.sub(
                    re.escape(needle),
                    lambda _match: self.replacement,
                    text,
                    flags=re.IGNORECASE,
                )
        return text


class RuleEngine:
    """按顺序执行一组规则。"""

    def __init__(
        self,
        rules: list[Rule] | None = None,
        bypass_patterns: list[str] | None = None,
    ) -> None:
        self.rules: list[Rule] = list(rules or [])
        self.errors: list[str] = []
        for rule in self.rules:
            for detail in rule.errors:
                self.errors.append(f"规则「{rule.name}」{detail}")
        self._bypass: list[re.Pattern[str]] = []
        for raw in bypass_patterns or []:
            pattern = str(raw or "").strip()
            if not pattern:
                continue
            try:
                self._bypass.append(re.compile(pattern))
            except re.error as exc:
                self.errors.append(f"白名单正则「{pattern}」无效：{exc}")

    # ------------------------------------------------------------ 配置转换
    @classmethod
    def from_config(
        cls,
        raw_rules: object,
        bypass_patterns: object = None,
    ) -> "RuleEngine":
        """把 WebUI 模板列表里的配置转换成规则对象。

        兼容两种情况：直接是 list[dict]，或者是一段 JSON 字符串。
        """
        if isinstance(raw_rules, str):
            try:
                raw_rules = json.loads(raw_rules)
            except json.JSONDecodeError:
                raw_rules = []

        rules: list[Rule] = []
        if isinstance(raw_rules, list):
            for item in raw_rules:
                if not isinstance(item, dict):
                    continue
                rules.append(
                    Rule(
                        name=item.get("name", ""),
                        pattern=item.get("pattern", ""),
                        action=item.get("action", "flag"),
                        match_type=item.get("match_type", "keyword"),
                        replacement=item.get("replacement", ""),
                        case_sensitive=item.get("case_sensitive", False),
                        enabled=item.get("enabled", True),
                    )
                )

        bypass: list[str] = []
        if isinstance(bypass_patterns, str):
            bypass = [line for line in bypass_patterns.splitlines() if line.strip()]
        elif isinstance(bypass_patterns, list):
            bypass = [str(x) for x in bypass_patterns if str(x).strip()]

        return cls(rules, bypass)

    # ------------------------------------------------------------------ 行为
    def is_bypassed(self, text: str) -> bool:
        if not text:
            return False
        return any(pattern.search(text) for pattern in self._bypass)

    def apply(self, text: str) -> ReviewResult:
        """执行所有规则；遇到 block 立即返回。"""
        result = ReviewResult(text=text)
        for rule in self.rules:
            if not rule.enabled or not rule.patterns:
                continue
            hits = rule.matches(result.text)
            if not hits:
                continue

            result.hits.append(
                RuleHit(rule=rule.name, action=rule.action, matched=hits[0])
            )

            if rule.action == "block":
                result.blocked = True
                result.reasons.append(
                    f"命中拦截规则「{rule.name}」：{_short(hits[0])}"
                )
                return result

            if rule.action == "replace":
                replaced = rule.apply_replace(result.text)
                if replaced != result.text:
                    result.text = replaced
                    result.reasons.append(
                        f"按替换规则「{rule.name}」替换了 {len(hits)} 处内容"
                    )
                continue

            # flag
            result.flagged = True
            result.reasons.append(f"命中标记规则「{rule.name}」：{_short(hits[0])}")

        return result

    def rewrite(self, text: str) -> str:
        """只执行 replace 规则，用于按消息段重写待发送文本。"""
        for rule in self.rules:
            if rule.enabled and rule.action == "replace" and rule.patterns:
                text = rule.apply_replace(text)
        return text


# ---------------------------------------------------------------- AI 结论解析
@dataclass
class Verdict:
    """AI 审查结论。"""

    passed: bool
    reason: str = ""
    rewrite: str = ""


_TRUE_WORDS = {
    "true",
    "yes",
    "pass",
    "passed",
    "ok",
    "allow",
    "allowed",
    "approved",
    "safe",
    "通过",
    "放行",
    "安全",
}
_FALSE_WORDS = {
    "false",
    "no",
    "fail",
    "failed",
    "reject",
    "rejected",
    "block",
    "blocked",
    "deny",
    "denied",
    "unsafe",
    "不通过",
    "拒绝",
    "拦截",
    "危险",
}


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_verdict(raw: str | None) -> Verdict | None:
    """从模型输出里解析 JSON 结论。

    解析失败返回 None，由调用方根据 fail_open 决定放行还是转人工。
    """
    if not raw:
        return None

    text = _strip_code_fence(str(raw))
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    passed: bool | None = None
    for key in ("pass", "passed", "ok", "approved", "allow", "safe"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool):
            passed = value
            break
        if isinstance(value, (int, float)):
            passed = bool(value)
            break
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUE_WORDS:
                passed = True
                break
            if lowered in _FALSE_WORDS:
                passed = False
                break

    if passed is None:
        risk = data.get("risk") or data.get("risk_level") or data.get("level")
        risk_text = str(risk or "").strip().lower()
        if risk_text in {"high", "critical", "危险", "高"}:
            passed = False
        elif risk_text in {"low", "none", "safe", "安全", "低"}:
            passed = True

    if passed is None:
        return None

    reason = data.get("reason") or data.get("comment") or data.get("explain") or ""
    rewrite = (
        data.get("rewrite")
        or data.get("suggested_text")
        or data.get("rewritten")
        or ""
    )
    return Verdict(
        passed=passed,
        reason=str(reason).strip(),
        rewrite=str(rewrite).strip(),
    )
