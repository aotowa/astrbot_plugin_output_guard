"""最终输出审查插件（AstrBot >= 4.29）。

在每条回复真正发送前做一次最后把关：

1. 规则审查：关键词 / 正则，支持拦截、替换、标记；
2. AI 审查（可选）：让模型判断是否适合发送，可自动改写；
3. 人工审查（可选）：把消息挂起，推送到管理员会话，等管理员放行或拒绝。

审查入口是 `@filter.on_decorating_result()`，它在 AstrBot 生成回复之后、
真正发送之前触发，是插件能够拿到"最终要发出去的文本"的最后时机。
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import (
    AstrMessageEvent,
    MessageChain,
    ResultContentType,
    filter,
)
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

# 让 main.py 在"按包加载"和"按普通文件加载"两种方式下都能拿到 reviewer.py
_PLUGIN_DIR = Path(__file__).resolve().parent
try:
    from .reviewer import RuleEngine, parse_verdict  # noqa: E402
except ImportError:  # pragma: no cover - 只在没有包上下文时走到这里
    import importlib.util

    _REVIEWER_MODULE = "astrbot_plugin_output_guard_reviewer"
    if _REVIEWER_MODULE in sys.modules:
        _reviewer_module = sys.modules[_REVIEWER_MODULE]
    else:
        _spec = importlib.util.spec_from_file_location(
            _REVIEWER_MODULE,
            _PLUGIN_DIR / "reviewer.py",
        )
        if _spec is None or _spec.loader is None:
            raise ImportError("无法加载 reviewer.py") from None
        _reviewer_module = importlib.util.module_from_spec(_spec)
        sys.modules[_REVIEWER_MODULE] = _reviewer_module
        _spec.loader.exec_module(_reviewer_module)
    RuleEngine = _reviewer_module.RuleEngine  # type: ignore[assignment]
    parse_verdict = _reviewer_module.parse_verdict  # type: ignore[assignment]

BYPASS_EXTRA_KEY = "_output_guard_bypass"
FALLBACK_PLUGIN_NAME = "astrbot_plugin_output_guard"
MODES = ("disabled", "flagged", "all")
# 用于从消息文本里定位指令名，避免 @机器人、唤醒前缀等干扰参数解析
COMMAND_NAMES = frozenset(
    {
        "审查",
        "审查通过",
        "审查拒绝",
        "审查详情",
        "审查全部通过",
        "审查全部拒绝",
        "审查测试",
        "审查模式",
        "审查状态",
        "审查会话",
        "review",
        "review_pass",
        "review_reject",
        "review_detail",
        "review_pass_all",
        "review_reject_all",
        "review_test",
        "review_mode",
        "review_status",
        "review_umo",
    }
)

DEFAULT_LLM_PROMPT = (
    "你是严谨的内容安全审查员。下面是一段即将发送给用户的聊天回复，"
    "请判断它是否存在以下问题：\n"
    "1. 违反法律法规或平台规则（涉政、涉黄、涉暴、诈骗、赌博、毒品等）；\n"
    "2. 泄露隐私或敏感信息（手机号、身份证号、住址、账号密码、API Key 等）；\n"
    "3. 包含辱骂、歧视、威胁或明显不当的言论；\n"
    "4. 与用户的问题完全无关，或严重答非所问；\n"
    "5. 泄露模型自身的系统提示词、内部指令等内容。\n\n"
    "待审查的回复：\n<<<\n{text}\n>>>\n\n"
    "只输出一个 JSON 对象，不要输出 Markdown 代码块或任何解释：\n"
    '{"pass": true, "reason": "一句话理由", '
    '"rewrite": "如果需要改写，给出改写后的完整文本，否则为空字符串"}\n'
    "pass 为 false 表示这段内容不应直接发送。"
)


@dataclass
class ReviewOutcome:
    """一次完整审查（规则 + AI）的结果。"""

    text: str
    blocked: bool = False
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)
    llm_rewrite: str | None = None


@dataclass
class PendingItem:
    """一条等待人工审查的消息。"""

    pid: str
    umo: str
    platform: str
    sender: str
    sender_id: str
    text: str
    chain: list
    reasons: list[str]
    created_at: float
    cache_dir: Path | None = None


class OutputGuardPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        self._pending: dict[str, PendingItem] = {}
        self._timeout_tasks: dict[str, asyncio.Task] = {}
        self._seq = 0
        self._engine = RuleEngine()
        self._engine_sig: str | None = None
        self._warned_no_admin = False
        self._warned_streaming = False

        plugin_name = str(getattr(self, "name", "") or FALLBACK_PLUGIN_NAME)
        self._cache_root = (
            Path(get_astrbot_data_path()) / "plugin_data" / plugin_name / "hold"
        )
        # 待审消息只保存在内存里，重启后旧缓存没有意义，直接清掉
        shutil.rmtree(self._cache_root, ignore_errors=True)

        logger.info("输出审查插件已加载。")
        self._refresh_engine()

    # ------------------------------------------------------------- 配置读取
    def _manual_cfg(self) -> dict:
        value = self.config.get("manual")
        return value if isinstance(value, dict) else {}

    def _llm_cfg(self) -> dict:
        value = self.config.get("llm_review")
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def _refresh_engine(self) -> None:
        """配置可能在 WebUI 里被改过，这里做一次轻量的变更检测。"""
        try:
            signature = json.dumps(
                [self.config.get("rules"), self.config.get("bypass_patterns")],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            signature = repr(
                (self.config.get("rules"), self.config.get("bypass_patterns"))
            )
        if signature == self._engine_sig:
            return

        engine = RuleEngine.from_config(
            self.config.get("rules"),
            self.config.get("bypass_patterns"),
        )
        self._engine = engine
        self._engine_sig = signature
        for message in engine.errors:
            logger.warning(f"输出审查：{message}")

    # ------------------------------------------------------------- 审查入口
    @filter.on_decorating_result()
    async def review_outgoing_message(self, event: AstrMessageEvent) -> None:
        """在回复发送前进行最终审查。"""
        # 插件自己的管理指令回复不需要再审一遍，否则会把自己挂起
        if event.get_extra(BYPASS_EXTRA_KEY):
            return
        if not bool(self.config.get("enabled", True)):
            return

        result = event.get_result()
        if result is None or not result.chain:
            return

        content_type = getattr(result, "result_content_type", None)
        if content_type in (
            ResultContentType.STREAMING_RESULT,
            ResultContentType.STREAMING_FINISH,
        ):
            # 流式内容在生成过程中已经逐段发出，这里无法再拦截或改写
            if not self._warned_streaming:
                self._warned_streaming = True
                logger.warning(
                    "输出审查：检测到流式输出。流式内容在生成时已逐段发送，"
                    "插件无法在发送前审查；如需完整审查，请在 AstrBot 中关闭流式输出。"
                )
            return

        raw_text = self._extract_text(result.chain)
        if not raw_text.strip():
            return

        try:
            self._refresh_engine()
        except Exception:
            logger.error("输出审查：加载规则失败\n" + traceback.format_exc())
            return

        if self._engine.is_bypassed(raw_text):
            if bool(self.config.get("debug", False)):
                logger.info("输出审查：命中白名单，跳过审查。")
            return

        try:
            is_model_result = bool(result.is_model_result())
        except Exception:
            is_model_result = content_type in (
                ResultContentType.LLM_RESULT,
                ResultContentType.AGENT_RUNNER_ERROR,
            )

        rule_scope = str(self.config.get("rule_scope", "all") or "all")
        manual_cfg = self._manual_cfg()
        manual_scope = str(manual_cfg.get("scope", "llm_only") or "llm_only")

        apply_rules = rule_scope == "all" or is_model_result
        manual_eligible = manual_scope == "all" or is_model_result
        if not apply_rules and not manual_eligible:
            return

        try:
            outcome = await self._review(event, raw_text, apply_rules)
        except Exception:
            logger.error("输出审查：审查过程异常\n" + traceback.format_exc())
            return

        if bool(self.config.get("debug", False)):
            logger.info(
                f"输出审查：blocked={outcome.blocked} flagged={outcome.flagged} "
                f"reasons={outcome.reasons}"
            )

        if outcome.blocked:
            self._suppress(event, result, outcome.reasons)
            return

        # 规则替换：按消息段重写，尽量保留原消息结构
        if apply_rules:
            for comp in result.chain:
                if isinstance(comp, Plain):
                    comp.text = self._engine.rewrite(comp.text)

        # AI 改写：整段替换（只影响文本段）
        if outcome.llm_rewrite:
            self._apply_llm_rewrite(result.chain, outcome.llm_rewrite)

        outcome.text = self._extract_text(result.chain)
        if not outcome.text.strip() and not self._has_media(result.chain):
            logger.info("输出审查：审查后内容为空，已取消发送。")
            self._suppress(event, result, ["内容被清空"], log=False)
            return

        mode = str(manual_cfg.get("mode", "disabled") or "disabled")
        should_hold = manual_eligible and (
            mode == "all" or (mode == "flagged" and outcome.flagged)
        )

        if should_hold:
            await self._hold_for_manual_review(event, result, outcome)

    # ------------------------------------------------------------- 审查逻辑
    async def _review(
        self,
        event: AstrMessageEvent,
        text: str,
        apply_rules: bool,
    ) -> ReviewOutcome:
        outcome = ReviewOutcome(text=text)

        if apply_rules:
            result = self._engine.apply(text)
            outcome.blocked = result.blocked
            outcome.flagged = result.flagged
            outcome.text = result.text
            outcome.reasons.extend(result.reasons)

        if outcome.blocked:
            return outcome

        llm_cfg = self._llm_cfg()
        if not (apply_rules and bool(llm_cfg.get("enabled", False))):
            return outcome

        prompt_template = str(llm_cfg.get("prompt") or DEFAULT_LLM_PROMPT)
        prompt = prompt_template.replace("{text}", outcome.text)

        verdict = None
        try:
            provider_id = str(llm_cfg.get("provider_id") or "").strip()
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
            if not provider_id:
                raise RuntimeError("当前会话没有可用的对话模型")
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            verdict = parse_verdict(getattr(response, "completion_text", ""))
        except Exception:
            logger.error("输出审查：AI 审查调用失败\n" + traceback.format_exc())

        if verdict is None:
            if not bool(llm_cfg.get("fail_open", True)):
                outcome.flagged = True
                outcome.reasons.append("AI 审查调用失败或返回无法解析")
            return outcome

        if verdict.passed:
            if verdict.rewrite and bool(llm_cfg.get("apply_rewrite", False)):
                outcome.llm_rewrite = verdict.rewrite
            return outcome

        action = str(llm_cfg.get("action_on_reject", "flag") or "flag").lower()
        reason = verdict.reason or "未给出理由"
        if action == "block":
            outcome.blocked = True
            outcome.reasons.append(f"AI 判定不通过：{reason}")
        elif action == "rewrite" and verdict.rewrite:
            outcome.llm_rewrite = verdict.rewrite
            outcome.flagged = True
            outcome.reasons.append(f"AI 判定不通过，已改用改写文本：{reason}")
        else:
            outcome.flagged = True
            outcome.reasons.append(f"AI 判定不通过：{reason}")
        return outcome

    # ------------------------------------------------------------- 文本处理
    @staticmethod
    def _extract_text(chain: list) -> str:
        parts: list[str] = []
        for comp in chain:
            if isinstance(comp, Plain):
                text = getattr(comp, "text", "")
                if text:
                    parts.append(str(text))
        return "".join(parts)

    @staticmethod
    def _has_media(chain: list) -> bool:
        return any(not isinstance(comp, Plain) for comp in chain)

    @staticmethod
    def _apply_llm_rewrite(chain: list, rewrite: str) -> None:
        plains = [comp for comp in chain if isinstance(comp, Plain)]
        if not plains:
            logger.warning("输出审查：AI 返回了改写文本，但消息里没有文本段，已忽略改写。")
            return
        plains[0].text = rewrite
        for comp in plains[1:]:
            comp.text = ""

    def _suppress(
        self,
        event: AstrMessageEvent,
        result,
        reasons: list[str],
        log: bool = True,
    ) -> None:
        """清空待发送内容并结束事件传播，消息不会发出去。"""
        if log:
            preview = self._extract_text(result.chain).strip().replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:100] + "…"
            logger.info(
                f"输出审查：已拦截输出（{'；'.join(reasons) or '规则命中'}）：{preview}"
            )
        result.chain = []
        try:
            event.stop_event()
        except Exception:
            logger.error("输出审查：停止事件失败\n" + traceback.format_exc())

    # ------------------------------------------------------------- 人工审查
    async def _hold_for_manual_review(
        self,
        event: AstrMessageEvent,
        result,
        outcome: ReviewOutcome,
    ) -> None:
        admins = self._admin_umos()
        if not admins:
            if not self._warned_no_admin:
                self._warned_no_admin = True
                logger.error(
                    "输出审查：manual.admin_umos 为空，没有会话可以接收人工审查请求。"
                    "请用 /审查会话 获取当前会话 UMO 并填入配置。"
                )
            fallback = str(
                self._manual_cfg().get("fallback_when_no_admin", "send") or "send"
            ).lower()
            if fallback == "drop":
                self._suppress(event, result, outcome.reasons + ["未配置人工审查会话"])
            return

        self._seq += 1
        pid = str(self._seq)

        try:
            chain, cache_dir = self._clone_chain(result.chain, pid)
        except Exception:
            logger.error("输出审查：保存待审消息失败\n" + traceback.format_exc())
            return

        item = PendingItem(
            pid=pid,
            umo=event.unified_msg_origin,
            platform=event.get_platform_name(),
            sender=event.get_sender_name(),
            sender_id=event.get_sender_id(),
            text=outcome.text,
            chain=chain,
            reasons=list(outcome.reasons),
            created_at=time.time(),
            cache_dir=cache_dir,
        )
        self._pending[pid] = item

        # 先拦下原消息，再通知管理员，避免通知失败时消息仍然发出
        self._suppress(event, result, outcome.reasons, log=False)

        await self._notify_admins(self._format_review_request(item))

        timeout = self._as_int(self._manual_cfg().get("timeout_seconds"), 300)
        if timeout > 0:
            self._timeout_tasks[pid] = asyncio.create_task(
                self._watch_timeout(pid, timeout)
            )

        logger.info(f"输出审查：消息 #{pid} 已转入人工审查，来源 {item.umo}")

    def _clone_chain(self, chain: list, pid: str) -> tuple[list, Path | None]:
        """复制消息链，并把本地附件改存到插件数据目录。

        AstrBot 在一次事件处理结束后会清理事件相关的临时文件，
        如果不复制，等待人工审查期间附件可能已经被删除。
        """
        try:
            cloned = copy.deepcopy(chain)
        except Exception:
            logger.warning("输出审查：消息链无法深拷贝，改用浅拷贝，附件可能无法保留。")
            cloned = list(chain)

        # 去掉 AI 改写后留下的空文本段
        cloned = [
            comp
            for comp in cloned
            if not (isinstance(comp, Plain) and not getattr(comp, "text", "").strip())
        ]

        cache_dir = self._cache_root / pid
        used_cache = False
        for comp in cloned:
            for attr in ("file", "path", "url"):
                value = getattr(comp, attr, None)
                if not isinstance(value, str) or not value:
                    continue
                if not os.path.isfile(value):
                    continue
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    target = cache_dir / f"{uuid.uuid4().hex[:8]}{Path(value).suffix}"
                    shutil.copy2(value, target)
                    setattr(comp, attr, str(target))
                    used_cache = True
                except Exception:
                    logger.warning(f"输出审查：复制附件失败：{value}")

        return cloned, (cache_dir if used_cache else None)

    async def _watch_timeout(self, pid: str, seconds: int) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return

        item = self._pending.pop(pid, None)
        if item is None:
            return

        action = str(self._manual_cfg().get("timeout_action", "drop") or "drop").lower()
        if action == "send":
            ok = await self._deliver(item)
            note = "超时，已自动放行" if ok else "超时，自动放行失败"
        else:
            note = "超时，已自动丢弃"

        self._cleanup_item(item)
        logger.info(f"输出审查：待审消息 #{pid} {note}。")
        await self._notify_admins(f"【输出审查】#{pid} {note}。")

    # ------------------------------------------------------------- 放行/拒绝
    async def _approve(self, pid: str) -> tuple[bool, str]:
        item = self._pending.get(pid)
        if item is None:
            return False, f"没有编号为 #{pid} 的待审消息。"
        ok = await self._deliver(item)
        if not ok:
            # 发送失败时保留这条记录，管理员可以稍后重试
            return False, f"#{pid} 放行失败，请查看 AstrBot 日志后重试。"
        self._pending.pop(pid, None)
        self._cleanup_item(item)
        return True, f"已放行 #{pid}。"

    def _reject(self, pid: str) -> tuple[bool, str]:
        item = self._pending.pop(pid, None)
        if item is None:
            return False, f"没有编号为 #{pid} 的待审消息。"
        self._cleanup_item(item)
        logger.info(f"输出审查：#{pid} 已被人工拒绝。")
        return True, f"已拒绝 #{pid}。"

    async def _deliver(self, item: PendingItem) -> bool:
        if not item.chain:
            return True
        try:
            await self.context.send_message(
                item.umo,
                MessageChain(chain=item.chain),
            )
            return True
        except Exception:
            logger.error(
                f"输出审查：放行消息 #{item.pid} 失败\n" + traceback.format_exc()
            )
            return False

    def _cleanup_item(self, item: PendingItem) -> None:
        task = self._timeout_tasks.pop(item.pid, None)
        if task is not None and not task.done():
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
            if task is not current:
                task.cancel()
        if item.cache_dir is not None:
            shutil.rmtree(item.cache_dir, ignore_errors=True)

    # ------------------------------------------------------------- 通知
    def _admin_umos(self) -> list[str]:
        raw = self._manual_cfg().get("admin_umos") or []
        if isinstance(raw, str):
            raw = [raw]
        result: list[str] = []
        if isinstance(raw, list):
            for value in raw:
                if not isinstance(value, str):
                    continue
                umo = value.strip()
                if umo and umo not in result:
                    result.append(umo)
        return result

    async def _notify_admins(self, text: str) -> None:
        for umo in self._admin_umos():
            try:
                await self.context.send_message(umo, MessageChain().message(text))
            except Exception:
                logger.error(
                    f"输出审查：向 {umo} 发送通知失败\n" + traceback.format_exc()
                )

    def _format_review_request(self, item: PendingItem) -> str:
        preview = item.text.strip()
        if len(preview) > 1200:
            preview = (
                preview[:1200]
                + f"\n……（已截断，发送 /审查详情 {item.pid} 查看全文）"
            )
        reasons = "；".join(item.reasons) if item.reasons else "人工抽检"
        return (
            f"【输出审查】待处理消息 #{item.pid}\n"
            f"来源：{item.platform} / {item.umo}\n"
            f"发送者：{item.sender}（{item.sender_id}）\n"
            f"原因：{reasons}\n"
            "—— 待发送内容 ——\n"
            f"{preview}\n"
            "————————\n"
            f"放行：/审查通过 {item.pid}\n"
            f"拒绝：/审查拒绝 {item.pid} [原因]\n"
            f"详情：/审查详情 {item.pid}"
        )

    # ------------------------------------------------------------- 指令工具
    @staticmethod
    def _command_args(event: AstrMessageEvent) -> list[str]:
        text = str(event.message_str or "").strip()
        if not text:
            return []
        tokens = text.split()
        for index, token in enumerate(tokens):
            name = token.lstrip("/／!！").lower()
            if name in COMMAND_NAMES:
                return tokens[index + 1 :]
        return tokens[1:]

    @staticmethod
    def _normalize_pid(raw: str) -> str:
        return str(raw or "").strip().lstrip("#")

    def _pending_text(self) -> str:
        if not self._pending:
            return "当前没有待审消息。"
        lines = ["待审消息："]
        for item in sorted(self._pending.values(), key=lambda x: int(x.pid)):
            age = int(time.time() - item.created_at)
            preview = item.text.strip().replace("\n", " ")[:60]
            lines.append(f"#{item.pid}（{age}s）{preview}")
        lines.append("")
        lines.append("放行：/审查通过 <编号>　拒绝：/审查拒绝 <编号> [原因]")
        return "\n".join(lines)

    def _help_text(self) -> str:
        manual = self._manual_cfg()
        llm = self._llm_cfg()
        return (
            "【输出审查】\n"
            f"总开关：{'开启' if bool(self.config.get('enabled', True)) else '关闭'}"
            f"｜规则 {len(self._engine.rules)} 条"
            f"｜AI 审查：{'开启' if bool(llm.get('enabled', False)) else '关闭'}"
            f"｜人工审查：{manual.get('mode', 'disabled')}\n\n"
            "指令：\n"
            "/审查 —— 查看待审消息\n"
            "/审查通过 <编号> —— 放行\n"
            "/审查拒绝 <编号> [原因] —— 拒绝\n"
            "/审查详情 <编号> —— 查看全文\n"
            "/审查全部通过、/审查全部拒绝 —— 批量处理\n"
            "/审查测试 <文本> —— 预览规则命中情况（不调用 AI）\n"
            "/审查模式 <disabled|flagged|all> —— 切换人工审查\n"
            "/审查状态 —— 查看配置与运行状态\n"
            "/审查会话 —— 获取当前会话 UMO（用于配置通知）"
        )

    def _status_text(self) -> str:
        engine = self._engine
        manual = self._manual_cfg()
        llm = self._llm_cfg()
        enabled_rules = sum(
            1 for rule in engine.rules if rule.enabled and rule.patterns
        )
        lines = [
            "输出审查状态",
            f"总开关：{'开启' if bool(self.config.get('enabled', True)) else '关闭'}",
            f"审查范围：{self.config.get('rule_scope', 'all')}",
            f"规则：{len(engine.rules)} 条（可用 {enabled_rules} 条）",
            f"AI 审查：{'开启' if bool(llm.get('enabled', False)) else '关闭'}",
            f"人工审查：{manual.get('mode', 'disabled')}"
            f"（范围 {manual.get('scope', 'llm_only')}）",
            f"通知会话：{len(self._admin_umos())} 个",
            f"待审消息：{len(self._pending)} 条",
        ]
        if engine.errors:
            lines.append("规则问题：")
            lines.extend(f"- {message}" for message in engine.errors)
        return "\n".join(lines)

    # ----------------------------------------------------------------- 指令
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查", alias={"review"})
    async def cmd_review_home(self, event: AstrMessageEvent):
        """查看输出审查状态与待审消息。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        yield event.plain_result(
            self._help_text() + "\n\n" + self._pending_text()
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查通过", alias={"review_pass"})
    async def cmd_review_pass(self, event: AstrMessageEvent):
        """放行一条待审消息。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        args = self._command_args(event)
        if not args:
            yield event.plain_result("用法：/审查通过 <编号>")
            return
        ok, message = await self._approve(self._normalize_pid(args[0]))
        yield event.plain_result(("✅ " if ok else "❌ ") + message)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查拒绝", alias={"review_reject"})
    async def cmd_review_reject(self, event: AstrMessageEvent):
        """拒绝一条待审消息。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        args = self._command_args(event)
        if not args:
            yield event.plain_result("用法：/审查拒绝 <编号> [原因]")
            return
        pid = self._normalize_pid(args[0])
        item = self._pending.get(pid)
        ok, message = self._reject(pid)
        reason = " ".join(args[1:]).strip()
        if ok and reason:
            logger.info(f"输出审查：#{pid} 拒绝原因：{reason}")
        if ok and item is not None and self._manual_cfg().get("notify_on_reject"):
            try:
                await self.context.send_message(
                    item.umo,
                    MessageChain().message("这条回复未通过内容审查，已被拦截。"),
                )
            except Exception:
                logger.error("输出审查：发送拒绝通知失败\n" + traceback.format_exc())
        yield event.plain_result(("✅ " if ok else "❌ ") + message)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查详情", alias={"review_detail"})
    async def cmd_review_detail(self, event: AstrMessageEvent):
        """查看一条待审消息的完整内容。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        args = self._command_args(event)
        if not args:
            yield event.plain_result("用法：/审查详情 <编号>")
            return
        pid = self._normalize_pid(args[0])
        item = self._pending.get(pid)
        if item is None:
            yield event.plain_result(f"没有编号为 #{pid} 的待审消息。")
            return
        text = item.text
        if len(text) > 3000:
            text = text[:3000] + "\n……（已截断）"
        reasons = "；".join(item.reasons) if item.reasons else "人工抽检"
        yield event.plain_result(
            f"#{item.pid}\n"
            f"来源：{item.platform} / {item.umo}\n"
            f"发送者：{item.sender}（{item.sender_id}）\n"
            f"原因：{reasons}\n"
            "—— 内容 ——\n"
            f"{text}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查全部通过", alias={"review_pass_all"})
    async def cmd_review_pass_all(self, event: AstrMessageEvent):
        """放行全部待审消息。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        pids = sorted(self._pending, key=int)
        if not pids:
            yield event.plain_result("当前没有待审消息。")
            return
        lines = []
        for pid in pids:
            ok, message = await self._approve(pid)
            lines.append(("✅ " if ok else "❌ ") + message)
            await asyncio.sleep(0.2)
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查全部拒绝", alias={"review_reject_all"})
    async def cmd_review_reject_all(self, event: AstrMessageEvent):
        """拒绝全部待审消息。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        pids = sorted(self._pending, key=int)
        if not pids:
            yield event.plain_result("当前没有待审消息。")
            return
        lines = [self._reject(pid)[1] for pid in pids]
        yield event.plain_result("\n".join("✅ " + line for line in lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查测试", alias={"review_test"})
    async def cmd_review_test(self, event: AstrMessageEvent):
        """用一段文本试跑规则，方便调试配置。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        args = self._command_args(event)
        if not args:
            yield event.plain_result("用法：/审查测试 <文本>")
            return
        text = " ".join(args)
        try:
            self._refresh_engine()
        except Exception:
            logger.error("输出审查：加载规则失败\n" + traceback.format_exc())
            yield event.plain_result("规则加载失败，请查看日志。")
            return
        outcome = self._engine.apply(text)
        lines = [f"命中规则：{len(outcome.hits)} 条"]
        for hit in outcome.hits:
            lines.append(f"- {hit.rule}（{hit.action}）：{hit.matched}")
        if outcome.reasons:
            lines.append("原因：" + "；".join(outcome.reasons))
        if outcome.blocked:
            verdict = "拦截"
        elif outcome.flagged:
            verdict = "标记可疑"
        else:
            verdict = "放行"
        lines.append("结论：" + verdict)
        if outcome.text != text:
            lines.append("替换后：" + outcome.text)
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查模式", alias={"review_mode"})
    async def cmd_review_mode(self, event: AstrMessageEvent):
        """查看或切换人工审查模式。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        args = self._command_args(event)
        current = str(self._manual_cfg().get("mode", "disabled") or "disabled")
        if not args:
            yield event.plain_result(
                f"当前人工审查模式：{current}\n"
                "用法：/审查模式 <disabled|flagged|all>\n"
                "disabled：关闭人工审查；flagged：仅标记的消息转人工；all：全部转人工。"
            )
            return
        mode = args[0].strip().lower()
        if mode not in MODES:
            yield event.plain_result("模式只能是 disabled / flagged / all。")
            return
        manual = self.config.get("manual")
        if not isinstance(manual, dict):
            manual = {}
            self.config["manual"] = manual
        manual["mode"] = mode
        try:
            self.config.save_config()
        except Exception:
            logger.error("输出审查：保存配置失败\n" + traceback.format_exc())
        yield event.plain_result(f"人工审查模式已切换为 {mode}。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查状态", alias={"review_status"})
    async def cmd_review_status(self, event: AstrMessageEvent):
        """查看插件配置与运行状态。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        yield event.plain_result(self._status_text())

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审查会话", alias={"review_umo"})
    async def cmd_review_umo(self, event: AstrMessageEvent):
        """获取当前会话的 UMO，用于配置审查通知。"""
        event.set_extra(BYPASS_EXTRA_KEY, True)
        yield event.plain_result(
            f"当前会话 UMO：\n{event.unified_msg_origin}\n\n"
            "把它加入插件配置 manual.admin_umos 后，审查请求就会推送到这里。"
        )

    # ------------------------------------------------------------- 生命周期
    async def terminate(self) -> None:
        """插件卸载时清理挂起消息与定时任务。"""
        pending = list(self._pending.values())
        self._pending.clear()
        for item in pending:
            self._cleanup_item(item)
        logger.info(f"输出审查插件已卸载，丢弃 {len(pending)} 条未处理的人工审查消息。")
