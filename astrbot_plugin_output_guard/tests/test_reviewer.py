"""reviewer.py 的单元测试（不依赖 AstrBot）。"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewer import RuleEngine, parse_verdict, split_patterns  # noqa: E402


def engine(*rules, bypass=None):
    return RuleEngine.from_config(list(rules), bypass)


class KeywordRuleTest(unittest.TestCase):
    def test_block(self):
        result = engine(
            {"name": "禁词", "pattern": "傻", "action": "block"}
        ).apply("你这个傻东西")
        self.assertTrue(result.blocked)
        self.assertIn("禁词", result.reasons[0])

    def test_block_stops_following_rules(self):
        result = engine(
            {"name": "拦截", "pattern": "a", "action": "block"},
            {"name": "替换", "pattern": "b", "action": "replace", "replacement": "X"},
        ).apply("ab")
        self.assertTrue(result.blocked)
        self.assertEqual(result.text, "ab")
        self.assertEqual(len(result.hits), 1)

    def test_case_insensitive_by_default(self):
        result = engine(
            {"name": "词", "pattern": "BadWord", "action": "flag"}
        ).apply("a badword here")
        self.assertTrue(result.flagged)

    def test_case_sensitive(self):
        result = engine(
            {
                "name": "词",
                "pattern": "BadWord",
                "action": "flag",
                "case_sensitive": True,
            }
        ).apply("a badword here")
        self.assertFalse(result.flagged)

    def test_replace_is_case_insensitive(self):
        result = engine(
            {
                "name": "替换",
                "pattern": "secret",
                "action": "replace",
                "replacement": "[已隐藏]",
            }
        ).apply("A SECRET value, secret again")
        self.assertEqual(result.text, "A [已隐藏] value, [已隐藏] again")

    def test_replace_does_not_flag(self):
        result = engine(
            {
                "name": "替换",
                "pattern": "x",
                "action": "replace",
                "replacement": "y",
            }
        ).apply("xx")
        self.assertFalse(result.flagged)
        self.assertEqual(result.text, "yy")

    def test_disabled_rule_is_ignored(self):
        result = engine(
            {"name": "关", "pattern": "x", "action": "block", "enabled": False}
        ).apply("x")
        self.assertFalse(result.blocked)

    def test_empty_pattern_is_ignored(self):
        result = engine({"name": "空", "pattern": "", "action": "block"}).apply("abc")
        self.assertFalse(result.blocked)


class RegexRuleTest(unittest.TestCase):
    def test_regex_block(self):
        result = engine(
            {
                "name": "手机号",
                "match_type": "regex",
                "pattern": r"1[3-9]\d{9}",
                "action": "block",
            }
        ).apply("我的号码是 13812345678")
        self.assertTrue(result.blocked)

    def test_regex_replace_with_backreference(self):
        result = engine(
            {
                "name": "脱敏",
                "match_type": "regex",
                "pattern": r"(1[3-9]\d)\d{4}(\d{4})",
                "action": "replace",
                "replacement": r"\1****\2",
            }
        ).apply("13812345678")
        self.assertEqual(result.text, "138****5678")

    def test_invalid_regex_is_reported_not_raised(self):
        rule_engine = engine(
            {
                "name": "坏正则",
                "match_type": "regex",
                "pattern": "(",
                "action": "block",
            }
        )
        self.assertEqual(len(rule_engine.errors), 1)
        self.assertFalse(rule_engine.apply("anything").blocked)

    def test_invalid_regex_does_not_break_other_rules(self):
        rule_engine = engine(
            {"name": "坏", "match_type": "regex", "pattern": "[", "action": "block"},
            {"name": "好", "pattern": "bad", "action": "block"},
        )
        self.assertTrue(rule_engine.apply("bad").blocked)


class BypassTest(unittest.TestCase):
    def test_bypass_pattern(self):
        rule_engine = engine(
            {"name": "拦截", "pattern": "x", "action": "block"},
            bypass=[r"^\[系统通知\]"],
        )
        self.assertTrue(rule_engine.is_bypassed("[系统通知] 维护完成"))
        self.assertFalse(rule_engine.is_bypassed("普通消息 x"))

    def test_invalid_bypass_is_reported(self):
        rule_engine = engine(bypass=["("])
        self.assertEqual(len(rule_engine.errors), 1)
        self.assertFalse(rule_engine.is_bypassed("abc"))


class MultiPatternTest(unittest.TestCase):
    def test_multiple_keywords_by_newline(self):
        result = engine(
            {"name": "多词", "pattern": "foo\nbar", "action": "block"}
        ).apply("xx bar yy")
        self.assertTrue(result.blocked)
        self.assertIn("bar", result.reasons[0])

    def test_multiple_keywords_by_pipe(self):
        result = engine(
            {"name": "多词", "pattern": "foo|bar", "action": "block"}
        ).apply("foo")
        self.assertTrue(result.blocked)

    def test_fullwidth_pipe_also_splits_keywords(self):
        result = engine(
            {"name": "多词", "pattern": "foo｜bar", "action": "block"}
        ).apply("bar")
        self.assertTrue(result.blocked)

    def test_spaces_are_not_separators(self):
        result = engine(
            {"name": "短语", "pattern": "bad word", "action": "flag"}
        ).apply("this is a bad word")
        self.assertTrue(result.flagged)

    def test_regex_pipe_keeps_alternation(self):
        result = engine(
            {
                "name": "正则",
                "match_type": "regex",
                "pattern": r"foo|bar",
                "action": "replace",
                "replacement": "X",
            }
        ).apply("foo and bar")
        self.assertEqual(result.text, "X and X")

    def test_multiple_regex_lines(self):
        result = engine(
            {
                "name": "正则多行",
                "match_type": "regex",
                "pattern": r"1[3-9]\d{9}" + "\n" + r"\d{17}[\dXx]",
                "action": "block",
            }
        ).apply("身份证 110101199001011234")
        self.assertTrue(result.blocked)

    def test_partially_invalid_regex_keeps_valid_pattern(self):
        rule_engine = engine(
            {
                "name": "部分坏",
                "match_type": "regex",
                "pattern": "(\nbad",
                "action": "block",
            }
        )
        self.assertEqual(len(rule_engine.errors), 1)
        self.assertTrue(rule_engine.apply("bad").blocked)

    def test_keyword_deduplicates_and_trims(self):
        rule = RuleEngine.from_config(
            [{"name": "重复", "pattern": "  x  \n x \ny", "action": "flag"}]
        ).rules[0]
        self.assertEqual(rule.patterns, ["x", "y"])

    def test_split_patterns_accepts_list(self):
        self.assertEqual(
            split_patterns(["a", "b", "a"], "keyword"),
            ["a", "b"],
        )
        self.assertEqual(split_patterns(None), [])


class RewriteTest(unittest.TestCase):
    def test_rewrite_only_applies_replace_rules(self):
        rule_engine = engine(
            {"name": "替换", "pattern": "a", "action": "replace", "replacement": "b"},
            {"name": "标记", "pattern": "b", "action": "flag"},
        )
        self.assertEqual(rule_engine.rewrite("aaa"), "bbb")

    def test_rewrite_ignores_block(self):
        rule_engine = engine(
            {"name": "拦截", "pattern": "a", "action": "block"},
            {"name": "替换", "pattern": "b", "action": "replace", "replacement": "c"},
        )
        self.assertEqual(rule_engine.rewrite("ab"), "ac")


class FromConfigTest(unittest.TestCase):
    def test_json_string(self):
        raw = json.dumps([{"name": "n", "pattern": "x", "action": "block"}])
        self.assertTrue(RuleEngine.from_config(raw).apply("x").blocked)

    def test_bad_json_string(self):
        rule_engine = RuleEngine.from_config("{not json")
        self.assertEqual(rule_engine.rules, [])

    def test_ignores_non_dict_items(self):
        rule_engine = RuleEngine.from_config(["x", 1, None])
        self.assertEqual(rule_engine.rules, [])

    def test_invalid_action_falls_back_to_flag(self):
        rule_engine = RuleEngine.from_config(
            [{"name": "n", "pattern": "x", "action": "explode"}]
        )
        self.assertEqual(rule_engine.rules[0].action, "flag")


class ParseVerdictTest(unittest.TestCase):
    def test_plain_json(self):
        verdict = parse_verdict(
            '{"pass": false, "reason": "包含隐私信息", "rewrite": "改写后的内容"}'
        )
        self.assertIsNotNone(verdict)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.reason, "包含隐私信息")
        self.assertEqual(verdict.rewrite, "改写后的内容")

    def test_markdown_fence(self):
        raw = '```json\n{"pass": true, "reason": "ok"}\n```'
        verdict = parse_verdict(raw)
        self.assertIsNotNone(verdict)
        self.assertTrue(verdict.passed)

    def test_nested_json_with_extra_text(self):
        raw = '结论如下：\n{"pass": true, "reason": "没有问题"}\n以上。'
        verdict = parse_verdict(raw)
        self.assertIsNotNone(verdict)
        self.assertTrue(verdict.passed)

    def test_string_boolean(self):
        verdict = parse_verdict('{"pass": "false", "reason": "no"}')
        self.assertIsNotNone(verdict)
        self.assertFalse(verdict.passed)

    def test_risk_level(self):
        verdict = parse_verdict('{"risk_level": "high", "reason": "违规"}')
        self.assertIsNotNone(verdict)
        self.assertFalse(verdict.passed)

    def test_invalid_returns_none(self):
        self.assertIsNone(parse_verdict("模型没有返回 JSON"))
        self.assertIsNone(parse_verdict(""))
        self.assertIsNone(parse_verdict(None))
        self.assertIsNone(parse_verdict('{"reason": "没有结论字段"}'))

    def test_json_array_returns_none(self):
        self.assertIsNone(parse_verdict("[1, 2, 3]"))


if __name__ == "__main__":
    unittest.main()
