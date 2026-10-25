"""用最小桩模块加载 main.py，验证插件接线与核心流程。

这些测试不依赖真实 AstrBot，只验证：
- 模块能正常导入、Star 子类能实例化；
- 指令参数解析、文本提取、改写；
- 规则的拦截/替换；
- 人工审查的挂起、通知、放行、超时流程。
"""

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))


# --------------------------------------------------------------------- 桩对象
class StubPlain:
    def __init__(self, text=""):
        self.text = text

    def __eq__(self, other):
        return isinstance(other, StubPlain) and self.text == other.text


class StubImage:
    def __init__(self, file=""):
        self.file = file


class StubMessageChain:
    def __init__(self, chain=None):
        self.chain = chain if chain is not None else []

    def message(self, text):
        self.chain.append(StubPlain(text))
        return self


class StubResultContentType:
    LLM_RESULT = "llm_result"
    AGENT_RUNNER_ERROR = "agent_runner_error"
    GENERAL_RESULT = "general_result"
    STREAMING_RESULT = "streaming_result"
    STREAMING_FINISH = "streaming_finish"


class StubPermissionType:
    ADMIN = "admin"


class StubFilter:
    PermissionType = StubPermissionType

    @staticmethod
    def on_decorating_result(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    @staticmethod
    def command(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    @staticmethod
    def permission_type(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


class StubStar:
    def __init__(self, context):
        self.context = context
        self.name = "astrbot_plugin_output_guard"


class StubConfig(dict):
    def save_config(self):
        pass


class StubContext:
    def __init__(self):
        self.sent = []

    async def send_message(self, umo, chain):
        self.sent.append((umo, chain))

    async def get_current_chat_provider_id(self, umo=""):
        return "stub-provider"

    async def llm_generate(self, **kwargs):
        raise RuntimeError("stub 不调用模型")


class StubEvent:
    def __init__(self, message_str="", umo="test:friend:2"):
        self.message_str = message_str
        self.unified_msg_origin = umo
        self._result = None
        self._extras = {}
        self.stopped = False

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def stop_event(self):
        self.stopped = True

    def get_platform_name(self):
        return "stub"

    def get_sender_name(self):
        return "tester"

    def get_sender_id(self):
        return "10001"


class StubResult:
    def __init__(self, chain, content_type=StubResultContentType.GENERAL_RESULT):
        self.chain = chain
        self.result_content_type = content_type

    def is_model_result(self):
        return self.result_content_type in (
            StubResultContentType.LLM_RESULT,
            StubResultContentType.AGENT_RUNNER_ERROR,
        )


def _install_stubs():
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = StubConfig
    api.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.AstrMessageEvent = StubEvent
    event_mod.MessageChain = StubMessageChain
    event_mod.ResultContentType = StubResultContentType
    event_mod.filter = StubFilter

    components = types.ModuleType("astrbot.api.message_components")
    components.Plain = StubPlain

    star = types.ModuleType("astrbot.api.star")
    star.Context = StubContext
    star.Star = StubStar

    utils = types.ModuleType("astrbot.core.utils")
    path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")
    path_mod.get_astrbot_data_path = lambda: tempfile.mkdtemp(
        prefix="output_guard_test_"
    )

    modules = {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event_mod,
        "astrbot.api.message_components": components,
        "astrbot.api.star": star,
        "astrbot.core": types.ModuleType("astrbot.core"),
        "astrbot.core.utils": utils,
        "astrbot.core.utils.astrbot_path": path_mod,
    }
    sys.modules.update(modules)


_install_stubs()
import main as plugin_main  # noqa: E402


def make_config(**overrides):
    config = {
        "enabled": True,
        "rule_scope": "all",
        "rules": [],
        "bypass_patterns": [],
        "llm_review": {"enabled": False},
        "manual": {
            "mode": "disabled",
            "scope": "all",
            "admin_umos": [],
            "timeout_seconds": 0,
            "timeout_action": "drop",
            "notify_on_reject": False,
            "fallback_when_no_admin": "send",
        },
        "debug": False,
    }
    config.update(overrides)
    return StubConfig(config)


def make_plugin(**overrides):
    return plugin_main.OutputGuardPlugin(StubContext(), make_config(**overrides))


class WiringTest(unittest.TestCase):
    def test_plugin_can_instantiate(self):
        plugin = make_plugin()
        self.assertIsInstance(plugin, StubStar)
        self.assertEqual(plugin._engine.rules, [])

    def test_command_args(self):
        plugin = make_plugin()
        cases = {
            "/审查通过 3": ["3"],
            "@bot /审查通过 3": ["3"],
            "/审查拒绝 3 含隐私内容": ["3", "含隐私内容"],
            "审查测试 你好 世界": ["你好", "世界"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    plugin._command_args(StubEvent(text)),
                    expected,
                )

    def test_text_views(self):
        plugin = make_plugin()
        self.assertIn("输出审查状态", plugin._status_text())
        self.assertIn("指令", plugin._help_text())
        self.assertIn("没有待审消息", plugin._pending_text())


class ChainHelperTest(unittest.TestCase):
    def test_extract_text_and_media(self):
        plugin = make_plugin()
        chain = [StubPlain("你好"), StubImage("x.png"), StubPlain("世界")]
        self.assertEqual(plugin._extract_text(chain), "你好世界")
        self.assertTrue(plugin._has_media(chain))
        self.assertFalse(plugin._has_media([StubPlain("a")]))

    def test_llm_rewrite_single_plain(self):
        chain = [StubPlain("旧内容")]
        plugin_main.OutputGuardPlugin._apply_llm_rewrite(chain, "新内容")
        self.assertEqual(chain[0].text, "新内容")

    def test_llm_rewrite_multiple_plains(self):
        chain = [StubPlain("a"), StubImage("x"), StubPlain("b")]
        plugin_main.OutputGuardPlugin._apply_llm_rewrite(chain, "合并")
        self.assertEqual(chain[0].text, "合并")
        self.assertEqual(chain[2].text, "")


class ReviewLogicTest(unittest.TestCase):
    def test_block_rule(self):
        plugin = make_plugin(
            rules=[{"name": "禁词", "pattern": "bad", "action": "block"}]
        )
        event = StubEvent("this is bad")
        outcome = asyncio.run(plugin._review(event, "this is bad", True))
        self.assertTrue(outcome.blocked)
        self.assertIn("禁词", outcome.reasons[0])

    def test_replace_rule(self):
        plugin = make_plugin(
            rules=[
                {
                    "name": "脱敏",
                    "pattern": "secret",
                    "action": "replace",
                    "replacement": "[hidden]",
                }
            ]
        )
        event = StubEvent("a secret value")
        outcome = asyncio.run(plugin._review(event, "a secret value", True))
        self.assertFalse(outcome.blocked)
        self.assertEqual(outcome.text, "a [hidden] value")

    def test_suppress_clears_and_stops(self):
        plugin = make_plugin()
        event = StubEvent()
        result = StubResult([StubPlain("hello")])
        plugin._suppress(event, result, ["test"])
        self.assertEqual(result.chain, [])
        self.assertTrue(event.stopped)

    def test_llm_scope_skips_plain_results(self):
        plugin = make_plugin(
            rule_scope="llm_only",
            rules=[{"name": "禁词", "pattern": "bad", "action": "block"}],
        )
        event = StubEvent("bad")
        result = StubResult([StubPlain("bad")])
        event.set_result(result)
        asyncio.run(plugin.review_outgoing_message(event))
        self.assertFalse(event.stopped)
        self.assertEqual(result.chain[0].text, "bad")

    def test_llm_scope_blocks_model_results(self):
        plugin = make_plugin(
            rule_scope="llm_only",
            rules=[{"name": "禁词", "pattern": "bad", "action": "block"}],
        )
        event = StubEvent("bad")
        result = StubResult(
            [StubPlain("bad")], content_type=StubResultContentType.LLM_RESULT
        )
        event.set_result(result)
        asyncio.run(plugin.review_outgoing_message(event))
        self.assertTrue(event.stopped)

    def test_bypass_extra_skips_review(self):
        plugin = make_plugin(
            rules=[{"name": "禁词", "pattern": "bad", "action": "block"}]
        )
        event = StubEvent("bad")
        event.set_extra(plugin_main.BYPASS_EXTRA_KEY, True)
        result = StubResult([StubPlain("bad")], content_type=StubResultContentType.LLM_RESULT)
        event.set_result(result)
        asyncio.run(plugin.review_outgoing_message(event))
        self.assertFalse(event.stopped)
        self.assertEqual(result.chain[0].text, "bad")

    def test_llm_failure_flags_when_fail_closed(self):
        plugin = make_plugin(
            llm_review={
                "enabled": True,
                "provider_id": "",
                "prompt": "检查：{text}",
                "fail_open": False,
                "action_on_reject": "flag",
                "apply_rewrite": False,
            }
        )
        outcome = asyncio.run(plugin._review(StubEvent("hi"), "hi", True))
        self.assertTrue(outcome.flagged)
        self.assertIn("AI 审查", outcome.reasons[-1])


class ManualFlowTest(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    async def _hold_and_approve(self):
        plugin = make_plugin(
            manual={
                "mode": "all",
                "scope": "all",
                "admin_umos": ["test:friend:1"],
                "timeout_seconds": 0,
                "timeout_action": "drop",
                "notify_on_reject": False,
                "fallback_when_no_admin": "send",
            }
        )
        event = StubEvent("需要审查的内容", umo="test:friend:2")
        result = StubResult([StubPlain("需要审查的内容")])
        event.set_result(result)

        await plugin.review_outgoing_message(event)

        self.assertTrue(event.stopped)
        self.assertEqual(result.chain, [])
        self.assertIn("1", plugin._pending)
        self.assertEqual(len(plugin.context.sent), 1)
        self.assertEqual(plugin.context.sent[0][0], "test:friend:1")

        ok, message = await plugin._approve("1")
        self.assertTrue(ok)
        self.assertIn("已放行", message)
        self.assertEqual(plugin._pending, {})
        self.assertEqual(plugin.context.sent[-1][0], "test:friend:2")
        self.assertEqual(len(plugin.context.sent[-1][1].chain), 1)
        await plugin.terminate()

    def test_hold_notify_and_approve(self):
        self._run(self._hold_and_approve())

    async def _reject(self):
        plugin = make_plugin(
            manual={
                "mode": "all",
                "scope": "all",
                "admin_umos": ["test:friend:1"],
                "timeout_seconds": 0,
                "timeout_action": "drop",
                "notify_on_reject": False,
                "fallback_when_no_admin": "send",
            }
        )
        event = StubEvent("违规内容")
        result = StubResult([StubPlain("违规内容")])
        event.set_result(result)
        await plugin.review_outgoing_message(event)
        ok, message = plugin._reject("1")
        self.assertTrue(ok)
        self.assertIn("已拒绝", message)
        self.assertEqual(plugin._pending, {})
        await plugin.terminate()

    def test_reject(self):
        self._run(self._reject())

    async def _no_admin_fallback_send(self):
        plugin = make_plugin(
            manual={
                "mode": "all",
                "scope": "all",
                "admin_umos": [],
                "timeout_seconds": 0,
                "timeout_action": "drop",
                "notify_on_reject": False,
                "fallback_when_no_admin": "send",
            }
        )
        event = StubEvent("普通内容")
        result = StubResult([StubPlain("普通内容")])
        event.set_result(result)
        await plugin.review_outgoing_message(event)
        self.assertFalse(event.stopped)
        self.assertEqual(result.chain[0].text, "普通内容")
        self.assertEqual(plugin._pending, {})
        await plugin.terminate()

    def test_no_admin_fallback_send(self):
        self._run(self._no_admin_fallback_send())

    async def _timeout_drop(self):
        plugin = make_plugin()
        item = plugin_main.PendingItem(
            pid="9",
            umo="test:friend:2",
            platform="stub",
            sender="tester",
            sender_id="10001",
            text="hello",
            chain=[StubPlain("hello")],
            reasons=["test"],
            created_at=0.0,
        )
        plugin._pending["9"] = item
        await plugin._watch_timeout("9", 0)
        self.assertNotIn("9", plugin._pending)
        # 没有配置通知会话，因此不应产生任何发送
        self.assertEqual(plugin.context.sent, [])

    def test_timeout_drop(self):
        self._run(self._timeout_drop())


if __name__ == "__main__":
    unittest.main()
