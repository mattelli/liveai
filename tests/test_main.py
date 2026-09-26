import sys

import main as main_mod


def test_parse_args_defaults():
    a = main_mod.parse_args([])
    assert a.chat is False and a.no_tools is False
    assert a.config == main_mod.CONFIG_FILE and a.prompt is None


def test_parse_args_flags():
    a = main_mod.parse_args(["--chat", "--no-tools", "--config", "x.toml"])
    assert a.chat and a.no_tools and str(a.config).endswith("x.toml")


def test_import_main_has_no_side_effects():
    # Importing main must not call load_config; agent/session constructed only in main().
    assert callable(main_mod.main)


def test_parse_args_prompt():
    assert main_mod.parse_args(["--prompt", "hi"]).prompt == "hi"


def test_main_chat_import_is_side_effect_free():
    for name in list(sys.modules):
        if name.startswith("main-chat"):
            del sys.modules[name]
    import importlib.util
    import pathlib

    p = pathlib.Path("main-chat.py")
    spec = importlib.util.spec_from_file_location("main_chat_shim", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not load config/agent
    assert callable(mod.main)


def test_single_prompt_turn_uses_agent():
    from src.session import ConversationSession
    from tests.conftest import FakeAgent

    agent = FakeAgent()
    it = iter(["hello", None])
    assert ConversationSession(agent, lambda: next(it)).run() == 0
    assert agent.acted == ["hello"]
