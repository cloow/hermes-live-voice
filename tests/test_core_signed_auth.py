"""Issue #20: Hermes signs the Codex requests; the plugin never holds or refreshes the token.

Requested by the Hermes maintainers (@teknium1). Codex refresh tokens are single-use, so a
second refresher outside Hermes's lock signs the user out; the old panel logout also deleted
~/.codex/auth.json, which signed the Codex CLI out too. Everything here is offline.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

try:  # Hermes ships ruamel via hermes_yaml, not PyYAML; CI installs PyYAML instead
    import hermes_yaml as yaml
except ImportError:
    import yaml

from test_plugin_api import PLUGIN_API, REPO, VENDOR, load_unit

TALK_AUTH = VENDOR / "talk_auth.py"
TALK_WIRE = VENDOR / "talk_wire.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def _expired_auth_json(home: Path) -> Path:
    import base64

    claims = base64.urlsafe_b64encode(json.dumps({"exp": 1}).encode()).decode().rstrip("=")
    path = home / "auth.json"
    path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {"access_token": f"h.{claims}.s", "refresh_token": "single-use", "account_id": "acct"},
    }), encoding="utf-8")
    return path


# ── 1. no refresh, no write-back ─────────────────────────────────────────────

def test_vendored_talk_auth_has_no_refresh_or_write_back():
    src = TALK_AUTH.read_text(encoding="utf-8")
    for forbidden in ("oauth/token", "grant_type", "_refresh_codex_credential", "_write_auth_json",
                      "os.replace", "import httpx", "app_EMoamEEZ73f0CkXaXp7hrann"):
        assert forbidden not in src, forbidden


def test_resolve_auth_codex_lane_carries_no_token_and_leaves_auth_json_untouched(tmp_path):
    talk_auth = _module(TALK_AUTH, "talk_auth_under_test")
    path = _expired_auth_json(tmp_path)
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    auth = talk_auth.resolve_auth(env={}, codex_home=tmp_path)
    assert auth.source == talk_auth.SOURCE_CODEX_OAUTH
    assert auth.token == ""
    pinned = talk_auth.resolve_auth(env={"TALK_PREFER_CODEX_OAUTH": "true", "OPENAI_API_KEY": "sk-x"},
                                    codex_home=tmp_path)
    assert pinned.source == talk_auth.SOURCE_CODEX_OAUTH and pinned.token == ""
    diag = talk_auth.auth_diagnostic(env={}, codex_home=tmp_path)
    assert diag["codex_oauth"] == "expired"

    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["auth.json"]


def test_api_key_lanes_still_win_by_default():
    talk_auth = _module(TALK_AUTH, "talk_auth_under_test")
    assert talk_auth.resolve_auth(env={"OPENAI_API_KEY": "sk-env"}).source == talk_auth.SOURCE_ENV
    assert talk_auth.resolve_auth(env={"TALK_OPENAI_API_KEY": "sk-talk"}).token == "sk-talk"
    with pytest.raises(talk_auth.TalkAuthError):
        talk_auth.resolve_auth(env={"TALK_OPENAI_API_KEY": "  "})


# ── 2. the mint goes through Hermes ──────────────────────────────────────────

class _NotSignedIn(RuntimeError):
    pass


def _ppr(response=None, raise_exc=None, calls=None):
    def credentialed_provider_request(provider, method, url, **kw):
        if calls is not None:
            calls.append((provider, method, url, kw))
        if raise_exc:
            raise raise_exc
        return response
    return SimpleNamespace(credentialed_provider_request=credentialed_provider_request,
                           ProviderNotSignedIn=_NotSignedIn)


def _post_ns(ppr):
    ns = load_unit(PLUGIN_API, "_post_client_secret")
    ns.update(talk_wire=_module(TALK_WIRE, "talk_wire_under_test"), _provider_requests=lambda: ppr)
    return ns


def _response(status, body):
    raw = json.dumps(body).encode()
    return SimpleNamespace(status=status, text=raw.decode(), json=lambda: json.loads(raw))


def test_post_client_secret_asks_hermes_to_sign_it():
    calls = []
    ns = _post_ns(_ppr(_response(200, {"value": "ek_x", "expires_at": 10}), calls=calls))
    out = ns["_post_client_secret"]({"type": "realtime", "model": "m"})
    assert out == {"value": "ek_x", "expires_at": 10}
    (provider, method, url, kw), = calls
    assert (provider, method, url) == ("openai-codex", "POST", "https://api.openai.com/v1/realtime/client_secrets")
    assert kw["json"] == {"session": {"type": "realtime", "model": "m"}}
    assert "headers" not in kw  # no Authorization from the plugin, ever


def test_post_client_secret_maps_not_signed_in_and_http_errors():
    wire = _module(TALK_WIRE, "talk_wire_under_test")
    ns = _post_ns(_ppr(raise_exc=_NotSignedIn("Run `hermes auth add openai-codex`")))
    ns["talk_wire"] = wire
    with pytest.raises(wire.TalkUpstreamError, match="hermes auth"):
        ns["_post_client_secret"]({})
    ns = _post_ns(_ppr(_response(401, {"error": "nope"})))
    ns["talk_wire"] = wire
    with pytest.raises(wire.TalkUpstreamError, match="401"):
        ns["_post_client_secret"]({})


def test_post_client_secret_on_old_hermes_says_update():
    ns = _post_ns(None)
    with pytest.raises(Exception, match="update Hermes"):
        ns["_post_client_secret"]({})


def test_mint_for_codex_lane_never_sends_a_token():
    wire = _module(TALK_WIRE, "talk_wire_under_test")
    posted = []
    ns = load_unit(PLUGIN_API, "_mint_for")
    ns.update(
        talk_tools=SimpleNamespace(default_talk_tools=lambda: []),
        _send_to_chat_tool=lambda: {"name": "send_to_chat"},
        talk_host=SimpleNamespace(host=lambda: SimpleNamespace(identity_sections=lambda: {})),
        talk_identity=SimpleNamespace(build_instructions=lambda sections, **kw: "You are Hermes, speaking live."),
        talk_capabilities=SimpleNamespace(instruction_section=lambda: ""),
        _language_directive=lambda language: "", _voice_policy=lambda language: "",
        talk_auth=SimpleNamespace(resolve_auth=lambda: SimpleNamespace(token="", source="codex-oauth")),
        talk_config=SimpleNamespace(talk_model=lambda: "gpt-realtime-2.1"),
        talk_wire=SimpleNamespace(
            build_session_payload=wire.build_session_payload,
            parse_client_secret=wire.parse_client_secret,
            TalkSessionDescriptor=wire.TalkSessionDescriptor,
            OPENAI_REALTIME_OFFER_URL=wire.OPENAI_REALTIME_OFFER_URL,
            mint_ephemeral_session=lambda **kw: pytest.fail("bearer mint used on the Hermes lane"),
        ),
        _post_client_secret=lambda session: posted.append(session) or {"value": "ek_1", "expires_at": 5},
    )
    descriptor, auth = ns["_mint_for"](None, "marin", False)
    assert descriptor.client_secret == "ek_1" and descriptor.expires_at_ms == 5000
    assert posted[0]["model"] == "gpt-realtime-2.1" and posted[0]["audio"]["output"] == {"voice": "marin"}


# ── 3. usage from Hermes, logout never deletes ───────────────────────────────

def test_codex_usage_uses_hermes_read_only_fetcher(monkeypatch):
    seen = {}

    def fetch_account_usage(provider, *, read_only=False, **kw):
        seen.update(provider=provider, read_only=read_only)
        reset = datetime(2030, 1, 1, tzinfo=timezone.utc)
        return SimpleNamespace(
            unavailable_reason=None, plan="Pro",
            windows=(SimpleNamespace(label="Session", used_percent=12.4, reset_at=reset, scope="account"),
                     SimpleNamespace(label="Weekly", used_percent=40.0, reset_at=None, scope="account"),
                     SimpleNamespace(label="Opus", used_percent=99.0, reset_at=None, scope="model")),
        )

    agent = types.ModuleType("agent")
    usage = types.ModuleType("agent.account_usage")
    usage.fetch_account_usage = fetch_account_usage
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.account_usage", usage)
    ns = load_unit(PLUGIN_API, "_codex_usage_snapshot")
    helpers = load_unit(PLUGIN_API, "_window_minutes")
    helpers["_CODEX_WINDOWS"] = {"session": ("5h", 300), "weekly": ("semanal", 7 * 1440)}
    ns["_window_minutes"] = helpers["_window_minutes"]
    out = ns["_codex_usage_snapshot"]()
    assert seen == {"provider": "openai-codex", "read_only": True}
    assert out == {"ok": True, "plan": "Pro", "windows": {
        "5h": {"usedPercent": 12, "windowMinutes": 300, "resetsAt": 1893456000},
        "semanal": {"usedPercent": 40, "windowMinutes": 10080, "resetsAt": None},
    }}


def test_backend_has_no_direct_usage_or_bearer_calls():
    src = PLUGIN_API.read_text(encoding="utf-8")
    for forbidden in ("wham/usage", "api/codex/usage", '"User-Agent": "codex-cli"', "Bearer {auth.token}",
                      "path.unlink()"):
        assert forbidden not in src, forbidden


def test_logout_never_deletes_auth_json(tmp_path):
    path = _expired_auth_json(tmp_path)
    ns = load_unit(PLUGIN_API, "codex_logout")
    ns["_HERMES_LOGOUT_CMD"] = "hermes auth logout openai-codex"
    out = asyncio.run(ns["codex_logout"]())
    assert path.exists()
    assert out["ok"] is False and out["command"] == "hermes auth logout openai-codex"
    js = (REPO / "desktop" / "plugin.js").read_text(encoding="utf-8")
    assert "hermes auth logout openai-codex" in js
    assert "/codex/logout" not in js


# ── 4. manifest declares the sign-in ─────────────────────────────────────────

def test_manifest_declares_requires_auth_and_hermes_floor():
    manifest = yaml.safe_load((REPO / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["requires_auth"] == ["openai-codex"]
    floor = manifest["requires_hermes"]
    assert floor.startswith(">=")
    assert tuple(int(x) for x in floor[2:].split(".")) >= (0, 21, 7)


# ── 5. one refresher in-process even with hermes-talk installed ──────────────

def test_vendored_talk_auth_wins_over_installed_hermes_talk(tmp_path):
    talk = tmp_path / ".hermes" / "plugins" / "hermes-talk"
    talk.mkdir(parents=True)
    (talk / "talk_auth.py").write_text("raise RuntimeError('hermes-talk talk_auth must not load')\n")
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("plugin_api", {str(PLUGIN_API)!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import talk_auth
        print(json.dumps({{"file": talk_auth.__file__, "same": talk_auth is mod.talk_auth}}))
    """), encoding="utf-8")
    env = dict(os.environ, HOME=str(tmp_path))
    res = subprocess.run([sys.executable, str(child)], capture_output=True, text=True, timeout=60, env=env)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert Path(out["file"]).resolve() == TALK_AUTH.resolve() and out["same"]
