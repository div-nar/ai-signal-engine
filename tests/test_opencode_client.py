"""_OpencodeClient: parse `opencode run --format json` event stream -> text."""
import subprocess
import pytest
from scoring.thesis_scorer import _OpencodeClient

# A real captured opencode --format json stream (one JSON object per line).
_STREAM = (
    '{"type":"step_start","part":{"type":"step-start"}}\n'
    '{"type":"text","part":{"type":"text",'
    '"text":"{\\"market_regime\\":\\"compute_constrained\\"}"}}\n'
    '{"type":"step_finish","part":{"type":"step-finish","tokens":{"total":10}}}\n'
)


def test_generate_extracts_assistant_text(monkeypatch):
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=_STREAM, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = _OpencodeClient().generate("hello")
    assert out == '{"market_regime":"compute_constrained"}'
    # invoked the right binary + model, non-interactive, pure
    assert calls["cmd"][:3] == ["opencode", "run", "--pure"]
    assert "opencode-go/qwen3.7-max" in calls["cmd"]
    assert "hello" in calls["cmd"]


def test_generate_concatenates_multiple_text_parts(monkeypatch):
    stream = (
        '{"type":"text","part":{"type":"text","text":"foo"}}\n'
        '{"type":"text","part":{"type":"text","text":"bar"}}\n'
    )
    monkeypatch.setattr(subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stream, stderr=""))
    assert _OpencodeClient().generate("x") == "foobar"


def test_generate_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"))
    with pytest.raises(RuntimeError, match="opencode"):
        _OpencodeClient().generate("x")


def test_generate_raises_on_empty_text(monkeypatch):
    stream = '{"type":"step_finish","part":{"type":"step-finish"}}\n'
    monkeypatch.setattr(subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stream, stderr=""))
    with pytest.raises(RuntimeError, match="opencode"):
        _OpencodeClient().generate("x")


def test_generate_raises_on_timeout(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 120)
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="opencode"):
        _OpencodeClient().generate("x")
