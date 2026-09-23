import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import route  # noqa: E402


@pytest.fixture
def rules():
    return route.load_rules()


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    monkeypatch.delenv("JEV_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_ROUTER", raising=False)
    monkeypatch.setenv("CLAUDE_ROUTER_LOG", str(tmp_path / "router.jsonl"))
    monkeypatch.setenv("CLAUDE_ROUTER_LEARNING_DIR", str(tmp_path / "learning"))


def payload(prompt, description="", subagent_type=None, cwd=None, **extra):
    tool_input = {"prompt": prompt, "description": description, **extra}
    if subagent_type:
        tool_input["subagent_type"] = subagent_type
    return {"tool_name": "Agent", "cwd": cwd, "tool_input": tool_input}


def hso(out):
    return out["hookSpecificOutput"]


# ---------- rule classifier ----------

@pytest.mark.parametrize("prompt,tier", [
    ("rename getUser to fetchUser everywhere", "mechanical"),
    ("fix the typo in README.md", "mechanical"),
    ("bump the version in package.json", "mechanical"),
    ("write unit tests for parse_date following the spec below", "middle"),
    ("scaffold a new CLI command called sync", "middle"),
    ("implement the function slugify in utils.py", "middle"),
    ("investigate the root cause of the flaky login test", "intelligent"),
    ("review this diff for security issues", "intelligent"),
    ("decide between Redis and Postgres for the job queue and explain the trade-offs", "intelligent"),
    ("build the whole feature end-to-end from the spec", "long_run"),
    ("migrate the codebase from Moshi to kotlinx.serialization", "long_run"),
    ("hello there", "intelligent"),  # no signal -> default_tier
])
def test_rule_tiers(rules, prompt, tier):
    assert route.score_rules(prompt, rules)[0] == tier


def test_tie_goes_to_heavier_tier(rules):
    rules["tiers"]["mechanical"]["signals"] = {"foo": 3}
    rules["tiers"]["middle"]["signals"] = {"foo": 3}
    assert route.score_rules("foo", rules)[0] == "middle"


def test_structural_signals_push_long_run(rules):
    files = " ".join(f"src/mod{i}/file{i}.py" for i in range(8))
    tier, scores, matched = route.score_rules(f"update these {files}", rules)
    assert scores["long_run"] >= 2
    assert any("paths" in m for m in matched)


# ---------- Jev ----------

def fake_jev(choice, confidence):
    return lambda d, p, r: {"choice": choice, "confidence": confidence, "probabilities": {choice: confidence}}


def test_confident_jev_wins_over_rules(rules, monkeypatch):
    monkeypatch.setattr(route, "ask_jev", fake_jev("long_run", 0.9))
    d = route.classify("", "rename foo to bar", rules)
    assert (d["tier"], d["source"]) == ("long_run", "jev")


def test_low_confidence_jev_escalates_to_heavier_of_top_two(rules, monkeypatch):
    # Regression: "Task 3 of the plan: add the RouterConfig loader..." came back
    # middle 0.7 / mechanical 0.3 @ 0.49, and the regex fallback sent it to Opus.
    monkeypatch.setattr(route, "ask_jev", lambda d, p, r: {
        "choice": "middle", "confidence": 0.49,
        "probabilities": {"middle": 0.7, "mechanical": 0.3, "intelligent": 0.0, "long_run": 0.0}})
    d = route.classify("", "Task 3 of the plan: add the RouterConfig loader", rules)
    assert (d["tier"], d["source"]) == ("middle", "jev-escalated")


def test_low_confidence_escalation_picks_heavier_even_if_less_likely(rules, monkeypatch):
    monkeypatch.setattr(route, "ask_jev", lambda d, p, r: {
        "choice": "middle", "confidence": 0.3,
        "probabilities": {"middle": 0.55, "intelligent": 0.4, "mechanical": 0.05, "long_run": 0.0}})
    assert route.classify("", "rename foo", rules)["tier"] == "intelligent"


def test_jev_error_falls_back_to_rules(rules, monkeypatch):
    monkeypatch.setattr(route, "ask_jev", lambda d, p, r: {"error": "URLError: boom"})
    d = route.classify("", "rename foo to bar", rules)
    assert d["source"] == "rules" and d["jev_error"] == "URLError: boom"


def test_no_token_skips_jev(rules):
    assert route.ask_jev("", "anything", rules) is None


def test_jev_request_shape(rules, monkeypatch):
    monkeypatch.setenv("JEV_TOKEN", "t0k")
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_urlopen(req, timeout):
        seen["body"] = json.loads(req.data)
        seen["auth"] = req.headers["Authorization"]
        return Resp(json.dumps({"answers": {"tier": {"type": "choice", "choice": "middle",
                                                     "confidence": 0.8, "probabilities": {}}}}).encode())

    monkeypatch.setattr(route.urllib.request, "urlopen", fake_urlopen)
    answer = route.ask_jev("desc", "do it", rules)
    assert answer["choice"] == "middle"
    assert seen["auth"] == "Bearer t0k"
    q = seen["body"]["questions"]["tier"]
    assert q["type"] == "choice" and set(q["criteria"]) == set(route.TIER_ORDER)
    assert seen["body"]["state"]["task_prompt"] == "do it"


# ---------- hook contract ----------

def test_mechanical_rewrites_model_to_haiku(rules):
    out, rec = route.decide(payload("fix the typo in README.md", description="typo"), rules, use_jev=False)
    h = hso(out)
    assert h["permissionDecision"] == "allow"
    assert h["updatedInput"] == {"prompt": "fix the typo in README.md", "description": "typo", "model": "haiku"}


def test_middle_denies_with_junie_instruction(rules):
    out, rec = route.decide(payload("scaffold a new CLI command"), rules, use_jev=False)
    h = hso(out)
    assert h["permissionDecision"] == "deny"
    assert "junie-with-qwen" in h["permissionDecisionReason"]
    assert "[route:opus]" in h["permissionDecisionReason"]


def test_long_run_sets_worktree_only_in_git_repo(rules, tmp_path):
    prompt = "build the whole feature end-to-end"
    out, _ = route.decide(payload(prompt, cwd=str(tmp_path)), rules, use_jev=False)
    assert hso(out)["updatedInput"]["model"] == "opus"
    assert "isolation" not in hso(out)["updatedInput"]
    (tmp_path / ".git").mkdir()
    out, _ = route.decide(payload(prompt, cwd=str(tmp_path)), rules, use_jev=False)
    assert hso(out)["updatedInput"]["isolation"] == "worktree"


def test_tag_overrides_and_is_stripped(rules):
    out, rec = route.decide(payload("[route:opus] fix the typo"), rules, use_jev=False)
    h = hso(out)
    assert h["updatedInput"]["model"] == "opus"
    assert h["updatedInput"]["prompt"] == "fix the typo"
    assert rec["source"] == "tag"


def test_route_off_tag_passes_through(rules):
    out, rec = route.decide(payload("[route:off] scaffold a thing", model="sonnet"), rules, use_jev=False)
    h = hso(out)
    assert h["updatedInput"] == {"prompt": "scaffold a thing", "description": "", "model": "sonnet"}


@pytest.mark.parametrize("st", ["Explore", "fork", "Plan"])
def test_skip_subagent_types(rules, st):
    out, rec = route.decide(payload("scaffold a thing", subagent_type=st), rules, use_jev=False)
    assert out is None and "skipped" in rec


def test_non_agent_tool_ignored(rules):
    out, _ = route.decide({"tool_name": "Bash", "tool_input": {"command": "ls"}}, rules, use_jev=False)
    assert out is None


def test_mode_off_and_env_kill_switch(rules, monkeypatch):
    rules["mode"] = "off"
    assert route.decide(payload("scaffold x"), rules, use_jev=False)[0] is None
    rules["mode"] = "enforce"
    monkeypatch.setenv("CLAUDE_ROUTER", "off")
    assert route.decide(payload("scaffold x"), rules, use_jev=False)[0] is None


def test_advise_mode_only_adds_context(rules):
    rules["mode"] = "advise"
    out, _ = route.decide(payload("scaffold x"), rules, use_jev=False)
    h = hso(out)
    assert "permissionDecision" not in h and "middle" in h["additionalContext"]


def test_project_router_json_merges(tmp_path):
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "router.json").write_text(json.dumps({"jev": {"enabled": False}}))
    rules = route.load_rules(str(tmp_path))
    assert rules["jev"]["enabled"] is False and rules["jev"]["model"] == "jev-latest"


def test_project_tiers_merge_and_become_learning_target(tmp_path):
    (tmp_path / ".claude").mkdir()
    project = tmp_path / ".claude" / "tiers.json"
    project.write_text(json.dumps({"middle": {
        "action": "allow", "set": {"model": "sonnet"},
        "examples": [{"task": "kotlin dto conversion", "why": "clear spec"}]}}))
    rules = route.load_rules(str(tmp_path))
    middle = rules["tiers"]["middle"]
    assert middle["action"] == "allow" and middle["set"] == {"model": "sonnet"}
    assert middle["description"]  # untouched keys survive the merge
    assert middle["examples"][0]["task"] == "kotlin dto conversion"
    assert rules["_tiers_target"] == str(project)


def test_claude_dir_wins_over_agents_dir(tmp_path):
    for d, model in ((".claude", "sonnet"), (".agents", "haiku")):
        (tmp_path / d).mkdir()
        (tmp_path / d / "tiers.json").write_text(json.dumps({"middle": {"action": "allow", "set": {"model": model}}}))
    assert route.load_rules(str(tmp_path))["tiers"]["middle"]["set"]["model"] == "sonnet"


def test_no_project_tiers_targets_global(tmp_path):
    assert route.load_rules(str(tmp_path))["_tiers_target"] == str(route.global_tiers_path())


def test_merge_tiers_prepends_project_examples():
    base = {"middle": {"description": "d", "examples": [{"task": "g"}]}}
    merged = route.merge_tiers(base, {"middle": {"examples": [{"task": "p"}]}})
    assert [e["task"] for e in merged["middle"]["examples"]] == ["p", "g"]
    assert base["middle"]["examples"] == [{"task": "g"}]  # base not mutated


def test_criterion_is_plain_description_without_examples():
    assert route.tier_criterion({"description": "d", "examples": []}, 8) == "d"
    c = route.tier_criterion({"description": "d", "examples": [{"task": "t", "why": "w"}] * 10}, 3)
    assert c == {"description": "d", "examples": ["t (w)"] * 3}


def test_escalated_case_is_queued(rules, monkeypatch, tmp_path):
    monkeypatch.setattr(route, "ask_jev", lambda d, p, r: {
        "choice": "middle", "confidence": 0.4, "probabilities": {"middle": 0.6, "mechanical": 0.4}})
    route.decide(payload("add the config loader", cwd=str(tmp_path)), rules)
    queued = [json.loads(l) for l in (route.learning_dir() / "queue.jsonl").read_text().splitlines()]
    assert queued[0]["prompt"] == "add the config loader" and queued[0]["target"] == rules["_tiers_target"]


def test_confident_case_is_not_queued(rules, monkeypatch):
    monkeypatch.setattr(route, "ask_jev", fake_jev("middle", 0.95))
    route.decide(payload("add the config loader"), rules)
    assert not (route.learning_dir() / "queue.jsonl").exists()


def test_session_start_reports_notices_once(capsys):
    route.learning_dir().mkdir(parents=True, exist_ok=True)
    (route.learning_dir() / "notices.jsonl").write_text(json.dumps({
        "kind": "learned", "target": str(route.global_tiers_path()), "tier": "middle",
        "why": "clear spec", "task": "Task 3: add loader", "description_changed": False}) + "\n")
    for _ in range(2):
        sys.stdin = io.StringIO(json.dumps({"hook_event_name": "SessionStart"}))
        route.main([])
    first, second = capsys.readouterr().out, ""
    msg = json.loads(first.splitlines()[0])["systemMessage"]
    assert "learned 1" in msg and "Task 3: add loader" in msg and "global" in msg
    assert len(first.splitlines()) == 1  # second SessionStart printed nothing


def test_session_end_without_queue_does_not_spawn(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("spawned learner"))
    assert route.start_learner() == {"skipped": "empty queue"}


def test_shipped_rules_are_valid(rules):
    assert route.check_rules(rules) == []


def test_check_catches_bad_regex(rules):
    rules["tiers"]["middle"]["signals"] = {"(unclosed": 1}
    assert any("bad regex" in e for e in route.check_rules(rules))


# ---------- implementation nudge (UserPromptSubmit) ----------

def prompt_payload(text):
    return {"hook_event_name": "UserPromptSubmit", "prompt": text, "cwd": None}


@pytest.mark.parametrize("text", [
    "ok, looks reasonable, lets implement",
    "go ahead and execute the plan",
    "start the implementation",
])
def test_nudge_fires_on_implementation_requests(rules, text):
    out, rec = route.nudge(prompt_payload(text), rules, use_jev=False)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "Agent tool" in ctx and rec["action"] == "nudge"


@pytest.mark.parametrize("text", ["what does this function do?", "design the caching layer"])
def test_nudge_ignores_non_implementation(rules, text):
    assert route.nudge(prompt_payload(text), rules, use_jev=False)[0] is None


def test_nudge_jev_vetoes_questions(rules, monkeypatch):
    monkeypatch.setattr(route, "jev_call", lambda s, q, r: {"wants_impl": {"type": "noul", "noul": 0.1}})
    out, rec = route.nudge(prompt_payload("how would you implement a trie?"), rules)
    assert out is None and rec["skipped"].startswith("jev noul")


def test_nudge_regex_decides_when_jev_errors(rules, monkeypatch):
    monkeypatch.setattr(route, "jev_call", lambda s, q, r: {"error": "URLError: down"})
    out, rec = route.nudge(prompt_payload("lets implement it"), rules)
    assert out is not None and rec["jev_error"]


def test_nudge_respects_off_switches(rules, monkeypatch):
    assert route.nudge(prompt_payload("[route:off] implement it"), rules, use_jev=False)[0] is None
    rules["implement_nudge"]["enabled"] = False
    assert route.nudge(prompt_payload("implement it"), rules, use_jev=False)[0] is None
    rules["implement_nudge"]["enabled"] = True
    monkeypatch.setenv("CLAUDE_ROUTER", "off")
    assert route.nudge(prompt_payload("implement it"), rules, use_jev=False)[0] is None


def test_main_dispatches_user_prompt_submit(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(prompt_payload("lets implement"))))
    assert route.main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


# ---------- process level: fail open ----------

def test_main_fails_open_on_garbage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    assert route.main([]) == 0
    assert capsys.readouterr().out == ""


def test_main_logs_decision(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload("fix the typo"))))
    assert route.main([]) == 0
    assert json.loads(capsys.readouterr().out)["hookSpecificOutput"]["updatedInput"]["model"] == "haiku"
    rec = json.loads((tmp_path / "router.jsonl").read_text().splitlines()[-1])
    assert rec["tier"] == "mechanical"
