"""learn.py with Sonnet and Jev stubbed. The fake Jev 'learns': it is confident about a prompt
only once that prompt appears among a tier's examples, which is what the real loop relies on."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import learn  # noqa: E402
import route  # noqa: E402

BASELINE = {  # prompt fragment -> tier Jev gets right with confidence before any learning
    "rename": "mechanical", "typo": "mechanical",
    "slugify": "middle", "root cause": "intelligent",
}


def fake_ask_jev(description, prompt, rules):
    for tier in route.TIER_ORDER:
        for e in rules["tiers"][tier].get("examples", []):
            if prompt[:40] in e["task"]:
                return {"choice": tier, "confidence": 0.9, "probabilities": {tier: 0.9}}
    for frag, tier in BASELINE.items():
        if frag in prompt:
            return {"choice": tier, "confidence": 0.95, "probabilities": {tier: 0.95}}
    return {"choice": "middle", "confidence": 0.3, "probabilities": {"middle": 0.6, "mechanical": 0.4}}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated copies of tiers.json and evals/cases.json, plus a learning dir."""
    root = tmp_path / "router"
    (root / "evals").mkdir(parents=True)
    shutil.copy(ROOT / "tiers.json", root / "tiers.json")
    (root / "evals" / "cases.json").write_text(json.dumps({"tiers": [
        {"prompt": "rename a to b", "want": "mechanical"},
        {"prompt": "implement slugify in utils.py", "want": "middle"},
        {"prompt": "find the root cause of the leak", "want": "intelligent"},
    ], "nudge": []}))
    monkeypatch.setattr(learn, "HERE", root)
    monkeypatch.setenv("CLAUDE_ROUTER_TIERS", str(root / "tiers.json"))
    monkeypatch.setenv("CLAUDE_ROUTER_LEARNING_DIR", str(root / "learning"))
    monkeypatch.setenv("CLAUDE_ROUTER_LOG", str(tmp_path / "router.jsonl"))
    monkeypatch.setenv("JEV_TOKEN", "test")
    monkeypatch.setattr(route, "ask_jev", fake_ask_jev)
    monkeypatch.setattr(learn, "LABELER", lambda case, rules: {
        "tier": "middle", "why": "clear spec, single file", "description_update": None})
    return root


def queue(root, prompt, cwd=None, target=None):
    rules = route.load_rules(cwd)
    route.enqueue_for_learning(cwd, "", prompt, {"tier": "middle", "jev": {"probabilities": {}}}, rules)


def tiers(root):
    return json.loads((root / "tiers.json").read_text())


def test_accepts_example_and_jev_then_routes_confidently(env):
    queue(env, "Task 3 of the plan: add the RouterConfig loader in config.py")
    learn.main([])
    ex = tiers(env)["middle"]["examples"]
    assert ex[0]["why"] == "clear spec, single file" and "RouterConfig" in ex[0]["task"]
    d = route.classify("", "Task 3 of the plan: add the RouterConfig loader in config.py", route.load_rules())
    assert (d["tier"], d["source"]) == ("middle", "jev")
    cases = json.loads((env / "evals" / "cases.json").read_text())["tiers"]
    assert cases[-1]["want"] == "middle"  # eval set grows with learned cases
    assert not (env / "learning" / "queue.jsonl").exists()


def test_notice_written_for_session_start(env):
    queue(env, "Task 3 of the plan: add the RouterConfig loader in config.py")
    learn.main([])
    notice = learn.read_jsonl(env / "learning" / "notices.jsonl")[0]
    assert notice["kind"] == "learned" and notice["tier"] == "middle"
    msg = route.learning_report()["systemMessage"]
    assert "learned 1" in msg and "RouterConfig" in msg


def test_regression_rejects_and_leaves_tiers_untouched(env, monkeypatch):
    before = tiers(env)

    def breaking_jev(description, prompt, rules):  # any learned example breaks the rename case
        if rules["tiers"]["middle"].get("examples") and "rename" in prompt:
            return {"choice": "middle", "confidence": 0.9, "probabilities": {"middle": 0.9}}
        return fake_ask_jev(description, prompt, rules)

    monkeypatch.setattr(route, "ask_jev", breaking_jev)
    queue(env, "Task 3 of the plan: add the RouterConfig loader in config.py")
    learn.main([])
    assert tiers(env) == before
    rejected = learn.read_jsonl(env / "learning" / "rejected.jsonl")
    assert "regressions" in rejected[0]["reason"]
    assert learn.read_jsonl(env / "learning" / "notices.jsonl")[0]["kind"] == "rejected"


def test_description_update_falls_back_to_example_only(env, monkeypatch):
    monkeypatch.setattr(learn, "LABELER", lambda case, rules: {
        "tier": "middle", "why": "clear spec",
        "description_update": {"tier": "mechanical", "description": "BROKEN"}})
    original = route.ask_jev

    def jev(description, prompt, rules):  # the rewrite breaks the rename case
        if rules["tiers"]["mechanical"]["description"] == "BROKEN" and "rename" in prompt:
            return {"choice": "middle", "confidence": 0.9, "probabilities": {"middle": 0.9}}
        return original(description, prompt, rules)

    monkeypatch.setattr(route, "ask_jev", jev)
    queue(env, "Task 3 of the plan: add the RouterConfig loader in config.py")
    learn.main([])
    t = tiers(env)
    assert t["mechanical"]["description"] != "BROKEN"
    assert t["middle"]["examples"][0]["why"] == "clear spec"


def test_learns_into_project_tiers_file(env, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".agents").mkdir(parents=True)
    (repo / ".agents" / "tiers.json").write_text("{}")
    global_before = tiers(env)
    queue(env, "Task 3 of the plan: add the RouterConfig loader in config.py", cwd=str(repo))
    learn.main([])
    project = json.loads((repo / ".agents" / "tiers.json").read_text())
    assert project["middle"]["examples"][0]["why"] == "clear spec, single file"
    assert tiers(env) == global_before
    assert json.loads((repo / ".agents" / "tiers.cases.json").read_text())[0]["want"] == "middle"


def test_examples_are_capped(env, monkeypatch):
    for i in range(12):
        queue(env, f"Task {i} of the plan: add loader number {i} in config{i}.py")
    learn.main([])
    assert len(tiers(env)["middle"]["examples"]) == 8
    assert "Task 11" in tiers(env)["middle"]["examples"][0]["task"]  # newest first


def test_secret_like_prompts_are_never_learned(env):
    before = tiers(env)
    queue(env, "deploy with token=ghp_abcdefghijklmnopqrstuvwxyz0123 and add the loader")
    learn.main([])
    assert tiers(env) == before
    assert learn.read_jsonl(env / "learning" / "rejected.jsonl")[0]["reason"] == "secret-like content"


def test_no_jev_token_defers_and_keeps_queue(env, monkeypatch):
    monkeypatch.delenv("JEV_TOKEN")
    queue(env, "Task 3 of the plan: add the RouterConfig loader")
    learn.main([])
    assert len(learn.read_jsonl(env / "learning" / "queue.jsonl")) == 1


def test_dry_run_writes_nothing(env):
    before = tiers(env)
    queue(env, "Task 3 of the plan: add the RouterConfig loader in config.py")
    learn.main(["--dry-run"])
    assert tiers(env) == before
    assert len(learn.read_jsonl(env / "learning" / "queue.jsonl")) == 1


def test_revert_restores_previous_tiers_and_cases(env):
    before = tiers(env)
    queue(env, "Task 3 of the plan: add the RouterConfig loader in config.py")
    learn.main([])
    assert tiers(env) != before
    assert learn.main(["--revert"]) == 0
    assert tiers(env) == before
    assert len(json.loads((env / "evals" / "cases.json").read_text())["tiers"]) == 3


def test_lock_blocks_second_learner(env):
    with learn.Lock():
        with pytest.raises(SystemExit):
            with learn.Lock():
                pass


def test_duplicate_queued_prompts_processed_once(env, monkeypatch):
    calls = []
    monkeypatch.setattr(learn, "LABELER", lambda case, rules: calls.append(1) or {
        "tier": "middle", "why": "w", "description_update": None})
    for _ in range(3):
        queue(env, "Task 3 of the plan: add the RouterConfig loader")
    learn.main([])
    assert len(calls) == 1


@pytest.mark.parametrize("text,ok", [
    ('{"tier": "middle", "why": "clear spec", "description_update": null}', True),
    ('```json\n{"tier": "long_run", "why": "x"}\n```', True),
    ('{"tier": "sonnet"}', False),
    ("I think middle", False),
])
def test_parse_label_is_strict(text, ok):
    assert (learn.parse_label(text, {"max_description_chars": 600}) is not None) == ok


def test_parse_label_caps_why_and_description():
    lbl = learn.parse_label(json.dumps({"tier": "middle", "why": " ".join(["w"] * 40),
                                        "description_update": {"tier": "middle", "description": "d" * 900}}),
                            {"max_description_chars": 600})
    assert len(lbl["why"].split()) == 15 and len(lbl["description_update"]["description"]) == 600
