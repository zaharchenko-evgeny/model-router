#!/usr/bin/env python3
"""Claude Code hooks: route Agent delegations to the right harness/model.

PreToolUse(Agent): classify the delegation and set its model / deny with a routing instruction.
UserPromptSubmit:  on "implement the plan"-style messages, nudge the main session to dispatch
                   tasks via Agent, because the router only sees delegated work.
SessionEnd:        launch learn.py (detached) when Jev-unsure cases were queued.
SessionStart:      show what learn.py learned since the last session.

Tiers (heaviest last): mechanical -> haiku, middle -> junie-with-qwen (deny + instruct),
intelligent -> opus, long_run -> opus in a worktree.

Classification order: [route:X] tag > Jev (TypeSafe System One) > weighted regex rules.
Any internal error fails open: no output, exit 0, the Agent call proceeds unchanged.

CLI:
  route.py                      hook mode (reads the PreToolUse payload on stdin)
  route.py --explain "prompt"   print the routing decision and scores
  route.py --explain "prompt" --no-jev
  route.py --nudge "message"    would this user message trigger the implementation nudge?
  route.py --check              validate rules.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIER_ORDER = ["mechanical", "middle", "intelligent", "long_run"]
TAG_RE = re.compile(r"\[route:([a-z_]+)\]", re.IGNORECASE)
PATH_RE = re.compile(r"(?:[\w.-]+/)*[\w-]+\.[A-Za-z]{1,5}\b|\b[\w.-]+/[\w./*-]+")
ACTIONS = {"allow", "deny"}


# ---------- config ----------

def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


PROJECT_DIRS = (".claude", ".agents")


def global_tiers_path() -> Path:
    return Path(os.environ.get("CLAUDE_ROUTER_TIERS", HERE / "tiers.json"))


def project_file(cwd: str | None, name: str) -> Path | None:
    """<cwd>/.claude/<name>, else <cwd>/.agents/<name>, else None."""
    if not cwd:
        return None
    for d in PROJECT_DIRS:
        p = Path(cwd) / d / name
        if p.is_file():
            return p
    return None


def merge_tiers(base: dict, over: dict) -> dict:
    """Project tiers over global: examples are prepended (project first), other keys deep-merged."""
    out = {name: dict(t) for name, t in base.items()}
    for name, t in over.items():
        merged = _merge(out.get(name, {}), {k: v for k, v in t.items() if k != "examples"})
        merged["examples"] = list(t.get("examples", [])) + list(out.get(name, {}).get("examples", []))
        out[name] = merged
    return out


def load_rules(cwd: str | None = None) -> dict:
    """rules.json + tiers.json, with per-project router.json / tiers.json from .claude or .agents.
    rules["_tiers_target"] is the tiers file that learning for this cwd writes to."""
    rules = json.loads(Path(os.environ.get("CLAUDE_ROUTER_RULES", HERE / "rules.json")).read_text())
    project_rules = project_file(cwd, "router.json")
    if project_rules:
        rules = _merge(rules, json.loads(project_rules.read_text()))
    tiers = json.loads(global_tiers_path().read_text())
    project_tiers = project_file(cwd, "tiers.json")
    if project_tiers:
        tiers = merge_tiers(tiers, json.loads(project_tiers.read_text()))
    rules["tiers"] = tiers
    rules["_tiers_target"] = str(project_tiers or global_tiers_path())
    return rules


def tier_criterion(tier: dict, max_examples: int) -> str | dict:
    """What Jev sees for one option: the description, plus learned boundary examples if any."""
    examples = [f"{e['task']} ({e['why']})" if e.get("why") else e["task"]
                for e in tier.get("examples", [])[:max_examples]]
    return {"description": tier["description"], "examples": examples} if examples else tier["description"]


def check_rules(rules: dict) -> list[str]:
    errors = []
    if rules.get("mode") not in {"enforce", "advise", "off"}:
        errors.append(f"mode must be enforce|advise|off, got {rules.get('mode')!r}")
    if rules.get("default_tier") not in TIER_ORDER:
        errors.append(f"default_tier must be one of {TIER_ORDER}")
    tiers = rules.get("tiers", {})
    for name in tiers:
        if name not in TIER_ORDER:
            errors.append(f"unknown tier {name!r}")
    for name in TIER_ORDER:
        tier = tiers.get(name)
        if tier is None:
            errors.append(f"missing tier {name!r}")
            continue
        if tier.get("action") not in ACTIONS:
            errors.append(f"{name}.action must be allow|deny")
        if tier.get("action") == "deny" and not tier.get("message"):
            errors.append(f"{name}: deny action needs a message")
        for pattern in tier.get("signals", {}):
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(f"{name}: bad regex {pattern!r}: {e}")
    nudge_cfg = rules.get("implement_nudge", {})
    if nudge_cfg.get("enabled", True):
        try:
            re.compile(nudge_cfg.get("trigger", ""))
        except re.error as e:
            errors.append(f"implement_nudge: bad trigger regex: {e}")
        if not nudge_cfg.get("message") or not nudge_cfg.get("jev_question"):
            errors.append("implement_nudge needs message and jev_question")
    for tag, tier in rules.get("tags", {}).items():
        if tier not in TIER_ORDER:
            errors.append(f"tag {tag!r} maps to unknown tier {tier!r}")
    return errors


# ---------- classification ----------

def score_rules(text: str, rules: dict) -> tuple[str, dict, list[str]]:
    """Weighted regex scoring. Ties go to the heavier tier; zero score -> default_tier."""
    scores = {t: 0 for t in TIER_ORDER}
    matched: list[str] = []
    words = len(text.split())
    paths = len(set(PATH_RE.findall(text)))
    for name in TIER_ORDER:
        tier = rules["tiers"][name]
        for pattern, weight in tier.get("signals", {}).items():
            hits = re.findall(pattern, text, re.IGNORECASE)
            if hits:
                scores[name] += weight
                matched.append(f"{name}:{_hit_text(hits[0])}")
        st = tier.get("structural")
        if st:
            if paths >= st.get("min_file_paths", 10**9):
                scores[name] += st.get("weight", 1)
                matched.append(f"{name}:{paths} paths")
            if words >= st.get("min_words", 10**9):
                scores[name] += st.get("weight", 1)
                matched.append(f"{name}:{words} words")
    best = max(scores.values())
    if best == 0:
        return rules["default_tier"], scores, matched
    tier = [t for t in TIER_ORDER if scores[t] == best][-1]
    return tier, scores, matched


def _hit_text(hit) -> str:
    return (hit if isinstance(hit, str) else next((h for h in hit if h), "")).strip()


def jev_call(state: dict, questions: dict, rules: dict) -> dict | None:
    """POST to the System One API. Returns the answers map, {"error": ...}, or None if Jev is off."""
    cfg = rules.get("jev", {})
    token = os.environ.get(cfg.get("token_env", "JEV_TOKEN"))
    if not cfg.get("enabled", True) or not token:
        return None
    body = {"model": cfg.get("model", "jev-latest"), "state": state, "questions": questions}
    req = urllib.request.Request(
        cfg.get("endpoint", "https://api.typesafe.ai/v1/systemone"),
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.get("timeout_s", 5)) as resp:
            return json.loads(resp.read())["answers"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def ask_jev(description: str, prompt: str, rules: dict) -> dict | None:
    """One Choice question over the tiers. Returns the answer dict or None if unavailable."""
    state = {"task_description": description,
             "task_prompt": prompt[: rules.get("jev", {}).get("max_prompt_chars", 6000)]}
    answers = jev_call(state, {"tier": {
        "type": "choice",
        "instructions": (
            "A coding agent wants to delegate `task_prompt` to a helper. "
            "Which kind of helper does this task need, judged by how much reasoning "
            "and how much sustained multi-step work it requires?"
        ),
        "criteria": {t: tier_criterion(rules["tiers"][t],
                                       rules.get("learning", {}).get("max_examples_per_tier", 8))
                     for t in TIER_ORDER},
    }}, rules)
    if answers is None or "error" in answers:
        return answers
    answer = answers.get("tier") or {}
    if answer.get("choice") not in TIER_ORDER:
        return {"error": f"unexpected choice {answer.get('choice')!r}"}
    return answer


def classify(description: str, prompt: str, rules: dict, use_jev: bool = True) -> dict:
    """Returns {tier, source, reason, ...details}. Tag handling is done by the caller."""
    text = f"{description}\n{prompt}"
    rule_tier, scores, matched = score_rules(text, rules)
    decision = {"rule_tier": rule_tier, "scores": scores, "matched": matched}
    jev = ask_jev(description, prompt, rules) if use_jev else None
    if jev and "error" not in jev:
        conf = jev.get("confidence") or 0
        decision["jev"] = {"choice": jev["choice"], "confidence": conf,
                           "probabilities": jev.get("probabilities")}
        if conf >= rules["jev"].get("min_confidence", 0.5):
            return {**decision, "tier": jev["choice"], "source": "jev",
                    "reason": f"jev {jev['choice']} @ {conf:.2f}"}
        # Unsure: escalate to the heavier of Jev's top two tiers. The regex rules are much
        # weaker than Jev, so they only decide when Jev is unavailable.
        probs = jev.get("probabilities") or {jev["choice"]: 1.0}
        top2 = sorted(probs, key=probs.get, reverse=True)[:2]
        tier = max(top2, key=TIER_ORDER.index)
        return {**decision, "tier": tier, "source": "jev-escalated",
                "reason": f"jev unsure ({'/'.join(top2)} @ {conf:.2f}) -> {tier}"}
    elif jev:
        decision["jev_error"] = jev["error"]
    reason = ", ".join(matched[:4]) or f"no signals, default {rule_tier}"
    return {**decision, "tier": rule_tier, "source": "rules", "reason": f"rules: {reason}"}


# ---------- hook ----------

def in_git_repo(cwd: str | None) -> bool:
    p = Path(cwd or os.getcwd()).resolve()
    return any((d / ".git").exists() for d in [p, *p.parents])


def decide(payload: dict, rules: dict, use_jev: bool = True) -> tuple[dict | None, dict]:
    """Returns (hook_output_or_None, log_record)."""
    tool_input = payload.get("tool_input") or {}
    prompt = tool_input.get("prompt", "")
    description = tool_input.get("description", "")
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cwd": payload.get("cwd"),
              "description": description, "prompt_head": prompt[:200]}

    if rules.get("mode") == "off" or os.environ.get("CLAUDE_ROUTER", "").lower() == "off":
        return None, {**record, "skipped": "mode off"}
    if payload.get("tool_name") != "Agent":
        return None, {**record, "skipped": "not Agent"}
    if tool_input.get("subagent_type") in rules.get("skip_subagent_types", []):
        return None, {**record, "skipped": f"subagent_type {tool_input['subagent_type']}"}

    tag = TAG_RE.search(prompt)
    clean_prompt = TAG_RE.sub("", prompt).strip() if tag else prompt
    if tag and tag.group(1).lower() == "off":
        out = _allow({**tool_input, "prompt": clean_prompt}, "router: [route:off]")
        return out, {**record, "skipped": "tag off"}
    tag_tier = rules.get("tags", {}).get(tag.group(1).lower()) if tag else None
    if tag_tier:
        decision = {"tier": tag_tier, "source": "tag", "reason": f"tag [route:{tag.group(1)}]"}
    else:
        decision = classify(description, clean_prompt, rules, use_jev)
        if decision["source"] == "jev-escalated":
            enqueue_for_learning(payload.get("cwd"), description, clean_prompt, decision, rules)
    record.update(decision)

    tier_name = decision["tier"]
    tier = rules["tiers"][tier_name]
    label = f"router: {tier_name} ({decision['reason']})"

    if rules.get("mode") == "advise":
        target = tier.get("set") or {"action": tier["action"]}
        return _context(f"Router suggestion: {tier_name} -> {json.dumps(target)} "
                        f"({decision['reason']})."), {**record, "action": "advise"}

    if tier["action"] == "deny":
        msg = tier["message"].format(reason=decision["reason"], tier=tier_name)
        return _deny(msg), {**record, "action": "deny"}

    new_input = {**tool_input, "prompt": clean_prompt}
    git_only = set(tier.get("requires_git_for", []))
    for key, value in tier.get("set", {}).items():
        if key in git_only and not in_git_repo(payload.get("cwd")):
            continue
        new_input[key] = value
    return _allow(new_input, label), {**record, "action": "allow",
                                      "set": {k: new_input[k] for k in tier.get("set", {}) if k in new_input}}


def nudge(payload: dict, rules: dict, use_jev: bool = True) -> tuple[dict | None, dict]:
    """UserPromptSubmit: when the user asks to implement/execute work, remind the main session
    to dispatch tasks through Agent so decide() can pick a model per task."""
    cfg = rules.get("implement_nudge", {})
    prompt = payload.get("prompt", "")
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "UserPromptSubmit",
              "cwd": payload.get("cwd"), "prompt_head": prompt[:200]}
    if (not cfg.get("enabled", True) or rules.get("mode") == "off"
            or os.environ.get("CLAUDE_ROUTER", "").lower() == "off"):
        return None, {**record, "skipped": "off"}
    if TAG_RE.search(prompt) and TAG_RE.search(prompt).group(1).lower() == "off":
        return None, {**record, "skipped": "tag off"}
    hit = re.search(cfg.get("trigger", r"$^"), prompt, re.IGNORECASE)
    if not hit:
        return None, {**record, "skipped": "no trigger"}
    record["trigger"] = hit.group(0)
    # Regex is a cheap prefilter; Jev rules out questions like "how would you implement X?".
    answers = jev_call({"user_message": prompt[:2000]}, {"wants_impl": {
        "type": "noul", "instructions": cfg["jev_question"]}}, rules) if use_jev else None
    if answers and "error" not in answers:
        p = answers.get("wants_impl", {}).get("noul", 1.0)
        record["jev_noul"] = p
        if p < cfg.get("min_probability", 0.5):
            return None, {**record, "skipped": f"jev noul {p:.2f}"}
    elif answers:
        record["jev_error"] = answers["error"]
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                   "additionalContext": cfg["message"]}}, {**record, "action": "nudge"}


def _allow(updated_input: dict, reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                   "permissionDecisionReason": reason, "updatedInput": updated_input}}


def _deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def _context(text: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": text}}


def learning_dir() -> Path:
    return Path(os.environ.get("CLAUDE_ROUTER_LEARNING_DIR", HERE / "learning"))


def enqueue_for_learning(cwd, description: str, prompt: str, decision: dict, rules: dict) -> None:
    """Jev was unsure: keep the case so learn.py can have Sonnet label it at session end."""
    cfg = rules.get("learning", {})
    if not cfg.get("enabled", True):
        return
    case = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cwd": cwd, "target": rules["_tiers_target"],
            "description": description, "prompt": prompt[: cfg.get("queue_prompt_chars", 2000)],
            "routed": decision["tier"], "jev": decision.get("jev")}
    try:
        learning_dir().mkdir(parents=True, exist_ok=True)
        with (learning_dir() / "queue.jsonl").open("a") as f:
            f.write(json.dumps(case) + "\n")
    except OSError:
        pass


def start_learner() -> dict:
    """SessionEnd: launch learn.py detached if there is anything queued (the hook budget is ~1.5s)."""
    queue = learning_dir() / "queue.jsonl"
    if not queue.is_file() or queue.stat().st_size == 0:
        return {"skipped": "empty queue"}
    import subprocess
    learning_dir().mkdir(parents=True, exist_ok=True)
    with (learning_dir() / "learn.log").open("a") as out:
        subprocess.Popen([sys.executable, str(HERE / "learn.py")], stdout=out, stderr=out,
                         stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())
    return {"action": "learner started"}


def learning_report() -> dict | None:
    """SessionStart: tell the user what the learner changed since the last session, once."""
    notices_path = learning_dir() / "notices.jsonl"
    if not notices_path.is_file():
        return None
    try:
        notices = [json.loads(line) for line in notices_path.read_text().splitlines() if line.strip()]
        with (learning_dir() / "notices.seen.jsonl").open("a") as f:
            f.writelines(json.dumps(n) + "\n" for n in notices)
        notices_path.unlink()
    except (OSError, ValueError):
        return None
    learned = [n for n in notices if n.get("kind") == "learned"]
    rejected = [n for n in notices if n.get("kind") == "rejected"]
    if not learned and not rejected:
        return None
    lines = [f"Router learned {len(learned)} routing case(s)"
             + (f", rejected {len(rejected)}" if rejected else "") + ":"]
    for n in learned:
        where = "global" if Path(n["target"]).resolve() == global_tiers_path().resolve() else n["target"]
        extra = ", description updated" if n.get("description_changed") else ""
        lines.append(f"  + {n['tier']}: \"{n['task']}\" ({n['why']}{extra}) -> {where}")
    for n in rejected[:5]:
        lines.append(f"  - rejected: \"{n['task']}\" ({n['reason']})")
    lines.append(f"Details: python3 {HERE / 'learn.py'} --status   Undo: --revert [tiers file]")
    return {"systemMessage": "\n".join(lines)}


def log(record: dict) -> None:
    path = Path(os.environ.get("CLAUDE_ROUTER_LOG", Path.home() / ".claude/logs/router.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def main(argv: list[str]) -> int:
    if "--check" in argv:
        errors = check_rules(load_rules(os.getcwd()))
        print("\n".join(errors) or "rules OK")
        return 1 if errors else 0
    if "--nudge" in argv:
        payload = {"hook_event_name": "UserPromptSubmit", "cwd": os.getcwd(),
                   "prompt": argv[argv.index("--nudge") + 1]}
        out, record = nudge(payload, load_rules(os.getcwd()), use_jev="--no-jev" not in argv)
        print(json.dumps({"decision": record, "hook_output": out}, indent=2))
        return 0
    if "--explain" in argv:
        prompt = argv[argv.index("--explain") + 1]
        payload = {"tool_name": "Agent", "cwd": os.getcwd(),
                   "tool_input": {"description": "", "prompt": prompt}}
        out, record = decide(payload, load_rules(os.getcwd()), use_jev="--no-jev" not in argv)
        print(json.dumps({"decision": record, "hook_output": out}, indent=2))
        return 0
    try:
        payload = json.load(sys.stdin)
        if payload.get("hook_event_name") == "SessionStart":
            report = learning_report()
            if report:
                print(json.dumps(report))
            return 0
        if payload.get("hook_event_name") == "SessionEnd":
            log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "SessionEnd", **start_learner()})
            return 0
        rules = load_rules(payload.get("cwd"))
        if payload.get("hook_event_name") == "UserPromptSubmit":
            out, record = nudge(payload, rules)
        else:
            out, record = decide(payload, rules)
        log(record)
        if out:
            print(json.dumps(out))
    except Exception as e:  # fail open: never block the user's work
        log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
