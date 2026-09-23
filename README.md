# Router

A Claude Code hook that decides **before every subagent delegation** which
harness/model should do the work. Routing no longer depends on the model obeying CLAUDE.md prose.

| Tier | Typical task | What the hook does |
|---|---|---|
| `mechanical` | rename, typo, bump, reformat, lookups | allow, rewrite `model: "haiku"` |
| `middle` | implement one function/file, tests to a spec, scaffolding | **deny**, tell Claude to use the `junie-with-qwen` skill |
| `intelligent` | design, trade-offs, root-cause debugging, review, security | allow, rewrite `model: "opus"` |
| `long_run` | whole feature end-to-end, migrations, many files/steps | allow, `model: "opus"` + `isolation: "worktree"` (only inside a git repo) |

## How a decision is made

```
Agent(prompt) ─► PreToolUse hook (route.py)
   1. mode off / CLAUDE_ROUTER=off / skipped subagent_type ─► pass through untouched
   2. [route:haiku|junie|opus|long|off] tag in the prompt ─► tag wins (tag is stripped)
   3. Jev (TypeSafe System One) Choice over the 4 tiers
        confidence ≥ jev.min_confidence ─► Jev's tier
        confidence below it             ─► heavier of Jev's top-two tiers (escalate when unsure)
   4. Jev unavailable (no token / error / timeout) ─► weighted regex rules
        (ties → heavier tier, no signal → default_tier)
   ─► allow + updatedInput  |  deny + instruction  |  advise-only context
```

The regex rules are only an outage fallback. On `evals/cases.json` they get 13/21, and Jev gets 21/21.

- **Jev** gets the Agent `description` and `prompt` (cut to `max_prompt_chars`) as state. It answers one
  `choice` question whose options are the tier `description`s in `rules.json`, so editing a
  description changes how Jev reads the tier. The token comes from the `JEV_TOKEN` env var (Claude Code
  inherits it from your shell). There is no token, the call fails, or it times out (`timeout_s`)?
  Then the rules decide.
- **Fails open.** Any error means no output and exit 0, so the Agent call runs as Claude wrote it.
- **Skipped types.** Specialised subagents (`Explore`, `Plan`, `fork`, …, see `skip_subagent_types`)
  are never touched.
- Typical latency is about 0.4 s with Jev and about 60 ms rules-only.

> Delegation prompts are sent to `api.typesafe.ai` for classification. Set `jev.enabled: false`
> (globally or per project) for code you can't send there.

## Routing during implementation

The router only sees **delegated** work, meaning `Agent` calls. If the main session implements a plan
itself with Edit/Write, every line is written by the main-session model. A hook can't change the
main model: `PreModelSwitch` only gates switches you request with `/model`.

So a second hook (`UserPromptSubmit`) watches your messages. When one looks like "implement / execute
the plan / go ahead", it adds a note to Claude's context. The note tells Claude to act as the orchestrator
and dispatch each task through `Agent` with a self-contained prompt, so each task gets routed on its own:

```
"lets implement" ─► regex trigger ─► Jev noul "is this a request to implement now?" ≥ min_probability
                 ─► additionalContext: "dispatch each task via Agent ..."
Claude ─► Agent("Task 2: implement parse_rules() ... tests ...") ─► router ─► middle → junie-with-qwen
Claude ─► Agent("Task 5: design the plugin loading order ...")  ─► router ─► intelligent → opus
```

Jev screens out questions such as "how would you implement a trie?". If Jev is unavailable, the regex
decides alone. This is a nudge, not enforcement, and the global CLAUDE.md carries the same rule. Configure it under
`implement_nudge` in `rules.json`, and check a message with `python3 route.py --nudge "lets implement"`.

## Install

```bash
./install.sh             # symlinks ~/.claude/hooks/router -> this repo, adds the hooks to ~/.claude/settings.json
./install.sh --uninstall # removes the hook entries and the symlink
```

The installer is idempotent and writes `settings.json.bak-<timestamp>` before each change. It adds:

```json
"hooks": {
  "PreToolUse":       [ { "matcher": "Agent", "hooks": [ { "type": "command", "command": "python3 ~/.claude/hooks/router/route.py", "timeout": 10 } ] } ],
  "UserPromptSubmit": [ {                     "hooks": [ { "type": "command", "command": "python3 ~/.claude/hooks/router/route.py", "timeout": 10 } ] } ],
  "SessionStart":     [ { "hooks": [ … same command … ] } ],
  "SessionEnd":       [ { "hooks": [ … same command … ] } ]
}
```

Because `~/.claude/hooks/router` is a symlink, edits in this repo take effect on the next Agent call
with no reinstall or restart.

## Editing rules

Both files are re-read on every call.

Settings are split across two files:

- [`rules.json`](rules.json) is hand-owned: mode, Jev, tags, skip list, nudge and learning settings.
- [`tiers.json`](tiers.json) holds the tier definitions. The learner edits only this file, and only the `description` and `examples` fields.

Per-tier fields in `tiers.json`:

- `<tier>.description` is the option text Jev sees. This is the main tuning knob.
- `<tier>.examples` holds boundary examples. Jev sees them as `{"description", "examples"}`, capped at `learning.max_examples_per_tier`. They're normally written by the learner.
- `<tier>.signals` maps `regex → weight` for the offline fallback.
- `long_run.structural` adds weight when the prompt names ≥ N file paths or has ≥ N words.
- `<tier>.action`: `allow` + `set` (fields merged into the Agent input), or `deny` + `message`
  (`{reason}` / `{tier}` placeholders). For example, to send middle work to pi instead, change its message
  to point at the `pi-dev` skill.
- `jev.min_confidence`: below this, the router escalates to the heavier of Jev's top two tiers.
- `implement_nudge`: the trigger regex, the Jev question, `min_probability` and the message for the implementation nudge.
- `default_tier` is used when the rules find no signal.

**Per-project overrides:** these live in `<repo>/.claude/` or `<repo>/.agents/`, and `.claude` wins if both exist.

- `router.json` is deep-merged over `rules.json`, e.g. `{ "jev": { "enabled": false } }`.
- `tiers.json` is merged over the global tiers. `description`, `action` and `set` replace the global values, and `examples` are added to the global ones (project examples first). For example:
  `{ "middle": { "action": "allow", "set": { "model": "sonnet" } } }`

When a repo has its own `tiers.json`, the learner writes that repo's cases there, not to the global file.
To opt a repo into local learning, create an empty `{}` file.

**Check a change:**

```bash
python3 route.py --check                                  # validate rules.json (+ project override in cwd)
python3 route.py --explain "rename foo to bar in a.py"    # full decision incl. Jev probabilities
python3 route.py --explain "..." --no-jev                 # rules only
python3 -m pytest -q tests                                # test suite (Jev stubbed, no network)
```

**Measure a change:** `evals/cases.json` holds hand-labelled prompts for the tiers and the nudge.
`python3 evals/run.py` scores them against live Jev (`--no-jev` scores the regex fallback). Run it after
editing a tier `description` or the nudge question, and add real misroutes from the log as new cases.

**Tune from real traffic:** every decision is appended to `~/.claude/logs/router.jsonl`, with the tier, source,
Jev probabilities, rule scores, matched signals and the first 200 prompt chars. Override the path with
`CLAUDE_ROUTER_LOG`.

```bash
tail -n 20 ~/.claude/logs/router.jsonl | jq -c '{tier, source, reason, d: .description}'
```

## Self-learning

When Jev is unsure (confidence < `jev.min_confidence`), the current call escalates as usual and the case
is appended to `learning/queue.jsonl`. When the session ends, a `SessionEnd` hook starts `learn.py` in the background:

```
queued case ─► Sonnet (claude -p --bare, no tools, router off) ─► {tier, why, description_update?}
            ─► candidate tiers file: case becomes a boundary example under its tier (+ optional description rewrite)
            ─► GATE (live Jev): every eval case that passed before still passes,
                                and the new case now routes to Sonnet's tier with confidence ≥ min_confidence
                 ✓ write tiers file, append the case to the eval set, record history
                 ✗ retry without the description rewrite, then give up ─► learning/rejected.jsonl
```

- **Where it writes:** the repo's `.claude|.agents/tiers.json` if it has one, otherwise the global `tiers.json`.
  Accepted cases go to `evals/cases.json` (global) or `tiers.cases.json` next to the project file. A project
  gate runs the global cases plus that project's own cases.
- **What it may change:** only `description` and `examples`. It never touches actions, models, messages or regex signals.
- **Caps:** at most `max_examples_per_tier` examples (newest first), descriptions up to `max_description_chars`,
  example excerpts up to `example_chars`. Prompts that look like they contain secrets are never learned.
- **What you see:** at the start of your next session, a `SessionStart` hook prints what was learned or rejected, and each notice is shown once.
- **No Jev token when the learner runs:** cases stay queued for next time, because without Jev there's no gate.
- **Commands:**

```bash
python3 learn.py --status              # queue size, learned and rejected cases
python3 learn.py --dry-run             # label and gate the queue, write nothing
python3 learn.py                       # process the queue now
python3 learn.py --revert [tiers.json] # undo the last learned change to that file (default: global)
```

Turn it off with `"learning": {"enabled": false}` in `rules.json` or a project `router.json`.

## Disabling

| Scope | How |
|---|---|
| One call | put `[route:off]` in the Agent prompt |
| One session | `CLAUDE_ROUTER=off claude` |
| Everywhere, soft | `"mode": "advise"`: nothing is blocked or rewritten, and Claude only gets a "Router suggestion" note |
| Everywhere, off | `"mode": "off"` |
| Only the implementation nudge | `"implement_nudge": {"enabled": false}` |
| Uninstall | `./install.sh --uninstall` (or remove the entry via `/hooks`) |

Forcing a tier for one call: `[route:haiku]`, `[route:junie]`, `[route:opus]`, `[route:long]`.

## Files

- `route.py` holds the hook, the classifier and the CLI. It uses only the Python 3 standard library.
- `rules.json` holds the mode, Jev, tags, nudge and learning settings.
- `tiers.json` holds the tier descriptions, learned examples, signals and actions.
- `learn.py` is the self-learning loop (Sonnet labels, the Jev gate applies).
- `learning/` is gitignored and holds the queue, history, rejected cases, notices and the learner log.
- `install.sh` installs and uninstalls the hooks in `~/.claude` (PreToolUse, UserPromptSubmit, SessionStart, SessionEnd).
- `evals/` holds the labelled routing cases and the accuracy runner (live Jev).
- `tests/test_route.py` has the pytest suite.
