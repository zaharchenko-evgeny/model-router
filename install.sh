#!/usr/bin/env bash
# Install / uninstall the router hook into ~/.claude.
#   ./install.sh            symlink ~/.claude/hooks/router -> this repo, register the PreToolUse hook
#   ./install.sh --uninstall remove the hook entry and the symlink
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
LINK="$CLAUDE_DIR/hooks/router"
SETTINGS="$CLAUDE_DIR/settings.json"
COMMAND="python3 $LINK/route.py"

patch_settings() {  # $1 = install|uninstall
  python3 - "$SETTINGS" "$COMMAND" "$1" <<'PY'
import json, shutil, sys, time
from pathlib import Path
path, command, op = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
settings = json.loads(path.read_text()) if path.exists() else {}
if path.exists():
    shutil.copy2(path, f"{path}.bak-{time.strftime('%Y%m%d%H%M%S')}")
hooks = settings.setdefault("hooks", {})
for event, matcher in [("PreToolUse", "Agent"), ("UserPromptSubmit", None),
                       ("SessionStart", None), ("SessionEnd", None)]:
    groups = hooks.setdefault(event, [])
    # drop any previous router entry so install is idempotent
    for group in groups:
        group["hooks"] = [h for h in group.get("hooks", []) if h.get("command") != command]
    groups[:] = [g for g in groups if g.get("hooks")]
    if op == "install":
        entry = {"hooks": [{"type": "command", "command": command, "timeout": 10}]}
        if matcher:
            entry = {"matcher": matcher, **entry}
        groups.append(entry)
    if not groups:
        del hooks[event]
if not settings["hooks"]:
    del settings["hooks"]
path.write_text(json.dumps(settings, indent=2) + "\n")
print(f"{op}: {path} updated (backup written alongside)")
PY
}

if [[ "${1:-}" == "--uninstall" ]]; then
  patch_settings uninstall
  [[ -L "$LINK" ]] && rm "$LINK" && echo "removed symlink $LINK"
  exit 0
fi

mkdir -p "$CLAUDE_DIR/hooks"
if [[ -e "$LINK" && ! -L "$LINK" ]]; then
  echo "error: $LINK exists and is not a symlink; move it away first" >&2
  exit 1
fi
ln -sfn "$REPO" "$LINK"
echo "linked $LINK -> $REPO"
chmod +x "$REPO/route.py"
python3 "$REPO/route.py" --check
patch_settings install
