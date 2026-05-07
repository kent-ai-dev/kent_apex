"""
Ralph loop — the continuous trainer/director for RCE.

Architecture:
  - Reads PLAN.md once at startup (the spec)
  - Each iteration: reads state.json, LOG.md, BENCHMARKS.md, code files
  - Composes a prompt to Claude (the engineer) using the template in PLAN.md §5
  - Claude returns: STATE_ASSESSMENT, NEXT_ACTION, EXPECTED_OUTCOME,
    UPDATE_LOG, UPDATE_BENCHMARKS — and performs the action via tool use
  - Ralph applies any tool calls, captures outputs, updates state, sleeps

Safety rails:
  - Per-iteration timeout (default 60 min)
  - Append-only LOG.md (every iteration adds a section, never edits prior)
  - Required human review every 10 iterations (PAUSE_FOR_REVIEW token)
  - ESCALATE token in LOG.md halts the loop
  - Modal cost / disk quota stubs (fill in for your environment)

This file is the conductor. It contains zero RCE logic. The intelligence
lives in PLAN.md (the spec) and the LLM (the engineer).

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python ralph.py --workspace /workspace/rce --interval 30m
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# anthropic is the only external dep
try:
    import anthropic
except ImportError:
    print("install: pip install anthropic")
    sys.exit(1)


# ---------- config ----------

DEFAULT_MODEL = "claude-opus-4-7"  # the engineer
ITERATION_TIMEOUT_SEC = 60 * 60
PER_API_CALL_TIMEOUT_SEC = 600
TOOL_USE_LOOP_CAP = 20
HUMAN_REVIEW_EVERY_N_ITERS = 10
ESCALATE_TOKEN = "ESCALATE"
PAUSE_TOKEN = "PAUSE_FOR_REVIEW"

# Bash deny-list. Match-on-substring/regex against the command before running.
# Goal: prevent destruction outside the workspace, system-level damage, and
# silent network exfil. Does NOT block legitimate workspace-local rm/mv/etc.
BASH_DENY_PATTERNS: list[tuple[str, str]] = [
    (r"\bsudo\b", "sudo not allowed"),
    (r"\brm\s+-[rRfF]+\s+(/[^\s]|~|\$HOME|\.\.)", "destructive rm to absolute, home, or parent path"),
    (r"\bdd\s+if=", "dd not allowed"),
    (r"\bmkfs\b", "mkfs not allowed"),
    (r"chmod\s+-R\s+0*7?77\b", "chmod -R 777 not allowed"),
    (r":\(\)\s*\{", "fork bomb pattern"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "system power commands not allowed"),
    (r"git\s+push\b[^\n]*--force", "force push not allowed"),
    (r"\bcurl\b[^|\n]*\|\s*(bash|sh|zsh)\b", "curl|sh pipe not allowed"),
    (r"\bwget\b[^|\n]*\|\s*(bash|sh|zsh)\b", "wget|sh pipe not allowed"),
    (r">\s*/dev/sd[a-z]", "writing to raw block devices not allowed"),
]


class IterationTimeout(Exception):
    """Raised when an iteration exceeds ITERATION_TIMEOUT_SEC."""


def _alarm_handler(signum, frame):
    raise IterationTimeout()


# ---------- state ----------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_state(workspace: Path) -> dict:
    p = workspace / "state.json"
    if not p.exists():
        return {"version": 1, "stage": "DESIGN", "iteration": 0,
                "last_gate": "in_progress", "last_human_review": now_iso()}
    return json.loads(p.read_text())


def write_state(workspace: Path, state: dict):
    (workspace / "state.json").write_text(json.dumps(state, indent=2))


def append_log(workspace: Path, content: str):
    log = workspace / "LOG.md"
    with log.open("a") as f:
        f.write(f"\n---\n## {now_iso()}\n\n{content}\n")


def append_bench(workspace: Path, rows: list[dict]):
    bench = workspace / "BENCHMARKS.md"
    if not bench.exists():
        bench.write_text("# Benchmarks\n\n| date | version | metric | value | notes |\n|---|---|---|---|---|\n")
    with bench.open("a") as f:
        for r in rows:
            f.write(f"| {r.get('date', now_iso())} | {r.get('version', '?')} | "
                    f"{r.get('metric', '?')} | {r.get('value', '?')} | "
                    f"{r.get('notes', '')} |\n")


# ---------- file gathering ----------

def read_workspace_context(workspace: Path) -> dict[str, str]:
    """Read the canonical files Claude needs each iteration."""
    files = {}
    for name in ["PLAN.md", "LOG.md", "BENCHMARKS.md", "state.json",
                 "engine.py", "rce.py"]:
        p = workspace / name
        if p.exists():
            files[name] = p.read_text()
        else:
            files[name] = "(file not present yet)"
    # also list any V{version}-related files
    state = read_state(workspace)
    v = state["version"]
    pattern = f"v{v}_"
    extra = [p for p in workspace.glob(f"{pattern}*") if p.is_file()]
    for p in extra:
        files[p.name] = p.read_text()
    return files


# ---------- prompt construction ----------

def build_prompt(workspace: Path, state: dict, files: dict[str, str]) -> str:
    """Build the iteration prompt, following PLAN.md §5."""
    hours_since_review = 0.0
    if state.get("last_human_review"):
        last = datetime.fromisoformat(state["last_human_review"])
        hours_since_review = (datetime.now(timezone.utc) - last).total_seconds() / 3600

    file_section = ""
    for name, content in files.items():
        # truncate huge files to last 8KB to avoid blowing context
        if len(content) > 8000:
            content = "...(truncated)...\n" + content[-8000:]
        file_section += f"\n### FILE: {name}\n```\n{content}\n```\n"

    prompt = f"""You are Claude, acting as the engineer in a Ralph loop training the RCE.

CURRENT STATE:
- Version: {state['version']}
- Stage: {state['stage']}
- Iteration: {state['iteration']}
- Last gate result: {state['last_gate']}
- Hours since last human review: {hours_since_review:.1f}

YOUR TASK THIS ITERATION (per PLAN.md §5):

Read the workspace files below. Determine the current stage of V{state['version']}.
Choose ONE focused next action. Output the response in the structured format
defined in PLAN.md §5: STATE_ASSESSMENT, NEXT_ACTION, EXPECTED_OUTCOME,
UPDATE_LOG, UPDATE_BENCHMARKS.

Use the available tools (str_replace, create_file, bash) to perform the action.

If you believe a gate has been failed twice or you need human input, write
"{ESCALATE_TOKEN}: <reason>" prominently in your UPDATE_LOG and stop without
performing further actions.

If hours_since_human_review is high or you've completed a major version,
write "{PAUSE_TOKEN}: <reason>" in your UPDATE_LOG to request review.

WORKSPACE FILES:
{file_section}

Begin.
"""
    return prompt


# ---------- LLM call ----------

def call_engineer(prompt: str, model: str, api_key: str,
                  workspace: Path) -> dict:
    """Send the iteration prompt to Claude and let it act via tool use.

    For simplicity in this skeleton, we use a single text-completion call
    and parse the structured response. A production version would use the
    full tool-use loop (file edits, bash calls) — but that's a deeper
    integration. This skeleton's job is to demonstrate the orchestration.
    """
    client = anthropic.Anthropic(api_key=api_key, timeout=PER_API_CALL_TIMEOUT_SEC)

    # Define the tools Claude can use to act on the workspace
    tools = [
        {
            "name": "read_file",
            "description": "Read a file from the workspace",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "run_bash",
            "description": "Run a bash command in the workspace directory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 600},
                },
                "required": ["command"],
            },
        },
    ]

    messages = [{"role": "user", "content": prompt}]
    final_text = ""
    tool_results = []
    cap_hit = False
    last_stop_reason: str | None = None

    for turn in range(TOOL_USE_LOOP_CAP):
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            tools=tools,
            messages=messages,
        )
        last_stop_reason = resp.stop_reason

        # collect THIS turn's text only — overwriting prior turns so the
        # final value is the engineer's terminating message, not a concat
        # of every intermediate think-aloud
        turn_text = ""
        for block in resp.content:
            if block.type == "text":
                turn_text += block.text + "\n"
        if turn_text:
            final_text = turn_text

        if resp.stop_reason != "tool_use":
            break

        if turn == TOOL_USE_LOOP_CAP - 1:
            cap_hit = True
            break

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": resp.content})

        results_for_msg = []
        for tu in tool_uses:
            try:
                result = execute_tool(tu.name, tu.input, workspace)
                ok = True
            except Exception as e:
                result = f"ERROR: {e}"
                ok = False
            tool_results.append({"name": tu.name, "input": tu.input,
                                 "ok": ok, "result_preview": str(result)[:300]})
            results_for_msg.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": str(result)[:8000],
                "is_error": not ok,
            })
        messages.append({"role": "user", "content": results_for_msg})

    return {
        "text": final_text,
        "tool_calls": tool_results,
        "cap_hit": cap_hit,
        "stop_reason": last_stop_reason,
    }


def _path_in_workspace(path: Path, workspace: Path) -> bool:
    """True iff `path` is `workspace` or a descendant of it. Resolves both
    sides first so symlinks can't be used to escape. Uses Path.is_relative_to
    rather than string-prefix to avoid /workspace/rce-evil bypassing /workspace/rce.
    """
    try:
        path.resolve().relative_to(workspace.resolve())
        return True
    except ValueError:
        return False


def _bash_deny_reason(command: str) -> str | None:
    """Return the deny-list reason if `command` matches a forbidden pattern."""
    for pattern, reason in BASH_DENY_PATTERNS:
        if re.search(pattern, command):
            return reason
    return None


def execute_tool(name: str, args: dict, workspace: Path):
    """Sandbox tool execution to the workspace dir."""
    if name == "read_file":
        path = (workspace / args["path"]).resolve()
        if not _path_in_workspace(path, workspace):
            raise ValueError("path escapes workspace")
        return path.read_text() if path.exists() else "(not found)"

    if name == "write_file":
        path = (workspace / args["path"]).resolve()
        if not _path_in_workspace(path, workspace):
            raise ValueError("path escapes workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return f"wrote {len(args['content'])} bytes to {path}"

    if name == "run_bash":
        command = args["command"]
        deny = _bash_deny_reason(command)
        if deny:
            return f"BLOCKED: {deny}\ncommand was: {command[:500]}"
        try:
            timeout = int(args.get("timeout_sec", 600))
        except (TypeError, ValueError):
            timeout = 600
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=str(workspace), timeout=timeout,
        )
        return (f"exit={proc.returncode}\n"
                f"---STDOUT---\n{proc.stdout[-4000:]}\n"
                f"---STDERR---\n{proc.stderr[-2000:]}")

    raise ValueError(f"unknown tool: {name}")


# ---------- safety checks ----------

def check_safety(workspace: Path, prospective_iteration: int) -> str | None:
    """Return a reason to pause, or None to proceed.

    Once an ESCALATE token is in LOG.md it halts the loop forever; the
    operator must edit LOG.md to clear it. The review checkpoint fires
    BEFORE iteration N runs, so the iteration counter is not consumed
    by a paused turn.
    """
    log_path = workspace / "LOG.md"
    if log_path.exists():
        log = log_path.read_text()
        if ESCALATE_TOKEN in log:
            return f"{ESCALATE_TOKEN} present in LOG.md (clear it manually to resume)"
    if prospective_iteration > 0 and prospective_iteration % HUMAN_REVIEW_EVERY_N_ITERS == 0:
        return f"reached {HUMAN_REVIEW_EVERY_N_ITERS}-iteration review checkpoint"
    try:
        free_gb = shutil.disk_usage(workspace).free / (1024 ** 3)
        if free_gb < 1.0:
            return f"low disk: {free_gb:.2f} GB free"
    except OSError:
        pass
    return None


# ---------- main loop ----------

_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([smh])\s*$", re.IGNORECASE)


def parse_interval(s: str) -> int:
    """Parse '30m', '2h', '45s' → seconds. Raises ValueError on malformed input."""
    m = _INTERVAL_RE.match(s)
    if not m:
        raise ValueError(f"bad --interval {s!r}; expected like '30s', '5m', '2h'")
    n = int(m.group(1))
    unit = m.group(2).lower()
    return {"s": 1, "m": 60, "h": 3600}[unit] * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--interval", default="30m",
                    help="sleep between iterations: 30s, 5m, 1h")
    ap.add_argument("--max-iters", type=int, default=1000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    ap.add_argument("--once", action="store_true",
                    help="run a single iteration and exit (for testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompt and print it; do not call the API")
    args = ap.parse_args()

    if not args.api_key and not args.dry_run:
        print("ERROR: set ANTHROPIC_API_KEY or pass --api-key (or use --dry-run)")
        sys.exit(1)

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        sleep_sec = parse_interval(args.interval)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Refresh the human-review timestamp on every script start. The "hours
    # since last human review" metric reported to the engineer is meant to be
    # "hours this autonomous run has been going" — restarting counts as review.
    state = read_state(workspace)
    state["last_human_review"] = now_iso()
    write_state(workspace, state)

    # Install the per-iteration alarm handler once; arm it per-iteration below.
    signal.signal(signal.SIGALRM, _alarm_handler)

    print(f"Ralph loop starting")
    print(f"  workspace: {workspace}")
    print(f"  interval: {args.interval} ({sleep_sec}s)")
    print(f"  model: {args.model}")
    print(f"  max iterations: {args.max_iters}")
    if args.dry_run:
        print(f"  DRY-RUN: no API calls, no state mutation")

    for _ in range(args.max_iters):
        state = read_state(workspace)
        prospective = state.get("iteration", 0) + 1

        pause_reason = check_safety(workspace, prospective)
        if pause_reason:
            print(f"\nPAUSING at iteration {prospective}: {pause_reason}")
            if not args.dry_run:
                append_log(workspace, f"**PAUSED before iteration {prospective}**: {pause_reason}")
            break

        # commit this iteration: increment counter only when we will actually run it
        state["iteration"] = prospective
        print(f"\n=== iteration {state['iteration']} (V{state['version']} {state['stage']}) ===")

        files = read_workspace_context(workspace)
        prompt = build_prompt(workspace, state, files)

        if args.dry_run:
            print("--- prompt preview (first 2000 chars) ---")
            print(prompt[:2000])
            print("--- end preview ---")
            break

        signal.alarm(ITERATION_TIMEOUT_SEC)
        try:
            t0 = time.time()
            result = call_engineer(prompt, args.model, args.api_key, workspace)
            elapsed = time.time() - t0
            print(f"  engineer responded in {elapsed:.1f}s; "
                  f"{len(result['tool_calls'])} tool calls; "
                  f"stop={result.get('stop_reason')}"
                  + ("  CAP_HIT" if result.get("cap_hit") else ""))

            summary = (
                f"**Iteration {state['iteration']}** "
                f"(V{state['version']} {state['stage']})\n\n"
                f"Engineer final text:\n```\n{result['text'][:2000]}\n```\n\n"
                f"Tool calls: {len(result['tool_calls'])}  "
                f"stop_reason={result.get('stop_reason')}\n"
            )
            if result.get("cap_hit"):
                summary += (f"\n**WARNING**: tool-use loop hit cap "
                            f"({TOOL_USE_LOOP_CAP}); engineer turn was truncated.\n")
            for tc in result["tool_calls"][:5]:
                summary += (f"  - {tc['name']}({list(tc['input'].keys())}): "
                            f"{'ok' if tc['ok'] else 'FAIL'}\n")
            append_log(workspace, summary)
            write_state(workspace, state)

        except IterationTimeout:
            print(f"  ITERATION TIMEOUT after {ITERATION_TIMEOUT_SEC}s")
            append_log(workspace,
                       f"**Iteration {state['iteration']} TIMEOUT** "
                       f"after {ITERATION_TIMEOUT_SEC}s wall clock")
            write_state(workspace, state)
        except Exception as e:
            print(f"  ERROR: {e}")
            append_log(workspace, f"**Iteration {state['iteration']} ERROR**: {e}")
            write_state(workspace, state)
        finally:
            signal.alarm(0)

        if args.once:
            print("--once specified; exiting")
            break

        print(f"  sleeping {sleep_sec}s")
        time.sleep(sleep_sec)

    print("Ralph loop done.")


if __name__ == "__main__":
    main()