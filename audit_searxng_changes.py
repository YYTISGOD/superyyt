from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CmdResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: list[str], *, cwd: Path | None = None) -> CmdResult:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CmdResult(cmd=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def _ensure_git() -> str:
    r = _run(["git", "--version"])
    if r.returncode != 0:
        raise RuntimeError(f"git not available: {r.stderr.strip()}")
    return r.stdout.strip() or "git"


def _md_code_block(text: str, lang: str = "") -> str:
    text = text.rstrip("\n")
    return f"```{lang}\n{text}\n```\n"


def _classify_path(path: str) -> str:
    path = path.replace("\\", "/")
    if path.startswith("searx/plugins/noai_"):
        return "NoAI plugin"
    if path.startswith("searx/templates/simple/preferences/noai_"):
        return "NoAI preferences UI"
    if path in {
        "searx/preferences.py",
        "searx/templates/simple/preferences.html",
        "searx/templates/simple/macros.html",
        "searx/templates/simple/result_templates/paper.html",
    }:
        return "NoAI integration"
    if path.startswith("searx/translations/"):
        return "i18n"
    if path in {"searx/webutils.py", "searx/valkeydb.py"}:
        return "Windows portability"
    return "Other"


def _parse_porcelain(porcelain: str) -> tuple[list[str], list[str]]:
    modified: list[str] = []
    untracked: list[str] = []
    for line in porcelain.splitlines():
        if not line:
            continue
        if line.startswith("?? "):
            untracked.append(line[3:])
            continue
        # XY PATH (we just keep PATH)
        if len(line) > 3:
            modified.append(line[3:])
    return modified, untracked


def _build_report(repo: Path, full_diff: bool) -> str:
    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    head = _run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
    status_sb = _run(["git", "-C", str(repo), "status", "-sb"])
    status = _run(["git", "-C", str(repo), "status", "--porcelain=v1"])
    diff_stat = _run(["git", "-C", str(repo), "diff", "--stat"])

    if status.returncode != 0:
        raise RuntimeError(status.stderr.strip() or "git status failed")

    modified, untracked = _parse_porcelain(status.stdout)

    lines: list[str] = []
    lines.append(f"# SearXNG change audit\n")
    lines.append(f"Generated: {now}\n")
    lines.append(f"Repo: {repo}\n")
    lines.append("\n")

    lines.append("## Baseline\n")
    if head.returncode == 0:
        lines.append(f"- HEAD: `{head.stdout.strip()}`\n")
    if status_sb.returncode == 0:
        lines.append("\n" + _md_code_block(status_sb.stdout, "text"))

    lines.append("## Working tree status\n")
    lines.append(_md_code_block(status.stdout or "(clean)", "text"))

    lines.append("## Diff summary\n")
    lines.append(_md_code_block(diff_stat.stdout or "(no diff)", "text"))

    def _section(title: str, paths: list[str]):
        lines.append(f"## {title}\n")
        if not paths:
            lines.append("(none)\n\n")
            return
        for p in sorted(paths):
            lines.append(f"- `{p}` — {_classify_path(p)}\n")
        lines.append("\n")

    _section("Modified files", modified)
    _section("Untracked files", untracked)

    noise = [p for p in untracked if p.startswith(".cache/") or p.startswith("__pycache__/") or p.startswith("node_modules/")]
    if noise:
        lines.append("## Noise candidates\n")
        lines.append(
            "These are usually build artifacts / caches and should not be included in a PR diff:\n\n"
        )
        for p in sorted(noise):
            lines.append(f"- `{p}`\n")
        lines.append("\n")

    if full_diff:
        diff = _run(["git", "-C", str(repo), "diff"])
        lines.append("## Full diff\n")
        if diff.returncode != 0:
            lines.append(_md_code_block(diff.stderr.strip() or "git diff failed", "text"))
        else:
            lines.append(_md_code_block(diff.stdout or "(empty)", "diff"))

    lines.append("## Suggested PR split\n")
    lines.append(
        "If you want the smallest possible upstream PR for *NoAI* only, consider splitting changes into:\n\n"
    )
    lines.append("- NoAI feature: plugin + preferences + templates\n")
    lines.append("- i18n: translations (po/mo)\n")
    lines.append("- Windows portability: `searx/webutils.py`, `searx/valkeydb.py` (may be a separate PR)\n")

    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit local modifications in vendored SearXNG (git-based, line-by-line)")
    ap.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[1] / "vendor" / "searxng"),
        help="Path to the vendored searxng git repository (default: ./vendor/searxng)",
    )
    ap.add_argument(
        "--output",
        default="",
        help="Write report to this path (markdown). If omitted, print to stdout",
    )
    ap.add_argument("--full-diff", action="store_true", help="Include full unified diff (may be large)")
    args = ap.parse_args()

    _ensure_git()

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a git repo: {repo}")

    report = _build_report(repo, full_diff=bool(args.full_diff))

    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(str(out))
        return 0

    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
