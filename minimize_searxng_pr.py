from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path


NONCORE_PATHS = [
    # Windows portability (local dev convenience)
    "searx/webutils.py",
    "searx/valkeydb.py",
    # i18n (local dev convenience for Chinese UI)
    "searx/translations/zh_Hans_CN/LC_MESSAGES/messages.po",
    "searx/translations/zh_Hans_CN/LC_MESSAGES/messages.mo",
    "searx/translations/zh_Hant_TW/LC_MESSAGES/messages.po",
    "searx/translations/zh_Hant_TW/LC_MESSAGES/messages.mo",
]


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export non-core local modifications (Windows+i18n) as a patch, then restore them in vendor/searxng "
            "so the remaining diff is suitable for a minimal upstream NoAI PR"
        )
    )
    ap.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[1] / "vendor" / "searxng"),
        help="Path to the vendored searxng git repo (default: ./vendor/searxng)",
    )
    ap.add_argument(
        "--patch-out",
        default="",
        help="Where to write the exported patch (default: ./noncore_local.patch)",
    )
    ap.add_argument(
        "--restore",
        action="store_true",
        help="Actually restore the non-core paths back to HEAD after exporting the patch",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"Not a git repo: {repo}", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    patch_out = Path(args.patch_out).resolve() if args.patch_out else (root / "noncore_local.patch").resolve()

    diff = _run(["git", "-C", str(repo), "diff", "--", *NONCORE_PATHS])
    if diff.returncode != 0:
        print(diff.stderr.strip() or "git diff failed", file=sys.stderr)
        return diff.returncode

    header = (
        f"# Exported non-core local changes\n"
        f"# Generated: {_dt.datetime.now().astimezone().isoformat()}\n"
        f"# Repo: {repo}\n"
        f"# Paths: {', '.join(NONCORE_PATHS)}\n\n"
    )
    patch_out.parent.mkdir(parents=True, exist_ok=True)
    patch_out.write_text(header + diff.stdout, encoding="utf-8")
    print(str(patch_out))

    if not args.restore:
        return 0

    restore = _run(["git", "-C", str(repo), "restore", "--", *NONCORE_PATHS])
    if restore.returncode != 0:
        print(restore.stderr.strip() or "git restore failed", file=sys.stderr)
        return restore.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
