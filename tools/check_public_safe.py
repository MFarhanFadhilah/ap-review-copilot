"""Fail if the repo contains anything that must not be published.

    python tools/check_public_safe.py                                  # every check that needs no private data
    python tools/check_public_safe.py --git-history                    # also scan every commit (git log -p --all)
    python tools/check_public_safe.py --private-data /path/to/file.xls # also look for strings from a private file

Checks:

1. **Denied words**, read from ``.denylist`` (git-ignored, so the list itself is
   never published). One entry per line; ``#`` starts a comment. Matching is
   case-sensitive and whole-word: an entry matches only when no letter, digit
   or underscore touches it on either side, so a short entry never matches
   inside a longer word.
2. **Tokens in the old anonymisation scheme**: ``Word`` followed by digits.
   The synthetic data uses Ven/Tok/Acc tokens, so any such token is a leak.
3. **Multi-token strings from a private file** (``--private-data``): every
   value of Col1, Col4 or Col6 with two or more tokens. The file is read
   only from the path given and nothing from it is written anywhere.
4. **File rules**: no file over 5 MB, no ``.xls`` file.

What is scanned: every file git would commit (tracked plus untracked files
that are not ignored). Text files are scanned line by line. ``.xlsx`` files
are opened and every cell is scanned. Other binary files are listed as not
scanned. With ``--git-history`` the full patch history and every commit
message are scanned too (checks 1 to 3).

Exit code 0 means clean; 1 means at least one finding.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 5 * 1024 * 1024
OLD_TOKEN = re.compile(r"(?<![A-Za-z0-9_])Word\d+(?![A-Za-z0-9_])")
WORD = re.compile(r"[A-Za-z0-9_]+")


# --------------------------------------------------------------------------
# What to look for
# --------------------------------------------------------------------------
def load_denylist(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    return [e for e in entries if e]


def denylist_pattern(entries: list[str]) -> re.Pattern | None:
    """One regex for all entries: case-sensitive, whole-word (no letter, digit or _ on either side)."""
    if not entries:
        return None
    alternatives = "|".join(re.escape(e) for e in sorted(entries, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])")


def private_strings(path: Path) -> tuple[set[str], int]:
    """Multi-token values of Col1, Col4 and Col6, normalised to single-space-joined words."""
    import pandas as pd

    df = pd.read_excel(path) if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(path, dtype=str)
    found = set()
    for col in ("Col1", "Col4", "Col6"):
        for value in df[col].dropna().astype(str).unique():
            words = WORD.findall(value)
            if len(words) >= 2:
                found.add(" ".join(words))
    longest = max((len(s.split()) for s in found), default=0)
    return found, longest


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------
def scan_text(lines, deny: re.Pattern | None, private: set[str], longest: int):
    """Yield ``(line_number, reason)`` for every finding in an iterable of lines."""
    for n, line in enumerate(lines, start=1):
        if deny is not None:
            for m in deny.finditer(line):
                yield n, f"denied word {m.group(0)!r}"
        for m in OLD_TOKEN.finditer(line):
            yield n, f"old-scheme token {m.group(0)!r}"
        if private:
            words = WORD.findall(line)
            hit = next((" ".join(words[i:i + k]) for k in range(2, longest + 1) for i in range(len(words) - k + 1)
                        if " ".join(words[i:i + k]) in private), None)
            if hit:
                yield n, f"string from the private file ({len(hit.split())} tokens)"


def repo_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                         cwd=ROOT, check=True, capture_output=True).stdout.decode()
    return [ROOT / p for p in out.split("\0") if p and (ROOT / p).is_file()]


def xlsx_lines(path: Path) -> list[str]:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    return [" | ".join(row.dropna()) for frame in sheets.values() for _, row in frame.iterrows()]


def check_files(deny, private, longest) -> tuple[list[str], list[str]]:
    findings, skipped = [], []
    for path in repo_files():
        rel = path.relative_to(ROOT)
        if rel.name == ".denylist":
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            findings.append(f"{rel}: file is {size / 1e6:.1f} MB (limit 5 MB)")
        if path.suffix.lower() == ".xls":
            findings.append(f"{rel}: .xls files are not allowed")
        if path.suffix.lower() == ".xlsx":
            lines = xlsx_lines(path)
        else:
            data = path.read_bytes()
            if b"\0" in data[:8192]:
                skipped.append(str(rel))
                continue
            lines = data.decode("utf-8", errors="replace").splitlines()
        findings += [f"{rel}:{n}: {reason}" for n, reason in scan_text(lines, deny, private, longest)]
    return findings, skipped


def check_history(deny, private, longest) -> list[str]:
    log = subprocess.run(["git", "log", "--all", "-p", "--format=commit %H%n%B"], cwd=ROOT,
                         capture_output=True, check=False).stdout.decode("utf-8", errors="replace")
    findings, commit, path = [], "?", "?"
    for n, line in enumerate(log.splitlines(), start=1):
        if line.startswith("commit "):
            commit = line.split()[1][:8]
        elif line.startswith("+++ b/"):
            path = line[6:]
        findings += [f"git history {commit} ({path}), log line {n}: {reason}"
                     for _, reason in scan_text([line], deny, private, longest)]
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fail if the repo contains anything that must not be published.")
    parser.add_argument("--denylist", default=str(ROOT / ".denylist"))
    parser.add_argument("--private-data", help="path to a private data file (read only, never copied)")
    parser.add_argument("--git-history", action="store_true", help="also scan git log -p --all")
    args = parser.parse_args(argv)

    entries = load_denylist(Path(args.denylist))
    deny = denylist_pattern(entries)
    private, longest = private_strings(Path(args.private_data)) if args.private_data else (set(), 0)
    print(f"denylist: {len(entries)} entries" + ("" if entries else " (no .denylist file: denied-word check skipped)"))
    print(f"private data: {len(private):,} multi-token strings" if args.private_data else
          "private data: not given (check skipped)")

    findings, skipped = check_files(deny, private, longest)
    print(f"scanned {len(repo_files())} files; binary files not scanned: {', '.join(skipped) or 'none'}")
    if args.git_history:
        history = check_history(deny, private, longest)
        print(f"git history: {len(history)} finding(s)")
        findings += history
    for f in findings:
        print("FAIL", f)
    print("clean" if not findings else f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
