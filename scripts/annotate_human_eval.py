"""Interactive annotator for the faithfulness human-eval sample."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from textwrap import fill


DEFAULT_PATH = Path("evaluation/faithfulness/human_eval/template_andrea.csv")


def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except OSError:
        return default


def clear():
    print("\x1b[2J\x1b[H", end="")


def banner(text: str, ch: str = "=") -> str:
    w = max(40, min(term_width(), 100))
    return ch * w + "\n" + text + "\n" + ch * w


def show_claim(i, total, row, labeled_so_far):
    w = max(40, min(term_width(), 100))
    print(banner(f"  Claim {i + 1} of {total}   (already labeled: {labeled_so_far})"))
    print(f"  claim_uid:  {row['claim_uid']}")
    print(
        f"  question:   {row['question_id']}  ({row.get('difficulty', '?')}, {row.get('category', '?')})"
    )
    print(f"  paper:      {row['paper_id']}")
    print()
    print("  CLAIM:")
    print(fill(row["claim_text"], width=w - 4, initial_indent="    ", subsequent_indent="    "))
    print()
    print("  SUPPORTING QUOTE:")
    print(
        fill(row["supporting_quote"], width=w - 4, initial_indent="    ", subsequent_indent="    ")
    )
    print()
    print("-" * w)
    print("  Does the quote SUPPORT the claim?")
    print("    s = supported     n = not_supported")
    print("    c = also flag as needs_context     u = undo previous     q = save & quit")
    print("-" * w)


def read_label(prompt):
    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            return "q"
        if choice in {"s", "n", "c", "u", "q"}:
            return choice
        print("  please enter one of: s / n / c / u / q")


def save(rows, fieldnames, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    if not args.file.is_file():
        print(f"ERROR: not a file: {args.file}", file=sys.stderr)
        return 2

    with args.file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    for col in ("human_label", "needs_context", "notes"):
        if col not in fieldnames:
            fieldnames.append(col)
        for r in rows:
            r.setdefault(col, "")

    total = len(rows)
    labeled = sum(1 for r in rows if r.get("human_label"))
    print(banner(f"  human-eval annotator   ({total} claims, {labeled} already labeled)", ch="#"))
    print(f"  File: {args.file}")
    input("\n  Press Enter to begin (Ctrl-C any time, progress is auto-saved). ")

    i = 0
    last_idx = -1
    try:
        while i < total:
            row = rows[i]
            if row.get("human_label"):
                i += 1
                continue
            clear()
            labeled_now = sum(1 for r in rows if r.get("human_label"))
            show_claim(i, total, row, labeled_now)
            choice = read_label("  > ")

            if choice == "q":
                break
            if choice == "u":
                if last_idx < 0:
                    print("  nothing to undo.")
                    input("  Press Enter. ")
                    continue
                rows[last_idx]["human_label"] = ""
                rows[last_idx]["needs_context"] = ""
                save(rows, fieldnames, args.file)
                i = last_idx
                last_idx = -1
                continue

            needs_ctx = False
            if choice == "c":
                needs_ctx = True
                choice = read_label("  flagged needs_context. now label > ")
                if choice == "q":
                    break
                if choice == "u":
                    continue

            row["human_label"] = "supported" if choice == "s" else "not_supported"
            if needs_ctx:
                row["needs_context"] = "1"
            save(rows, fieldnames, args.file)
            last_idx = i
            i += 1
    except KeyboardInterrupt:
        print("\n  interrupted - progress saved.")

    labeled_final = sum(1 for r in rows if r.get("human_label"))
    flagged = sum(1 for r in rows if r.get("needs_context"))
    print()
    print(banner(f"  DONE. {labeled_final}/{total} labeled   ({flagged} flagged needs_context)"))
    print(f"  File: {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
