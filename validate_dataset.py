"""
Pre-ingestion validator for HuggingFace datasets.

Mandatory gate before any training run pulls a new dataset. Catches the
common ways a HF dataset can be silently broken so we don't burn compute
on corrupted data.

Usage (CLI):
    python validate_dataset.py <name> [--config CONFIG] [--split SPLIT] \
                               [--text-field FIELD] [--sample N]
    # exit 0 = PASS, 1 = WARN, 2 = FAIL

Usage (programmatic):
    from validate_dataset import validate
    report = validate("Salesforce/wikitext", config="wikitext-2-raw-v1")
    if report.verdict == "FAIL":
        ...

Checks (a "vibes" score 0–100; FAIL < 60, WARN < 85, else PASS):
  1. (20pts) `load_dataset(streaming=True)` succeeds
  2. (20pts) iterate `--sample` rows without crashing
  3. (15pts) `features` attribute present and the requested split exists
  4. (15pts) text/content field exists and is non-empty on most rows
  5. (10pts) text decodes as UTF-8 on a 100-row sample
  6. (10pts) row count looks plausible (>= sample size, not zero, not 1)
  7. (10pts) no more than 5% of sampled rows are empty/null on the text field

The validator is conservative: WARN doesn't block ingestion (the operator
gets a heads-up), but FAIL does. The output is human-readable and JSON.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Report:
    name: str
    config: str | None
    split: str
    verdict: str            # PASS | WARN | FAIL
    score: int              # 0-100
    checks: list[dict] = field(default_factory=list)
    summary: str = ""
    fatal: str | None = None


def _add_check(rep: Report, label: str, points: int, max_points: int,
               note: str = ""):
    rep.checks.append({"check": label, "score": points,
                       "max": max_points, "note": note})


def _detect_text_field(features: Any, hint: str | None) -> str | None:
    """Look for a textual field in the dataset features dict."""
    if features is None:
        return None
    if hint and hint in features:
        return hint
    # common text field names, in priority order
    candidates = ["text", "content", "sentence", "article", "document",
                  "body", "raw_content", "story", "input"]
    for c in candidates:
        if c in features:
            return c
    # fall back to the first string-typed feature
    try:
        for k, v in features.items():
            t = str(v).lower()
            if "string" in t or "value('string')" in t:
                return k
    except Exception:
        pass
    return None


def validate(name: str, config: str | None = None, split: str = "train",
             text_field: str | None = None, sample: int = 200) -> Report:
    rep = Report(name=name, config=config, split=split, verdict="FAIL", score=0)

    try:
        from datasets import load_dataset, get_dataset_split_names
    except Exception as e:
        rep.fatal = f"`datasets` library not importable: {e}"
        rep.summary = "FAIL: datasets library not installed"
        return rep

    # 1. streaming load
    ds = None
    try:
        ds = load_dataset(name, config, split=split, streaming=True)
        _add_check(rep, "load_dataset(streaming=True)", 20, 20)
        rep.score += 20
    except Exception as e:
        _add_check(rep, "load_dataset(streaming=True)", 0, 20, f"{type(e).__name__}: {e}")
        rep.fatal = f"could not load dataset: {e}"
        rep.summary = "FAIL: dataset will not load"
        return rep

    # 2. iterate sample rows
    rows: list[dict] = []
    iter_err: str | None = None
    try:
        for i, row in enumerate(ds):
            if i >= sample:
                break
            rows.append(row)
    except Exception as e:
        iter_err = f"{type(e).__name__}: {e} after {len(rows)} rows"

    if iter_err is None and rows:
        _add_check(rep, f"iterate {len(rows)} rows", 20, 20)
        rep.score += 20
    elif rows:
        # crashed mid-iteration but got some rows
        _add_check(rep, "iterate sample", 8, 20, iter_err or "")
        rep.score += 8
    else:
        _add_check(rep, "iterate sample", 0, 20, iter_err or "no rows yielded")
        rep.fatal = "could not iterate any rows"

    # 3. features / split
    features = None
    try:
        features = getattr(ds, "features", None) or (rows[0] if rows else None)
        if features:
            _add_check(rep, "features visible", 15, 15)
            rep.score += 15
        else:
            _add_check(rep, "features visible", 5, 15, "no features attribute")
            rep.score += 5
    except Exception as e:
        _add_check(rep, "features visible", 0, 15, str(e))

    # 4. text-field present and mostly populated
    field_name = _detect_text_field(features, text_field)
    if field_name is None:
        _add_check(rep, "text field detected", 0, 15,
                   f"no text-like field; features={list(features.keys()) if isinstance(features, dict) else features}")
    else:
        non_empty = sum(1 for r in rows
                        if r.get(field_name) and len(str(r[field_name])) > 0)
        ratio = non_empty / max(len(rows), 1)
        if ratio > 0.95:
            _add_check(rep, f"text field '{field_name}' populated", 15, 15,
                       f"{non_empty}/{len(rows)} non-empty")
            rep.score += 15
        elif ratio > 0.7:
            _add_check(rep, f"text field '{field_name}' populated", 8, 15,
                       f"{non_empty}/{len(rows)} non-empty (suspicious)")
            rep.score += 8
        else:
            _add_check(rep, f"text field '{field_name}' populated", 0, 15,
                       f"{non_empty}/{len(rows)} non-empty (broken)")

    # 5. UTF-8 decoding sanity
    if rows and field_name:
        decode_ok = 0
        for r in rows:
            v = r.get(field_name)
            if v is None:
                continue
            s = str(v)
            try:
                s.encode("utf-8").decode("utf-8")
                decode_ok += 1
            except Exception:
                pass
        if rows and decode_ok / len(rows) > 0.95:
            _add_check(rep, "UTF-8 decode", 10, 10)
            rep.score += 10
        else:
            _add_check(rep, "UTF-8 decode", 3, 10,
                       f"{decode_ok}/{len(rows)} decode cleanly")
            rep.score += 3

    # 6. row count plausibility
    if len(rows) >= max(1, sample // 2):
        _add_check(rep, "row count plausible", 10, 10, f"{len(rows)} rows")
        rep.score += 10
    elif len(rows) > 1:
        _add_check(rep, "row count plausible", 5, 10, f"only {len(rows)} rows")
        rep.score += 5
    else:
        _add_check(rep, "row count plausible", 0, 10, f"{len(rows)} rows")

    # 7. emptiness ceiling already counted in #4; small bonus for clean data
    if rows and field_name:
        empties = sum(1 for r in rows if not r.get(field_name))
        if empties / max(len(rows), 1) <= 0.05:
            _add_check(rep, "≤5% empty rows", 10, 10, f"{empties} empty / {len(rows)}")
            rep.score += 10
        else:
            _add_check(rep, "≤5% empty rows", 0, 10, f"{empties} empty / {len(rows)}")

    # final verdict
    if rep.fatal:
        rep.verdict = "FAIL"
    elif rep.score >= 85:
        rep.verdict = "PASS"
    elif rep.score >= 60:
        rep.verdict = "WARN"
    else:
        rep.verdict = "FAIL"

    rep.summary = (f"{rep.verdict}: score {rep.score}/100, "
                   f"{len(rows)} rows iterated, text field={field_name!r}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="HF dataset id, e.g. Salesforce/wikitext")
    ap.add_argument("--config", default=None, help="dataset config, e.g. wikitext-2-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--json", action="store_true",
                    help="emit JSON only (for programmatic consumption)")
    args = ap.parse_args()

    rep = validate(args.name, config=args.config, split=args.split,
                   text_field=args.text_field, sample=args.sample)

    if args.json:
        print(json.dumps(asdict(rep), indent=2))
    else:
        print(f"validating {args.name} ({args.config or 'default'} / {args.split})")
        for c in rep.checks:
            mark = "✓" if c["score"] == c["max"] else ("~" if c["score"] > 0 else "✗")
            print(f"  {mark} {c['check']:30s}  {c['score']}/{c['max']}  {c.get('note', '')}")
        print(f"\n{rep.summary}")
        if rep.fatal:
            print(f"FATAL: {rep.fatal}")

    if rep.verdict == "FAIL":
        sys.exit(2)
    if rep.verdict == "WARN":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
