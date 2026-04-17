import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "v1.0"
PAIR_FILE = SNAPSHOT / "base_conversation_pairs.jsonl"
JUDGED_FILE = SNAPSHOT / "base_set.gemini-2.5-flash.jsonl"
MISSING_PAIRS = ROOT / "base_conversation_pairs.missing.jsonl"
TRANSLATION_COMPARER = ROOT.parent / "translation_comparer_any_model.py"

JUDGE_MODEL = os.getenv("BACKFILL_JUDGE_MODEL", "gpt-5")
BASE_URL = os.getenv("BACKFILL_BASE_URL", "https://api.openai.com/v1")
MAX_WORKERS = int(os.getenv("BACKFILL_MAX_WORKERS", "30"))
CONCURRENCY_LIMIT = int(os.getenv("BACKFILL_CONCURRENCY_LIMIT", "30"))
SAFE_JUDGE_MODEL = JUDGE_MODEL.replace("/", "__")
BACKFILL_FILE = ROOT.parent / "base_sets" / f"base_set.{SAFE_JUDGE_MODEL}.jsonl"
MERGED_OUT = SNAPSHOT / f"base_set.gemini-2.5-flash+{SAFE_JUDGE_MODEL}.jsonl"
DEPRECATION_MESSAGE = (
    "deprecated: baseset/backfill-1.0-judgements.py still targets pre-refactor paths. "
    "Use `mamba run -n shisa-jp-tl-bench python baseset/prepare_v1_0.py --auto-judge` "
    "or `mamba run -n shisa-jp-tl-bench python baseset/generate_set.py --snapshot-dir <dir> --auto-judge` instead."
)


def run_translation_comparer(missing_count: int) -> None:
    print("\n=== Stage 3: run comparer on missing pairs ===")
    if missing_count == 0:
        print("  nothing to backfill; skipping comparer run.")
        return
    if BACKFILL_FILE.exists():
        print(f"  backfill file already present: {BACKFILL_FILE} (skipping comparer run)")
        return
    if not TRANSLATION_COMPARER.exists():
        raise SystemExit(f"Missing comparer script: {TRANSLATION_COMPARER}")

    cmd = [
        sys.executable,
        str(TRANSLATION_COMPARER),
        "--pairs-file",
        str(MISSING_PAIRS),
        "--generate-base-set",
        "--judge-model",
        JUDGE_MODEL,
        "--base-url",
        BASE_URL,
        "--max-workers",
        str(MAX_WORKERS),
        "--concurrency-limit",
        str(CONCURRENCY_LIMIT),
    ]
    print("  running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT.parent, check=True)


def main() -> None:
    raise SystemExit(DEPRECATION_MESSAGE)
    print("=== Stage 1: scan for missing/invalid answers ===")
    if not PAIR_FILE.exists():
        raise SystemExit(f"Missing pair file: {PAIR_FILE}")
    if not JUDGED_FILE.exists():
        raise SystemExit(f"Missing judged file: {JUDGED_FILE}")
    ans_re = re.compile(r"<answer>(.*?)</answer>", re.I)
    pair_ids = {json.loads(l)["id"] for l in PAIR_FILE.open()}
    judged, missing = set(), set()

    for line in JUDGED_FILE.open():
        data = json.loads(line)
        match = ans_re.search(data.get("analysis", "") or "")
        if not match:
            missing.add(data["id"])
            continue
        answer = "".join(c for c in match.group(1) if c.isalpha()).lower()
        if answer not in {"a", "b"}:
            missing.add(data["id"])
            continue
        judged.add(data["id"])

    missing |= (pair_ids - judged)  # rows absent from judged file
    missing_count = len(missing)
    print(f"  total pairs: {len(pair_ids):,}")
    print(f"  judged (valid A/B): {len(judged):,} (extraneous rows present: {max(len(judged)-len(pair_ids),0):,})")
    print(f"  missing/invalid: {missing_count:,}")

    print("\n=== Stage 2: write missing-only pair file ===")
    keep = 0
    with PAIR_FILE.open() as f_in, MISSING_PAIRS.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            if data["id"] in missing:
                f_out.write(line)
                keep += 1
    print(f"  wrote {keep} rows -> {MISSING_PAIRS}")
    run_translation_comparer(missing_count)

    print("\n=== Stage 4: merge backfill into mixed file ===")
    if not BACKFILL_FILE.exists():
        print(f"  backfill file not found yet: {BACKFILL_FILE}")
        if missing_count:
            print("  comparer was not able to produce a backfill file; check the output above.")
        else:
            print("  nothing to merge; no missing judgments and no backfill file present.")
        return
    records, order = {}, []
    for line in JUDGED_FILE.open():
        data = json.loads(line)
        i = data.get("id")
        if not i:
            continue
        records[i] = data
        order.append(i)
    for line in BACKFILL_FILE.open():
        data = json.loads(line)
        i = data.get("id")
        if not i:
            continue
        if i not in records:
            order.append(i)
        records[i] = data  # prefer backfill when overlapping
    with MERGED_OUT.open("w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    print(f"  merged -> {MERGED_OUT} ({len(order):,} rows)")
    print("\n=== Stage 5: re-score (optional) ===")
    print(
        "  python generate_set.py "
        f"--analysis-file v1.0/{MERGED_OUT.name} "
        "--no-auto-judge"
    )


if __name__ == "__main__":
    main()
