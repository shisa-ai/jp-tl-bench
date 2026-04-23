#!/usr/bin/env bash
set -euo pipefail

cd /home/aomori/jp-tl-bench

log_ts() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

eval "$(grep '^export BYTEPLUS_TOKEN=' ~/.bashrc)"

SNAP="baseset/zh-ja-chinese-models-v1.0"
CHUNK_DIR="$SNAP/rerun_chunks_seed_pro_xml"
OUT_DIR="$SNAP/rerun_judgments_seed_pro_xml"

mkdir -p "$OUT_DIR"

for chunk in "$CHUNK_DIR"/chunk-*.jsonl; do
  name=$(basename "$chunk" .jsonl)
  out="$OUT_DIR/${name}.judgments.jsonl"
  expected=$(wc -l < "$chunk")

  if [ -f "$out" ] && [ "$(wc -l < "$out")" -eq "$expected" ]; then
    echo "===== skipping $name complete $(wc -l < "$out")/$expected $(log_ts) ====="
    continue
  fi

  rm -f "$out"
  echo "===== rejudging $name expected $expected $(log_ts) ====="
  BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python translation_comparer_any_model.py \
    --task translation_zh_ja_bidirectional_v1 \
    --judge-profile cn_judge \
    --base-url https://ark.ap-southeast.bytepluses.com/api/v3/responses \
    --judge-model seed-2-0-pro-260328 \
    --generate-base-set \
    --pairs-file "$chunk" \
    --api-key-env BYTEPLUS_TOKEN \
    --skylark-judge \
    --max-workers 8 \
    --concurrency-limit 8 \
    --rejudge
  cp "$SNAP/base_set.seed-2-0-pro-260328.jsonl" "$out"
  echo "===== saved $out lines $(wc -l < "$out") $(log_ts) ====="
done

cat "$OUT_DIR"/chunk-*.judgments.jsonl > "$SNAP/base_set.seed-2-0-pro-260328.jsonl"
echo "===== concatenated base lines $(wc -l < "$SNAP/base_set.seed-2-0-pro-260328.jsonl") $(log_ts) ====="

mamba run -n shisa-jp-tl-bench python - <<'PY'
import json
import re
from pathlib import Path

p = Path("baseset/zh-ja-chinese-models-v1.0/base_set.seed-2-0-pro-260328.jsonl")
counts = {"A": 0, "B": 0, "Tie": 0, "other": 0}
missing = []
lines = 0
for lines, line in enumerate(p.open(encoding="utf-8"), 1):
    row = json.loads(line)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", row.get("analysis", ""), re.I | re.S)
    if not m:
        missing.append((lines, row.get("id"), row.get("name")))
    else:
        ans = m.group(1).strip()
        counts[ans if ans in counts else "other"] += 1
print("lines", lines)
print("answers", counts)
print("missing", len(missing), missing[:10])
if lines != 15939 or missing or counts["other"]:
    raise SystemExit(2)
PY

mamba run -n shisa-jp-tl-bench python generate_base_scores.py \
  --judge-model seed-2-0-pro-260328 \
  --baseset-version baseset/zh-ja-chinese-models-v1.0 \
  --task translation_zh_ja_bidirectional_v1

BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python translation_comparer_any_model.py \
  --task translation_zh_ja_bidirectional_v1 \
  --judge-profile cn_judge \
  --base-url https://ark.ap-southeast.bytepluses.com/api/v3/responses \
  --judge-model seed-2-0-pro-260328 \
  --test-model shisa-ai/chotto-e4b-20260408 \
  --api-key-env BYTEPLUS_TOKEN \
  --skylark-judge \
  --max-workers 8 \
  --concurrency-limit 8 \
  --rejudge

BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python choix_analyzer.py \
  --task translation_zh_ja_bidirectional_v1 \
  --judge-profile cn_judge \
  --test-model shisa-ai/chotto-e4b-20260408 \
  --judge-model seed-2-0-pro-260328

BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python score_visualizer.py \
  --baseset-version zh-ja-chinese-models-v1.0 \
  --judge seed-2-0-pro-260328 \
  --task translation_zh_ja_bidirectional_v1 \
  --filter shisa-ai/chotto-e4b-20260408

echo "===== DONE $(log_ts) ====="
