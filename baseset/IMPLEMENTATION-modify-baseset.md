# Modifying the base set

Working notes for small, surgical tweaks to the v1.0 snapshot without blowing away expensive judge outputs.

## Backfill missing judgments (alternate judge)

Goal: fill the ~30 missing/invalid comparisons using another judge (e.g., GPT-5) while keeping the 12.7k Gemini rows intact.

1) Collect the missing IDs
```bash
python - <<'PY'
import json, re
from pathlib import Path
ANS = re.compile(r"<answer>(.*?)</answer>", re.I)
pair_ids = {json.loads(l)["id"] for l in Path("baseset/v1.0/base_conversation_pairs.jsonl").open()}
judged, missing = set(), set()
for l in Path("baseset/v1.0/base_set.gemini-2.5-flash.jsonl").open():
    d = json.loads(l)
    m = ANS.search(d.get("analysis","") or "")
    if not m:
        missing.add(d["id"]); continue
    ans = "".join(c for c in m.group(1) if c.isalpha()).lower()
    if ans not in {"a","b"}:
        missing.add(d["id"]); continue
    judged.add(d["id"])
missing |= (pair_ids - judged)          # rows absent from the judged file
Path("missing_ids.txt").write_text("\n".join(sorted(missing)))
print(f"missing_ids written: {len(missing)}")
PY
```

2) Judge only those IDs with the alternate judge (no rejudging)
```bash
# Ensure the pair file is at repo root for the comparer
cp baseset/v1.0/base_conversation_pairs.jsonl base_conversation_pairs.jsonl

python translation_comparer_any_model.py \
  --generate-base-set \
  --judge-model gpt-5 \
  --skip-ids-file missing_ids.txt \
  --max-workers 10 --concurrency-limit 10
# Output: base_sets/base_set.gpt-5.jsonl containing just the backfilled rows
```

3) Merge backfilled rows into a mixed base set (keep the pure Gemini file unchanged)
```bash
python - <<'PY'
import json
from pathlib import Path
base = Path("baseset/v1.0/base_set.gemini-2.5-flash.jsonl")
backfill = Path("base_sets/base_set.gpt-5.jsonl")
out = Path("baseset/v1.0/base_set.gemini-2.5-flash+gpt-5.jsonl")
records, order = {}, []
for line in base.open():
    d = json.loads(line); i = d["id"]
    records[i] = d; order.append(i)
for line in backfill.open():
    d = json.loads(line); i = d["id"]
    if i not in records:
        order.append(i)
    records[i] = d  # prefer backfill when overlapping
with out.open("w", encoding="utf-8") as f:
    for i in order:
        f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
print(f"merged -> {out} ({len(order)} rows)")
PY
```

4) Re-score using the mixed file (no new judging)
```bash
python baseset/generate_set.py \
  --analysis-file baseset/v1.0/base_set.gemini-2.5-flash+gpt-5.jsonl \
  --no-auto-judge
```
Notes:
- Judged rows keep their `judge_model` field, so provenance is preserved.
- `translation_comparer_any_model.py` now dedupes by ID when merging; reruns with the same `--skip-ids-file` are safe.

## Swap a model in the base set
If we're swapping something into the same baseset/v1.0:
- copy translation file
- replace item in manifest.json
- run `generate_set.py --no-auto-judge` to see what's up
- run `generate_set.py --auto-judge --gemini-judge` and it should fill in just the new judgements




Goal: replace one model’s translations with another, while reusing existing judgments for unchanged pairs and only judging the new pairs.

Outline (assume original snapshot is baseset/v1.0, judge model gemini-2.5-flash):

1) Create a new snapshot dir (e.g., `baseset/v1.1/`) and update the manifest
- Copy `baseset/v1.0/manifest.json` to `baseset/v1.1/manifest.json`.
- Remove the old model entry, add the new model entry (`model` and `source` path).
- Place the new translation dump in `baseset/v1.1/translations/<safe_name>.jsonl` (safe name = `/` → `__`). Remove the old one if you want a clean dir.

2) Rebuild pairs for the new manifest (no judging yet)
```bash
python baseset/generate_set.py \
  --snapshot-dir baseset/v1.1 \
  --pair-filename base_conversation_pairs.v1.1.jsonl \
  --no-auto-judge
```
Copy the new pair file to repo root for the comparer:
```bash
cp baseset/v1.1/base_conversation_pairs.v1.1.jsonl base_conversation_pairs.jsonl
```

3) Seed a base_set with all reusable judgments from the old snapshot
```bash
python - <<'PY'
import json
from pathlib import Path
new_manifest = json.load(open("baseset/v1.1/manifest.json"))["models"]
keep = {m["model"].replace("/", "__") for m in new_manifest}
old_base = Path("baseset/v1.0/base_set.gemini-2.5-flash.jsonl")
seed = Path("baseset/v1.1/base_set.gemini-2.5-flash.seed.jsonl")
count = 0
with old_base.open() as f_in, seed.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        d = json.loads(line)
        if d["llm_a"] in keep and d["llm_b"] in keep:
            f_out.write(line)
            count += 1
print(f"seeded {count} judgments into {seed}")
PY
```
This keeps all pairs between unchanged models (IDs match because file names are unchanged). Pairs involving the removed model are dropped; pairs involving the new model don’t exist yet.

4) Collect judged IDs from the seed to skip in the new judge run
```bash
python - <<'PY'
import json, re
from pathlib import Path
ANS = re.compile(r"<answer>(.*?)</answer>", re.I)
seed = Path("baseset/v1.1/base_set.gemini-2.5-flash.seed.jsonl")
ids = set()
for l in seed.open():
    d = json.loads(l); m = ANS.search(d.get("analysis","") or "")
    if not m: continue
    ans = "".join(c for c in m.group(1) if c.isalpha()).lower()
    if ans in {"a","b"} and d.get("id"):
        ids.add(d["id"])
Path("skip_ids_seed.txt").write_text("\n".join(sorted(ids)))
print(f"seed ids written: {len(ids)}")
PY
```

5) Judge only the new pairs (those not in the seed)
```bash
python translation_comparer_any_model.py \
  --generate-base-set \
  --judge-model gemini-2.5-flash \
  --skip-ids-file skip_ids_seed.txt \
  --max-workers 40 --concurrency-limit 40
# Outputs base_sets/base_set.gemini-2.5-flash.jsonl containing new + any retried rows
```

6) Merge seed + new judgments into the v1.1 snapshot file
```bash
python - <<'PY'
import json
from pathlib import Path
seed = Path("baseset/v1.1/base_set.gemini-2.5-flash.seed.jsonl")
fresh = Path("base_sets/base_set.gemini-2.5-flash.jsonl")
out = Path("baseset/v1.1/base_set.gemini-2.5-flash.jsonl")
records, order = {}, []
for src in (seed, fresh):
    for line in src.open():
        d = json.loads(line); i = d.get("id")
        if not i: continue
        if i not in records:
            order.append(i)
        records[i] = d  # prefer fresher where overlapping
with out.open("w", encoding="utf-8") as f:
    for i in order:
        f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
print(f"merged -> {out} ({len(order)} rows)")
PY
```

7) Re-score (no further judging)
```bash
python baseset/generate_set.py \
  --snapshot-dir baseset/v1.1 \
  --pair-filename base_conversation_pairs.v1.1.jsonl \
  --analysis-file baseset/v1.1/base_set.gemini-2.5-flash.jsonl \
  --no-auto-judge
```

Notes:
- Keep the original v1.0 files unchanged for provenance. Work in a new snapshot directory when swapping models.
- The seed step reuses all unchanged judgments; only pairs touching the new model (and any IDs not present in the seed) are judged.
- The comparer’s merge logic dedupes by ID, so re-runs with the same skip file are safe. adaptive concurrency to control spend.

## Extraneous judgments (from swapped/removed models)

If a base_set file still contains judgments for models that are no longer in the snapshot manifest, it doesn’t affect scoring:
- `generate_set.py` builds pair IDs from the manifest and filters comparisons to manifest models before fitting, so out-of-manifest rows are ignored.
- Auto-judge runs also rely on the current pair file; skip-ID lists may be larger, but unmatched IDs are harmless.

Downside is just bloat/noise. If you want to prune to manifest-only rows:
```bash
python - <<'PY'
import json
from pathlib import Path
pairs = {json.loads(l)["id"] for l in Path("baseset/v1.0/base_conversation_pairs.jsonl").open()}
src = Path("baseset/v1.0/base_set.gemini-2.5-flash.jsonl")
out = Path("baseset/v1.0/base_set.gemini-2.5-flash.pruned.jsonl")
keep = 0
with src.open() as f_in, out.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        d = json.loads(line)
        if d.get("id") in pairs:
            f_out.write(line); keep += 1
print(f"kept {keep} rows -> {out}")
PY
```
Then point `--analysis-file` at the pruned file if you want a clean artifact; otherwise, leaving extras is fine. Keeps scores identical either way.
