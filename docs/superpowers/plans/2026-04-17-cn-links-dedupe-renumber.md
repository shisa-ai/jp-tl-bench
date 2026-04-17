# CN Links Dedupe And Renumber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate the Chinese source link list, update it so each entry matches the actual retained source text, and renumber the retained `docs/cn_texts` files into a contiguous `01`-`26` sequence with all dependent docs updated.

**Architecture:** Treat the current `26` retained text files as the source of truth. Build one canonical ordered mapping from old filename to new sequential filename plus canonical source URL, then apply that mapping consistently to `docs/cn_links.txt`, the filenames in `docs/cn_texts`, and the references in the audit/sanity docs.

**Tech Stack:** Markdown, CSV, plain text, `python`, `mv`, `rg`

---

### Task 1: Build The Canonical Mapping

**Files:**
- Modify: `docs/cn_links.txt`
- Read: `docs/cn_texts/*.txt`
- Read: `docs/cn_texts_audit.csv`
- Read: `docs/cn_sanity_check.md`

- [ ] **Step 1: Inspect the retained corpus order**

Run:

```bash
python - <<'PY'
from pathlib import Path
for p in sorted(Path('docs/cn_texts').glob('*.txt')):
    print(p.name)
PY
```

Expected: `26` retained files with numbering gaps at `03`, `20`, and `22`.

- [ ] **Step 2: Define canonical URLs from the retained text sources**

Run:

```bash
python - <<'PY'
from pathlib import Path
for p in sorted(Path('docs/cn_texts').glob('*.txt')):
    lines=p.read_text(encoding='utf-8').splitlines()
    final=next((line.split('：',1)[1] for line in lines if line.startswith('最终链接：')), '')
    repl=next((line.split('：',1)[1] for line in lines if line.startswith('替代来源：')), '')
    print(p.name, repl or final)
PY
```

Expected: one canonical URL per retained source, using replacement/final source URLs where the original link was broken or intentionally superseded.

- [ ] **Step 3: Write the canonical deduped link list**

Implementation: replace `docs/cn_links.txt` with the `26` canonical URLs in the same order as the retained corpus so the list and the files remain aligned.

### Task 2: Renumber The Text Files

**Files:**
- Modify: `docs/cn_texts/*.txt` via rename only

- [ ] **Step 1: Create the old-to-new filename mapping**

Mapping:

```text
01_15078370_html.txt -> 01_15078370_html.txt
02_1012917.txt -> 02_1012917.txt
04_tvos.txt -> 03_tvos.txt
05__E5_9C_A8_E7_94_B5_E8_84_91_E6_88_96-mac-_E4_B8_8A_E4_B8_8B_.txt -> 04__E5_9C_A8_E7_94_B5_E8_84_91_E6_88_96-mac-_E4_B8_8A_E4_B8_8B_.txt
06_cold-calling-guide.txt -> 05_cold-calling-guide.txt
07_mzc002007knmh3g_html.txt -> 06_mzc002007knmh3g_html.txt
08_c418958-32106952_html.txt -> 07_c418958-32106952_html.txt
09_c418925-32686461_html.txt -> 08_c418925-32686461_html.txt
10_c418967-40115538_html.txt -> 09_c418967-40115538_html.txt
11_c418925-40439145_html.txt -> 10_c418925-40439145_html.txt
12_uncategorized-298956_html.txt -> 11_uncategorized-298956_html.txt
13_about.txt -> 12_about.txt
14_onebook_php.txt -> 13_onebook_php.txt
15_c_html.txt -> 14_c_html.txt
16_c_html.txt -> 15_c_html.txt
17_1039955910.txt -> 16_1039955910.txt
18_1047229389.txt -> 17_1047229389.txt
19_newsDetail_forward_2488152.txt -> 18_newsDetail_forward_2488152.txt
21_newsDetail_forward_2752725.txt -> 19_newsDetail_forward_2752725.txt
23_newsDetail_forward_32975767.txt -> 20_newsDetail_forward_32975767.txt
24_BLG026-ru-he-zhuan-xie-dui-hua-bao-kuo-ge-shi-li_html.txt -> 21_BLG026-ru-he-zhuan-xie-dui-hua-bao-kuo-ge-shi-li_html.txt
25_overcoming-sales-objections-40-examples-strategies-and-rebut.txt -> 22_overcoming-sales-objections-40-examples-strategies-and-rebut.txt
26_page_htm.txt -> 23_page_htm.txt
27_79258.txt -> 24_79258.txt
28__E5_8D_B7007.txt -> 25__E5_8D_B7007.txt
29__E5_A0_B1_E4_BB_BB_E5_B0_91_E5_8D_BF_E6_9B_B8.txt -> 26__E5_A0_B1_E4_BB_BB_E5_B0_91_E5_8D_BF_E6_9B_B8.txt
```

- [ ] **Step 2: Rename files in descending order to avoid collisions**

Run:

```bash
mv docs/cn_texts/29__E5_A0_B1_E4_BB_BB_E5_B0_91_E5_8D_BF_E6_9B_B8.txt docs/cn_texts/26__E5_A0_B1_E4_BB_BB_E5_B0_91_E5_8D_BF_E6_9B_B8.txt
mv docs/cn_texts/28__E5_8D_B7007.txt docs/cn_texts/25__E5_8D_B7007.txt
mv docs/cn_texts/27_79258.txt docs/cn_texts/24_79258.txt
mv docs/cn_texts/26_page_htm.txt docs/cn_texts/23_page_htm.txt
mv docs/cn_texts/25_overcoming-sales-objections-40-examples-strategies-and-rebut.txt docs/cn_texts/22_overcoming-sales-objections-40-examples-strategies-and-rebut.txt
mv docs/cn_texts/24_BLG026-ru-he-zhuan-xie-dui-hua-bao-kuo-ge-shi-li_html.txt docs/cn_texts/21_BLG026-ru-he-zhuan-xie-dui-hua-bao-kuo-ge-shi-li_html.txt
mv docs/cn_texts/23_newsDetail_forward_32975767.txt docs/cn_texts/20_newsDetail_forward_32975767.txt
mv docs/cn_texts/21_newsDetail_forward_2752725.txt docs/cn_texts/19_newsDetail_forward_2752725.txt
mv docs/cn_texts/19_newsDetail_forward_2488152.txt docs/cn_texts/18_newsDetail_forward_2488152.txt
mv docs/cn_texts/18_1047229389.txt docs/cn_texts/17_1047229389.txt
mv docs/cn_texts/17_1039955910.txt docs/cn_texts/16_1039955910.txt
mv docs/cn_texts/16_c_html.txt docs/cn_texts/15_c_html.txt
mv docs/cn_texts/15_c_html.txt docs/cn_texts/14_c_html.txt
mv docs/cn_texts/14_onebook_php.txt docs/cn_texts/13_onebook_php.txt
mv docs/cn_texts/13_about.txt docs/cn_texts/12_about.txt
mv docs/cn_texts/12_uncategorized-298956_html.txt docs/cn_texts/11_uncategorized-298956_html.txt
mv docs/cn_texts/11_c418925-40439145_html.txt docs/cn_texts/10_c418925-40439145_html.txt
mv docs/cn_texts/10_c418967-40115538_html.txt docs/cn_texts/09_c418967-40115538_html.txt
mv docs/cn_texts/09_c418925-32686461_html.txt docs/cn_texts/08_c418925-32686461_html.txt
mv docs/cn_texts/08_c418958-32106952_html.txt docs/cn_texts/07_c418958-32106952_html.txt
mv docs/cn_texts/07_mzc002007knmh3g_html.txt docs/cn_texts/06_mzc002007knmh3g_html.txt
mv docs/cn_texts/06_cold-calling-guide.txt docs/cn_texts/05_cold-calling-guide.txt
mv docs/cn_texts/05__E5_9C_A8_E7_94_B5_E8_84_91_E6_88_96-mac-_E4_B8_8A_E4_B8_8B_.txt docs/cn_texts/04__E5_9C_A8_E7_94_B5_E8_84_91_E6_88_96-mac-_E4_B8_8A_E4_B8_8B_.txt
mv docs/cn_texts/04_tvos.txt docs/cn_texts/03_tvos.txt
```

Expected: `docs/cn_texts` contains `01` through `26` with no numbering gaps.

### Task 3: Update Dependent Docs

**Files:**
- Modify: `docs/cn_texts_audit.csv`
- Modify: `docs/cn_sanity_check.md`

- [ ] **Step 1: Rewrite file references to the new numbering**

Implementation: update every old filename reference in the audit CSV and sanity-check report using the mapping from Task 2.

- [ ] **Step 2: Update corpus-count statements**

Implementation: keep `26` files / `26` audit rows / no duplicate-body groups wording consistent with the deduped corpus.

### Task 4: Verify End State

**Files:**
- Verify: `docs/cn_links.txt`
- Verify: `docs/cn_texts/*.txt`
- Verify: `docs/cn_texts_audit.csv`
- Verify: `docs/cn_sanity_check.md`

- [ ] **Step 1: Verify file count and numbering**

Run:

```bash
python - <<'PY'
from pathlib import Path
files=sorted(p.name for p in Path('docs/cn_texts').glob('*.txt'))
print('count', len(files))
print(files)
PY
```

Expected: `26` files numbered `01` through `26` with no gaps.

- [ ] **Step 2: Verify audit coverage**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
files={p.name for p in Path('docs/cn_texts').glob('*.txt')}
rows=list(csv.DictReader(Path('docs/cn_texts_audit.csv').open(encoding='utf-8', newline='')))
print('audit_rows', len(rows))
print('missing_from_disk', [r['file'] for r in rows if r['file'] not in files])
print('extra_on_disk', sorted(files - {r['file'] for r in rows}))
PY
```

Expected: `26` rows, no missing files, no extra files.

- [ ] **Step 3: Verify deduped link list length**

Run:

```bash
python - <<'PY'
from pathlib import Path
links=[line.strip() for line in Path('docs/cn_links.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
print('links', len(links))
print('unique_links', len(set(links)))
PY
```

Expected: `26` links and `26` unique links.
