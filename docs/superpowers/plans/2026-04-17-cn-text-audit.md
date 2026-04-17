# Chinese Source Text Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit every file in `docs/cn_texts` and confirm it matches its intended source and text type, repairing any broken or clearly mismatched captures.

**Architecture:** Treat this as a three-phase audit: build a metadata inventory, review every capture against explicit checks, then fix and re-verify anything that fails. Use the link list and sourcing template as the reference points so the audit is consistent across all files.

**Tech Stack:** Bash, Python standard library, local text files in `docs/`

---

### Task 1: Build Audit Inventory

**Files:**
- Read: `docs/cn_links.txt`
- Read: `docs/cn_texts/*.txt`
- Read: `docs/chinese_pair_sourcing_template.csv`

- [ ] **Step 1: Enumerate every captured text file**

Run: `python - <<'PY'`
```python
from pathlib import Path
for path in sorted(Path("docs/cn_texts").glob("*.txt")):
    print(path.name)
```
`PY`

- [ ] **Step 2: Extract audit metadata**

Run: `python - <<'PY'`
```python
from pathlib import Path
import re

for path in sorted(Path("docs/cn_texts").glob("*.txt")):
    text = path.read_text(encoding="utf-8")
    title = re.search(r"^标题：(.*)$", text, re.M)
    method = re.search(r"^提取方式：(.*)$", text, re.M)
    print(path.name, "|", title.group(1).strip() if title else "MISSING", "|", method.group(1).strip() if method else "MISSING")
```
`PY`

- [ ] **Step 3: Flag suspicious captures for deeper review**

Check for:
- missing title or extraction method
- HTTP errors or anti-bot text
- very short bodies
- obviously duplicated or placeholder content

### Task 2: Review All Files

**Files:**
- Read: `docs/cn_texts/*.txt`
- Reference: `docs/cn_links.txt`
- Reference: `docs/chinese_pair_sourcing_template.csv`

- [ ] **Step 1: Review each file against the source URL and apparent text type**

Confirm:
- the recorded source URL matches the intended link
- the extracted text is real content, not boilerplate or an error shell
- the content family matches what the source appears to be

- [ ] **Step 2: Record failures**

Mark any file that is:
- broken
- too thin to be useful
- replaced with the wrong kind of text
- inconsistent with the category implied by the link

### Task 3: Repair and Verify

**Files:**
- Modify: `docs/cn_texts/*.txt` as needed

- [ ] **Step 1: Replace or repair bad captures**

For each failed file:
- fetch a better text-first source if the original link is unusable
- preserve the original link in the file metadata
- keep the extracted core text concise but substantive

- [ ] **Step 2: Re-run inventory checks**

Run: `python - <<'PY'`
```python
from pathlib import Path
print(len(list(Path("docs/cn_texts").glob("*.txt"))))
```
`PY`

- [ ] **Step 3: Summarize final audit status with evidence**

Report:
- total files reviewed
- files fixed
- any remaining weak files or judgment calls
