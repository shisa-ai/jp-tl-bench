# Chinese Missing Slots Sourcing Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the seven currently uncovered Chinese sourcing slots with clean source texts and update the corpus metadata to reflect full 33-slot coverage.

**Architecture:** Source one concrete public page per missing slot, save each cleaned core text as a standalone `.txt` file under `docs/cn_texts`, append the retained URLs to `docs/cn_links.txt`, then update the audit and sanity-check documents so they reflect the new files and coverage counts.

**Tech Stack:** local filesystem, Python `requests`/`BeautifulSoup` for source inspection, Markdown/CSV docs

---

### Task 1: Add Seven Missing Source Files

**Files:**
- Create: `docs/cn_texts/27_*.txt` through `docs/cn_texts/33_*.txt`

- [ ] Add an easy franchise-summary source for `zh_07`.
- [ ] Add an easy reserve source for `zh_12`.
- [ ] Add a hard news/feature source for `zh_14`.
- [ ] Add a hard youth/coming-of-age fiction source for `zh_18`.
- [ ] Add a hard culture/music profile source for `zh_26`.
- [ ] Add a hard technical/procedural how-to source for `zh_28`.
- [ ] Add a hard reserve source for `zh_33`.

### Task 2: Update Source Index

**Files:**
- Modify: `docs/cn_links.txt`

- [ ] Append the seven retained source URLs in the same order as the new numbered files.

### Task 3: Refresh Audit Metadata

**Files:**
- Modify: `docs/cn_texts_audit.csv`

- [ ] Add seven `pass` rows describing the source family and why each new capture fits its intended slot.

### Task 4: Refresh Sanity Check

**Files:**
- Modify: `docs/cn_sanity_check.md`

- [ ] Update corpus counts from `26` to `33`.
- [ ] Update fit and length sections to reflect the seven new files.
- [ ] Replace the “unrepresented slots” section with a note that all `33` template slots are now represented.

### Task 5: Verify Corpus Consistency

**Files:**
- Verify: `docs/cn_texts/*.txt`
- Verify: `docs/cn_links.txt`
- Verify: `docs/cn_texts_audit.csv`
- Verify: `docs/cn_sanity_check.md`

- [ ] Confirm `docs/cn_texts` contains `33` numbered files with no gaps.
- [ ] Confirm `docs/cn_links.txt` contains `33` unique links.
- [ ] Confirm `docs/cn_texts_audit.csv` contains `33` rows.
- [ ] Confirm there are no duplicate text bodies.
