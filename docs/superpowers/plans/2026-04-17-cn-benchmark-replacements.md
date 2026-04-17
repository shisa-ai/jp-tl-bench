# CN Benchmark Replacement Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four weak Chinese benchmark source files with cleaner, better-matched texts for their intended benchmark slots.

**Architecture:** Use the retained `docs/cn_texts` files as the write targets, but swap each one to a stronger single-source text. Keep metadata headers explicit about original versus replacement URLs when a replacement source is used, then refresh the audit and sanity-check docs to reflect the new source fit.

**Tech Stack:** plain text, Markdown, CSV, `python`, `requests`, `BeautifulSoup`, `apply_patch`

---

### Task 1: Replace Weak Source Texts

**Files:**
- Modify: `docs/cn_texts/01_15078370_html.txt`
- Modify: `docs/cn_texts/02_1012917.txt`
- Modify: `docs/cn_texts/06_mzc002007knmh3g_html.txt`
- Modify: `docs/cn_texts/23_page_htm.txt`

- [ ] Replace `01` with the smaller `自我介绍(3篇)` source from DIYFanwen.
- [ ] Replace `02` with the richer Xiachufang “家常煎牛排万能公式” procedural recipe.
- [ ] Replace `06` with the Sogou TV synopsis/cast page for the 2023 `三体` drama.
- [ ] Replace `23` with a cleaner beginner-facing story-writing advice article.

### Task 2: Refresh Benchmark Review Docs

**Files:**
- Modify: `docs/cn_texts_audit.csv`
- Modify: `docs/cn_sanity_check.md`

- [ ] Update audit rows for `01`, `02`, `06`, and `23`.
- [ ] Update the sanity-check findings so they reflect the new replacements and any remaining benchmark risks.

### Task 3: Verify End State

**Files:**
- Verify: `docs/cn_texts/*.txt`
- Verify: `docs/cn_texts_audit.csv`
- Verify: `docs/cn_sanity_check.md`

- [ ] Recompute body lengths for the replaced files.
- [ ] Recheck duplicate bodies.
- [ ] Re-scan the replaced files for obvious extraction garbage.
