# Research Note: Potential Scoring Improvements (Bradley–Terry + LT anchoring)

**Status / scope**
- This note documents **the scoring behavior we shipped and used for Base Set v1.0**.
- We are **not changing v1.0 scoring** (we have months of historical runs).
- The goal is to make the current mechanism unambiguous and to enumerate **future, versioned** alternatives (v1.1+/v2.0) with clear trade-offs.

---

## 1) Current behavior (Base Set v1.0) in plain English

When we “score a candidate model” in JP‑TL‑Bench today:

1. We already have a **frozen base set**: 20 anchor models with a large set of **anchor–anchor** A/B judgments (a near round‑robin over the benchmark items) produced by a particular judge under a particular prompt.
2. For a new candidate model, we collect **candidate–anchor** A/B judgments (candidate vs each of the 20 anchors on each benchmark item).
3. To produce the candidate score, we run a **single Bradley–Terry fit** on the union of:
   - the frozen **anchor–anchor** judgments from the Base Set snapshot, and
   - the newly collected **candidate–anchor** judgments.
4. We then report the candidate’s win rate and its derived **LT score** (0–10) for the relevant slices.

Important clarifications:

- We are **not** “freezing anchor parameters” in the solver. Anchors are re‑estimated in each fit, but they are strongly constrained by the much larger anchor–anchor dataset.
- We also **do not** build one global pool of all candidates. Each candidate is scored in its own fit (base + that candidate’s edges), so other candidates cannot change an already‑scored candidate.

This corresponds to the “Option B1” interpretation:
> Reuse the frozen anchor–anchor results, add candidate–anchor results, and fit BT on the combined graph.

---

## 2) Exact pipeline + file inputs (what the code actually consumes)

At a high level, the v1.0 pipeline is:

1. **Pair generation** (candidate vs anchors)
   - `generate_shootout_data.py` reads:
     - candidate translations from `translations/<safe_model>.jsonl`
     - anchor translations from `$BASESET_SNAPSHOT_DIR/translations/*.jsonl`
   - It writes per-candidate pairs to:
     - `results/<baseset_version>/<safe_model>/<safe_judge>/pairs.jsonl`
   - Code: `repos/jp-tl-bench/generate_shootout_data.py`

2. **Judging** (LLM-as-a-judge picks A/B)
   - `translation_comparer_any_model.py` reads `pairs.jsonl` and writes:
     - `results/<baseset_version>/<safe_model>/<safe_judge>/judgments.jsonl`
   - It is idempotent by default: it reuses existing `judgments.jsonl` unless `--rejudge` is set.
   - Code: `repos/jp-tl-bench/translation_comparer_any_model.py`

3. **Scoring** (Bradley–Terry + LT)
   - `choix_analyzer.py` reads:
     - base set judgments: `$BASESET_SNAPSHOT_DIR/base_set.<safe_judge>.jsonl`
     - candidate judgments: `results/<baseset_version>/<safe_model>/<safe_judge>/judgments.jsonl`
   - It concatenates those comparisons and fits Bradley–Terry.
   - It outputs a per-candidate summary:
     - `results/<baseset_version>/<safe_model>/<safe_judge>/scores.json`
   - Code: `repos/jp-tl-bench/choix_analyzer.py`

Naming conventions:
- “Safe names” replace `/` with `__` in file paths and in `llm_a`/`llm_b` fields (e.g., `meta-llama__Llama-3.3-70B-Instruct`).
- `baseset_version` in results paths is derived from the basename of `$BASESET_SNAPSHOT_DIR` (e.g., `baseset/v1.0` → `v1.0`).

---

## 3) Mathematical model (what we are fitting)

### 3.1 Bradley–Terry basics

Each model \(i\) has a latent log‑strength \(\theta_i \in \mathbb{R}\). The probability that model \(i\) wins against model \(j\) is:

\[
P(i \succ j) \;=\; \frac{\exp(\theta_i)}{\exp(\theta_i) + \exp(\theta_j)}
\;=\; \sigma(\theta_i - \theta_j)
\]

where \(\sigma(x) = \frac{1}{1 + e^{-x}}\).

Given observed A/B outcomes, Bradley–Terry maximum likelihood finds \(\theta\) that maximizes the log-likelihood.

### 3.2 What data goes into the fit (v1.0)

Let:
- \(A = \{1,\dots,20\}\) be the set of anchors,
- \(c\) be the candidate model.

We fit on the union of:

1) Frozen anchor–anchor comparisons from the Base Set snapshot:
- For many pairs \((i,j)\in A\times A\), and many items, we observe outcomes of \(i\) vs \(j\).

2) Candidate–anchor comparisons:
- For each anchor \(i \in A\) and many items, we observe outcomes of \(c\) vs \(i\).

So the fit graph is “dense among anchors” plus a “star” from candidate to all anchors.

### 3.3 Separate fits per slice

In the tooling, we do not fit one multi-task model. We fit separate BT models on subsets (“slices”) of the data:
- difficulty: `easy`, `hard`, or `all`
- direction: `english` (EN→JA) vs `japanese` (JA→EN) vs overall

In code, each slice is a separate call to `choix.opt_pairwise(...)` on the corresponding list of winner/loser pairs.

---

## 4) What “LT (0–10)” means today (and the candidate-in-the-mean effect)

### 4.1 Definitions in the current implementation

The tool computes:

- Raw BT strength: \(\theta_i\) (what `choix` returns)
- LT score: a logistic squashing of a **centered** \(\theta\)

In code (`LLMRanker.get_rankings()`), LT is computed as:

\[
\mathrm{LT}_i \;=\; 10 \cdot \sigma(\theta_i - \overline{\theta})
\]

with

\[
\overline{\theta} \;=\; \frac{1}{N}\sum_{k=1}^{N}\theta_k
\]

where \(N\) is the number of models in that slice’s BT fit.

For a candidate scoring run, \(N = 21\) (20 anchors + the candidate), so:
- \(\overline{\theta}\) **includes the candidate**.

Code location:
- `repos/jp-tl-bench/choix_analyzer.py` constructs the comparisons list as `base_set + candidate` and fits once per slice.
- `repos/jp-tl-bench/choix_analyzer.py` extracts the candidate’s LT from the fitted rankings.
- `repos/jp-tl-bench/choix_analyzer.py` → `LLMRanker.get_rankings()` computes `mean_param = np.mean(params)` and centers by that mean.

### 4.2 Implication: the “zero point” shifts slightly per candidate

Because we center on the mean over **all 21 models**, a very strong candidate pulls the mean up a little, and a very weak candidate pulls it down a little. This does not change any BT win probabilities (which only depend on differences \(\theta_i - \theta_j\)), but it does change the displayed LT mapping.

If we assume (for intuition) that the anchors do not move between fits and define:
- \(\mu_A = \frac{1}{20}\sum_{i\in A}\theta_i\) (anchor mean),
- \(\mu_{A\cup\{c\}} = \frac{20\mu_A + \theta_c}{21}\) (mean including candidate),

then:

\[
\theta_c - \mu_{A\cup\{c\}} \;=\; \frac{20}{21}(\theta_c - \mu_A)
\]

So relative to an “anchor-mean centered” scale, the candidate’s centered log‑strength is deterministically shrunk by a factor of \(20/21 \approx 0.952\) before applying the logistic.

**Key nuance**: this “\(20/21\)” claim is exact for the *logit axis* (\(\theta\) differences), not for the *final LT value* itself. LT applies a sigmoid, so the effect on LT is nonlinear—but still monotonic.

On the 0–10 LT scale, this tends to be small but not strictly zero. For example (holding anchors fixed):

| Candidate advantage vs anchor mean (\(\theta_c - \mu_A\)) | LT if centered on anchors | LT if centered on all 21 | Difference |
|---:|---:|---:|---:|
| 1.0 | 7.31 | 7.22 | −0.10 |
| 2.0 | 8.81 | 8.70 | −0.10 |
| −2.0 | 1.19 | 1.30 | +0.10 |

Numerically, for \(n=20\) anchors the maximum absolute LT change between these two centerings is about **0.11** (on a 0–10 scale), occurring near the middle of the sigmoid (not in the saturated extremes).

### 4.3 A second, smaller effect: anchors are re-fit per candidate

Even though anchor–anchor outcomes are frozen, the joint MLE includes candidate edges too, so anchor \(\theta\) values can move slightly when adding a candidate.

Intuition:
- anchor–anchor comparisons are roughly an order of magnitude more numerous than candidate–anchor comparisons in a typical run
- so the candidate’s edges have limited leverage over the anchor configuration

But mathematically, the anchor parameters are not “locked” today; they are estimated jointly with the candidate.

---

## 5) Why the current approach is reasonable (and why we kept it for v1.0)

From a modelling perspective, the current approach is the **joint maximum-likelihood estimate** for the graph we are fitting (anchors + candidate), assuming:
- all outcomes are generated by the same Bradley–Terry process under the same judge/prompt conditions, and
- we want the best-fitting parameters for the combined dataset.

From an engineering/product perspective, it also has practical advantages:

- The “reference” is locked by **data reuse** (frozen base judgments), not by special-casing parameters.
- It naturally supports missing candidate pairs (some candidate–anchor judgments can be absent and the fit still works).
- It allows the tooling to compute consistent rankings internally for inspection/debugging.

One more practical observation (often phrased as “self‑punishment” / “self‑reward”): because the LT centering uses the mean over all \(n+1\) models in the fit, the candidate’s \(\theta_c - \overline{\theta}\) is a deterministic \(n/(n+1)\) shrink of its distance from the anchor mean (under the “anchors fixed” intuition). This does **not** introduce randomness; it just chooses a particular “zero point” (gauge) for the slice.

---

## 6) What might be “more correct” or “more useful” in future versions?

There isn’t one universally correct choice; it depends on what we want the number to mean.

Below are the main alternatives, with pros/cons in the context of an **anchored, versioned** benchmark.

### Option F1: Keep the joint fit, but center LT on anchors-only

Change only the LT post-processing:

\[
\mathrm{LT}_i = 10\cdot\sigma(\theta_i - \mu_A)
\quad\text{where}\quad
\mu_A = \frac{1}{20}\sum_{i\in A}\theta_i
\]

Pros:
- Removes the “candidate-in-the-mean” shift while leaving everything else unchanged.
- Minimal conceptual change; still uses a full BT fit with the base graph + candidate edges.

Cons:
- Anchors are still re-fit slightly per candidate, so \(\mu_A\) can drift a little across candidates.
- Requires the scorer to know which items are anchors (from the snapshot manifest or base_set file).

### Option F2 (B2): Freeze anchor strengths; solve only the candidate parameter

Compute anchor \(\theta_i\) once from the Base Set snapshot, freeze them, and for each candidate solve:

\[
\max_{\theta_c}\;\sum_{i\in A}
w_{c,i}\log\sigma(\theta_c-\theta_i)
 w_{i,c}\log\sigma(\theta_i-\theta_c)
\]

Pros:
- The anchor coordinate system is *truly fixed* for the Base Set version; no anchor drift.
- Candidate scoring becomes a fast 1D convex optimization per slice.
- Candidate LT becomes strictly comparable across runs in the same Base Set/judge/prompt (subject to judgment noise).

Cons:
- It is no longer the joint MLE on the combined graph; it is a “plug-in” estimate conditioned on frozen anchors.
- If the judge distribution shifts (e.g., the judge model changes silently), freezing anchors won’t “adapt”; but arguably we want such shifts to invalidate comparability anyway.

### Option F3: Penalized/Bayesian anchoring (anchors as priors)

Treat anchor strengths as having a prior around their Base Set estimates, and fit a regularized objective:

\[
\max_{\theta}\;\log L(\theta) \;-\; \lambda\sum_{i\in A}(\theta_i - \theta_i^{(0)})^2
\]

Pros:
- Interpolates between fully joint fit and fully frozen anchors.
- Can explicitly control how much candidate edges are allowed to move anchors.

Cons:
- Introduces an extra hyperparameter \(\lambda\) and more complexity.
- Harder to communicate/reproduce cleanly than “frozen anchors”.

### Option F4: Calibrated win-rate mapping (no BT re-fit per candidate)

Skip BT for candidates and map candidate win rates vs anchors to an LT-like scale via interpolation (e.g., isotonic regression or a parametric curve).

Pros:
- Extremely simple and robust.
- Zero risk of anchor drift.

Cons:
- Throws away transitive information and “strength-of-opponent” modelling that BT provides.
- Less principled when anchors are unevenly spaced or when some pairs are missing.

---

## 7) Implementation notes (exact code behavior to revisit later)

### 7.1 Where “joint fit” happens

In `repos/jp-tl-bench/choix_analyzer.py`, candidate scoring does:

1. Load base set comparisons:
- `$BASESET_SNAPSHOT_DIR/base_set.<safe_judge>.jsonl`
2. Load candidate comparisons:
- `results/<baseset_version>/<safe_model>/<safe_judge>/judgments.jsonl`
3. Concatenate and fit:
- `ranker.fit(comparisons)`

This is the core “B1 joint fit” choice.

### 7.2 Where LT centering is defined

In `repos/jp-tl-bench/choix_analyzer.py` → `LLMRanker.get_rankings()`:

```python
mean_param = np.mean(params)
shifted_params = params - mean_param
lt_scores = 1.0 / (1.0 + np.exp(-shifted_params))
lt_scores_0_10 = lt_scores * 10
```

Because `params` is the full parameter vector for that slice’s fit, the mean includes the candidate whenever the candidate is part of the fit.

### 7.3 “Structural stability” (what is stable and what is not)

Stable today (given Base Set snapshot + judge + prompt + decoding + scorer version):
- A candidate’s score does not depend on what other candidates have been run, because each candidate fit includes only base edges + that candidate’s edges.

Not strictly invariant today:
- The LT “zero point” varies slightly across candidates because we center on the mean over all models in that candidate’s fit.

---

## 8) If we revisit this later: recommended experiments

If/when we consider changing scoring in a new Base Set version, we can quantify impact by:

1. Recomputing candidate scores two ways on the same stored judgments:
   - current (joint fit + mean over all models)
   - anchor-only centering (Option F1) and/or frozen-anchor solve (Option F2)
2. Measuring deltas in:
   - candidate LT per slice
   - rank ordering stability
   - sensitivity vs missing judgments (`missing_pairs`)
3. Deciding whether the extra strictness is worth a version bump (likely **minor** if we keep LT semantics, **major** if we redefine the scale).

---

## Appendix A: Empirical delta check on stored data (one example)

In this **paper workspace checkout** (`papers/jp-tl-bench`), we have example candidate judgments at:
- `results/gemini-3-pro-preview/gemini-2.5-flash/judgments.jsonl`

Using `baseset/v1.0/base_set.gemini-2.5-flash.jsonl` and fitting with the current code, we can compare:
- **Current**: joint fit; LT centered on mean over all models in the fit (anchors + candidate)
- **Alt (F1)**: joint fit; LT centered on the mean over anchors only
- **Alt (F2)**: frozen-anchor solve; anchors from base fit are fixed and only \(\theta_c\) is optimized

For `gemini-3-pro-preview` (a very strong candidate, so LT is near saturation), the differences are small:

| Slice | Current LT (mean over all) | LT w/ anchor-only centering (F1) | LT w/ frozen anchors (F2) |
|---|---:|---:|---:|
| EN→JA overall | 9.9889 | 9.9920 | 9.9925 |
| EN→JA easy | 9.9930 | 9.9951 | 9.9955 |
| EN→JA hard | 9.9917 | 9.9941 | 9.9949 |
| JA→EN overall | 9.9392 | 9.9522 | 9.9536 |
| JA→EN easy | 9.8784 | 9.9011 | 9.9072 |
| JA→EN hard | 9.9753 | 9.9815 | 9.9817 |

We also measured anchor movement from adding the candidate edges (base-only fit vs joint fit, after mean-centering anchors within each fit):
- max absolute anchor drift in \(\theta\) was about **0.05–0.23** across slices in this example (RMS ~0.02–0.06).

Interpretation:
- For very strong/weak candidates, the LT sigmoid saturates, so the centering choice matters less in absolute LT points.
- Mid-range candidates (where \(\theta_c\) lands near the sigmoid’s steep region) are where the centering choice can matter most (up to ~0.11 LT points for \(n=20\) in the “anchors fixed” intuition).
