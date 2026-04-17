# Chinese Text Sanity Check

Date: 2026-04-17

Method:
- Reviewed the current `docs/cn_texts` corpus against `docs/chinese_pair_sourcing_template.csv`.
- Mapped each unique current source to the slot it most plausibly fills.
- Evaluated two things:
  - whether the source is the right broad kind of text for that slot
  - whether its current captured length is at least long enough relative to the Japanese reference character count

Corpus status:
- `33` text files total
- `33` unique sources

## 1. Type / Difficulty Fit

Overall:
- `33` unique sources are a strong fit for the slot they appear to target.
- `0` are currently flagged as borderline in slot fit.

Recent replacements that resolved the earlier borderline cases:
- `01_15078370_html.txt` -> `zh_10`
  - Now uses a compact three-sample self-introduction page rather than a large template dump.
- `02_1012917.txt` -> `zh_27`
  - Now uses a longer recipe with ingredients, ordered steps, and practical process commentary.
- `23_page_htm.txt` -> `zh_09`
  - Now uses a cleaner beginner-facing conceptual writing explainer instead of a damaged lecture recap.
- `06_mzc002007knmh3g_html.txt` -> `zh_06`
  - Now uses a single-source Sogou TV synopsis page rather than a stitched Wikipedia-plus-episode-summary replacement.

New additions that close the remaining slot gaps:
- `27_hongmao_lantu_qixiazhuan.txt` -> `zh_07`
- `28_wo_yao_huahua_zhengqian.txt` -> `zh_12`
- `29_bali_island_feature.txt` -> `zh_14`
- `30_women_de_qingchun_ch1.txt` -> `zh_18`
- `31_siqinge_rile_profile.txt` -> `zh_26`
- `32_drone_flight_guide.txt` -> `zh_28`
- `33_xue_fan_profile.txt` -> `zh_33`

The prior `Dragon Raja / 龙族` placeholder is no longer borderline:
- `19_newsDetail_forward_2752725.txt` -> `zh_22`
  - Now uses a real chapter-1 prose excerpt instead of an interview/profile, so the pinned literary-reference requirement is substantively covered.

## 2. Length Adequacy

Working rule:
- Longer captures are acceptable for now.
- The main risk is sources that are too short to carry comparable translation pressure.

Thresholds used:
- `adequate for now`: at least `0.5x` the Japanese reference length
- `short caution`: `0.33x` to `<0.5x`
- `too short`: below `0.33x`

Overall:
- `33` unique sources are adequately long for now.
- `0` are a short caution.
- `0` are too short.

Notes:
- `02_1012917.txt` -> `zh_27` is no longer underlength after replacement; its current capture is comfortably above the Japanese reference length.
- The seven newly added files are all above the current `0.5x` adequacy threshold for their reference slots.
- `19_newsDetail_forward_2752725.txt` -> `zh_22` is now long rather than short (`2.22x`), which is acceptable under the current sourcing preference.
- Several other files are full-page captures and therefore much longer than the Japanese references; they remain candidates for later excerpt trimming, but they are not blocking under the current “avoid too short” rule.

## 3. Unrepresented Slots

All `33` template slots now have at least one corresponding Chinese source text in `docs/cn_texts`.

Bottom line:
- Broad source-type fit is mostly good.
- Difficulty fit is mostly good at the easy/hard level, and the previously identified borderline replacements have been resolved.
- The pinned `Dragon Raja / 龙族` slot is now covered by a real prose excerpt rather than a profile/interview substitute.
- Under the current length rule, the set is generally fine; no current file is flagged as underlength.
- Coverage is now complete at the template level, with `33` represented slots and no remaining slot gaps.
