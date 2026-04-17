# Chinese Benchmark QA Checklist

Date: 2026-04-17

Purpose:
- Evaluate whether each file in `docs/cn_texts` is usable as a translation benchmark item, not just whether it fills the intended slot.
- Keep this separate from the earlier slot-coverage and source-inventory audits.

File-level checks:

1. `source_quality`
- `pass`: one coherent passage, clean extraction, no benchmark-breaking boilerplate, no obvious placeholder loss, no markup damage that changes the task.
- `caution`: mostly usable, but minor extraction roughness remains and may slightly distract from translation.
- `fail`: extraction damage, placeholder loss, or stray site text materially changes the task from translation into cleanup or reconstruction.

2. `slot_fit`
- `pass`: the file matches its intended broad source family and easy/hard target closely enough for the template.
- `caution`: broad family is plausible, but the register or difficulty is noticeably off.
- `fail`: wrong kind of text for the slot.

3. `length_floor`
- `pass`: long enough to create comparable translation pressure for its slot under the current “avoid too short” rule.
- `caution`: borderline short.
- `fail`: clearly too short for the intended slot.

4. `judgeability`
- `pass`: the passage gives the judge enough signal on at least one dimension from `prompts/compare_prompt.txt` such as accuracy, naturalness, tone/register, cultural handling, technical precision, structural flow, consistency, or audience fit.
- `caution`: the passage is usable, but weak, thin, or slightly noisy enough that comparisons may be less stable than ideal.
- `fail`: the passage is dominated by source defects or missing content such that the comparison would partly measure cleanup or guessing rather than translation quality.

Corpus-level checks:
- no duplicate bodies
- file numbering and source list stay aligned
- files parse cleanly as UTF-8 text

Current audit summary:
- `33` files reviewed
- `33` overall `pass`

Recent fixes:
- `04__E5_9C_A8_E7_94_B5_E8_84_91_E6_88_96-mac-_E4_B8_8A_E4_B8_8B_.txt` was rebuilt as a cleaned support excerpt from the original Microsoft article, removing the dropped-character and bullet-fragment artifacts from the earlier extraction.
- `05_cold-calling-guide.txt` was rebuilt from the Thunderbit page's Next.js source data, removing the dropped statistics, empty step content, and truncated closing CTA from the earlier extraction.
