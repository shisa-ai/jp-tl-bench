# Think-Tag Prompt And Parser Design

## Goal

Adopt `<think>...</think>` as the visible reasoning convention for Chinese translation prompts while preserving the existing Japanese prompt contract for now, and harden the translation parser so only `<translation>...</translation>` is treated as canonical output.

## Scope

In scope:

- Update Chinese translation prompts to request reasoning in `<think>` tags.
- Leave Japanese translation prompts unchanged, including their current `<translation_analysis>` guidance.
- Update the translation parser so it ignores both `<think>` and `<translation_analysis>` when extracting the final translation.
- Preserve a fallback path when `<translation>` is missing, but make that fallback less likely to store reasoning text as the translation.
- Add parser-focused tests for the mixed old/new prompt contracts.

Out of scope:

- Changing Japanese prompt wording.
- Changing judge prompts.
- Removing `full_response` from stored translation artifacts.
- Reworking model-specific generation settings beyond what is needed for parser safety.

## Background

The current translation pipeline stores the full model response and extracts the final translation from `<translation>` tags. Japanese prompts explicitly ask models to place their reasoning inside `<translation_analysis>`. For modern reasoning-heavy models, especially Chinese-oriented ones, `<think>` is a more natural visible reasoning convention.

The current parser is too permissive when `<translation>` is missing. It can fall back to large portions of the raw response, which risks storing reasoning text instead of only the final translation. This is acceptable as a safety net, but it should prefer non-reasoning remainder and explicitly ignore known reasoning tags.

## Requirements

### Functional

1. Chinese translation prompts may instruct the model to put visible reasoning in `<think>...</think>`.
2. Japanese translation prompts remain unchanged.
3. The parser must continue to accept legacy responses containing `<translation_analysis>...</translation_analysis>`.
4. The parser must accept responses containing `<think>...</think>`.
5. The parser must always prefer the last `<translation>...</translation>` block as the canonical translation.
6. If `<translation>` is missing, fallback extraction must attempt to remove reasoning-tag content before storing the translation.
7. The raw response must still be preserved in `full_response`.

### Non-Functional

1. Existing Japanese artifacts and workflows must remain valid.
2. The change must be backward-compatible with previously generated prompt contracts.
3. Parser behavior must be deterministic for the same response string.

## Prompt Contract

Chinese translation prompts will be updated so that any visible reasoning is requested in `<think>...</think>`, followed by the final translation inside `<translation>...</translation>`.

Japanese prompts will remain unchanged and continue to request reasoning in `<translation_analysis>...</translation_analysis>`, followed by `<translation>...</translation>`.

The canonical output contract for all translation tasks remains:

- final answer must be enclosed in `<translation>...</translation>`

Reasoning tags are non-canonical. They exist only to guide model behavior and for optional inspection through `full_response`.

## Parser Contract

The parser will operate in this order:

1. Search for `<translation>...</translation>` blocks in the full response.
2. If one or more are present, return the last block as the translation.
3. If none are present:
   - remove `<think>...</think>` spans
   - remove `<translation_analysis>...</translation_analysis>` spans
   - if a closing reasoning tag exists, prefer the remainder after the last reasoning block
   - otherwise fall back to the cleaned response body
4. Never intentionally keep recognized reasoning tags inside the stored `translation` field.

This preserves compatibility while making fallback behavior less likely to contaminate translations with chain-of-thought content.

## Data And Artifact Impact

No schema change is required for translation artifacts. The following fields remain:

- `translation`
- `full_response`
- `prompt_template`
- `generation_config`

The only behavioral change is that `translation` extraction becomes stricter about ignoring reasoning-tag content.

## Error Handling

If a model omits `<translation>`:

- the run should still succeed through fallback parsing
- the fallback translation should be based on reasoning-stripped content
- the raw response remains available for inspection in `full_response`

This avoids turning prompt-format drift into a hard pipeline failure while still improving output cleanliness.

## Testing

Add parser tests covering:

1. Legacy JP response:
   - contains `<translation_analysis>` and `<translation>`
   - parser returns only the translation body

2. New CN response:
   - contains `<think>` and `<translation>`
   - parser returns only the translation body

3. Missing `<translation>` with reasoning present:
   - contains `<think>` or `<translation_analysis>` but no `<translation>`
   - parser falls back to cleaned non-reasoning content

4. Stray reasoning tags:
   - malformed or extra `<think>` blocks should not be copied into the final `translation` when a valid `<translation>` exists

5. Regression check:
   - prompt selection still resolves JP and CN prompt files correctly after the CN prompt text change

## Risks

1. Some models may emit malformed `<think>` tags.
   Mitigation: remove recognized spans opportunistically and rely on `<translation>` as the canonical field.

2. Some providers may internally treat `<think>` specially or suppress it from visible output.
   Mitigation: this is acceptable because the parser only depends on `<translation>`.

3. Visible reasoning can still consume completion budget.
   Mitigation: keep this change parser-focused for now and handle model-specific token-budget tuning separately.

## Alternatives Considered

### Keep `<translation_analysis>` everywhere

Pros:

- no prompt divergence between JP and CN
- lowest churn

Cons:

- less aligned with modern reasoning-model conventions for CN-oriented prompts

### Remove visible reasoning instructions entirely

Pros:

- cleanest long-term contract
- minimal parser complexity

Cons:

- larger prompt-behavior change
- harder to compare against existing workflows

### Recommended Option

Use `<think>` for Chinese prompts, keep Japanese prompts unchanged, and make the parser explicitly ignore both `<think>` and `<translation_analysis>`.

This is the narrowest change that improves compatibility with modern reasoning-heavy models without forcing a prompt migration across all tasks at once.

## Implementation Outline

1. Update the Chinese translation prompt file(s) to use `<think>` instead of `<translation_analysis>`.
2. Refactor the translation parser to strip recognized reasoning tags before fallback extraction.
3. Add parser tests for both legacy and new contracts.
4. Run targeted parser/prompt tests.
