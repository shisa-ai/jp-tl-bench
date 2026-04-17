---
configs:
- config_name: translation_ja_en_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_ja_en_bidirectional_v1/train.jsonl
- config_name: translation_zh_en_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_zh_en_bidirectional_v1/train.jsonl
---

# bt_translation_set_global

Private benchmark dataset export for the task configs in this repository.

## Configs

- `translation_ja_en_bidirectional_v1`: immutable JP v1 export with `70` rows
- `translation_zh_en_bidirectional_v1`: bidirectional ZH/EN export with `33` curated Chinese-source rows and shared English-source rows

## Source Provenance

- Chinese source manifest rows: `33`
- Chinese source families: literary prose / excerpt (3), consumer help / official support (2), classical prose / literary-historical text (2), business advice / cold-calling guide (1), business advice / replacement source (1), business advice / CRM explainer (1)
- Inventory category highlights: conversation (6), recipe (4), novel_excerpt (3), sales_training (3), light_novel (3), consumer_help (2)

## Row Schema

Each row uses the same lightweight contract:

- `item_id`: stable item identifier
- `name`: display name
- `text`: source text
- `difficulty`: `easy` or `hard`
- `language`: source-language code
- `metadata`: `NA` for legacy rows or a JSON string for curated Chinese rows
