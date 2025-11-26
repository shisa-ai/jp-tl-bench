#!/usr/bin/env bash

set -euo pipefail

BASESET_SNAPSHOT_DIR="${BASESET_SNAPSHOT_DIR:-baseset/v1.0}"
BASE_VERSION="$(basename "$BASESET_SNAPSHOT_DIR")"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
JUDGE_URL="${JUDGE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-GEMINI_API_KEY}"
SAFE_JUDGE="${JUDGE_MODEL//\//__}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if ! compgen -G "translations/*.jsonl" >/dev/null; then
    log "No translations/*.jsonl found. Nothing to do."
    exit 0
fi

log "Starting refactor transition"
log "Base set: $BASESET_SNAPSHOT_DIR (version: $BASE_VERSION)"
log "Judge model: $JUDGE_MODEL (safe: $SAFE_JUDGE)"
log "Judge URL: $JUDGE_URL"
log "Judge API key env: $JUDGE_API_KEY_ENV"

TOTAL=0

# Pass 1: copy legacy judgments into new locations (idempotent)
log "Pass 1: migrating existing judgments to results/ if absent"
for tf in translations/*.jsonl; do
    [ -e "$tf" ] || continue
    SAFE_MODEL="$(basename "$tf" .jsonl)"
    MODEL="${SAFE_MODEL//__/\/}"
    MODEL_DIR="results/${BASE_VERSION}/${SAFE_MODEL}/${SAFE_JUDGE}"
    JUDGMENTS_PATH="${MODEL_DIR}/judgments.jsonl"
    OLD_JUDGMENTS="scores/${SAFE_MODEL}.${SAFE_JUDGE}.jsonl"

    mkdir -p "$MODEL_DIR"
    if [ -f "$OLD_JUDGMENTS" ] && [ ! -f "$JUDGMENTS_PATH" ]; then
        cp "$OLD_JUDGMENTS" "$JUDGMENTS_PATH"
        log "Copied existing judgments: $OLD_JUDGMENTS -> $JUDGMENTS_PATH"
    fi
done
log "Pass 1 complete."

# Pass 2: generate pairs, backfill judgments, compute scores (reuse-by-default)
log "Pass 2: generating pairs, judgments, scores"

for tf in translations/*.jsonl; do
    [ -e "$tf" ] || continue
    TOTAL=$((TOTAL + 1))
    SAFE_MODEL="$(basename "$tf" .jsonl)"
    MODEL="${SAFE_MODEL//__/\/}"
    MODEL_DIR="results/${BASE_VERSION}/${SAFE_MODEL}/${SAFE_JUDGE}"
    PAIRS_PATH="${MODEL_DIR}/pairs.jsonl"
    JUDGMENTS_PATH="${MODEL_DIR}/judgments.jsonl"
    SCORES_PATH="${MODEL_DIR}/scores.json"
    OLD_JUDGMENTS="scores/${SAFE_MODEL}.${SAFE_JUDGE}.jsonl"

    log "=== Processing $MODEL (safe: $SAFE_MODEL) ==="
    mkdir -p "$MODEL_DIR"

    if [ -f "$JUDGMENTS_PATH" ]; then
        log "Existing judgments present: $JUDGMENTS_PATH"
    elif [ -f "$OLD_JUDGMENTS" ]; then
        log "Found legacy judgments to migrate: $OLD_JUDGMENTS"
        cp "$OLD_JUDGMENTS" "$JUDGMENTS_PATH"
    else
        log "No existing judgments found for $MODEL; will backfill."
    fi

    # Generate pairs file
    log "Generating pairs: $PAIRS_PATH"
    if ! python generate_shootout_data.py --test-model "$MODEL" --judge-model "$JUDGE_MODEL" --output "$PAIRS_PATH" --yes; then
        log "Pair generation failed for $MODEL (likely missing/incomplete translations); skipping."
        continue
    fi

    # Run comparer (reuse existing judgments unless rejudge is specified by caller)
    log "Running comparer (reuse-by-default)…"
    if ! python translation_comparer_any_model.py \
        --base-url "$JUDGE_URL" \
        --judge-model "$JUDGE_MODEL" \
        --test-model "$MODEL" \
        --pairs-file "$PAIRS_PATH" \
        --api-key-env "$JUDGE_API_KEY_ENV"; then
        log "Comparer failed for $MODEL; skipping."
        continue
    fi

    # choic analyzer to emit canonical scores
    log "Generating canonical scores: $SCORES_PATH"
    if ! python choix_analyzer.py \
        --test-model "$MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --judgments-file "$JUDGMENTS_PATH" \
        --pairs-file "$PAIRS_PATH" \
        --baseset-version "$BASE_VERSION"; then
        log "Choix analyzer failed for $MODEL; skipping."
        continue
    fi

    log "Completed $MODEL"
done

log "Refactor transition done. Processed $TOTAL model(s)."
