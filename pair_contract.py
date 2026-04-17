import hashlib
import json


PAIR_ID_SCHEMA_V1 = "v1"
PAIR_FINGERPRINT_V1_KEYS = (
    "llm_a",
    "llm_b",
    "formatted_data",
    "name",
    "english",
    "difficulty",
)
PAIR_FINGERPRINT_V1_PREFIXES = ("llm_a_", "llm_b_")


def compute_pair_id_v1(file_a: str, file_b: str, example_name: str) -> str:
    return hashlib.md5(f"{file_a}_{file_b}_{example_name}".encode()).hexdigest()


def pair_fingerprint_payload_v1(pair: dict) -> dict:
    payload = {}
    for key in PAIR_FINGERPRINT_V1_KEYS:
        if key in pair:
            payload[key] = pair[key]
    for key in sorted(pair):
        if key.startswith(PAIR_FINGERPRINT_V1_PREFIXES):
            payload[key] = pair[key]
    return payload


def compute_pair_fingerprint(pair: dict) -> str:
    payload = pair_fingerprint_payload_v1(pair)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ensure_pair_contract_metadata(pair: dict) -> dict:
    normalized = dict(pair)
    normalized.setdefault("pair_id_schema", PAIR_ID_SCHEMA_V1)
    normalized.setdefault("pair_fingerprint", compute_pair_fingerprint(normalized))
    return normalized
