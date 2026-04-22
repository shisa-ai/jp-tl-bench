import os
import json
from itertools import combinations
from io import StringIO
import click
from artifact_paths import preferred_result_file
from benchmark_tasks import load_judge_profile
from pair_contract import (
    PAIR_ID_SCHEMA_V1,
    compute_pair_fingerprint,
    compute_pair_id_v1,
)
from benchmark_tasks import load_task_config

BASESET_SNAPSHOT_DIR = os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0")
FAILED_TRANSLATION_PREFIX = "[TRANSLATION FAILED:"
TASK_IDENTITY_FIELDS = (
    "item_id",
    "name",
    "task_id",
    "task_type",
    "task_version",
    "source_text",
    "difficulty",
    "source_language",
    "target_language",
    "english",
)
TASK_SLICE_TAG_FIELDS = ("category", "tags", "slice_tags")


def default_pairs_path(
    test_model: str,
    judge_model: str | None = None,
    judge_profile_id: str = "default",
) -> str:
    """Compute per-model pairs path under results/<baseset_version>/<model>/<judge_dir>/pairs.jsonl."""
    base_version = os.path.basename(os.path.normpath(BASESET_SNAPSHOT_DIR))
    return str(
        preferred_result_file(
            base_version,
            test_model,
            judge_model or "default",
            "pairs.jsonl",
            judge_profile_id=judge_profile_id,
        )
    )

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def validate_translation_record(item, file_label):
    """Reject unresolved failed generations before they enter pair generation."""
    status = item.get("status")
    translation = item.get("translation", "")
    item_label = item.get("item_id") or item.get("name", "unknown")
    if status and status != "ok":
        raise ValueError(
            f"Cannot generate pairs from failed generation in {file_label} for item {item_label}"
        )
    if isinstance(translation, str) and translation.startswith(FAILED_TRANSLATION_PREFIX):
        raise ValueError(
            f"Cannot generate pairs from failed generation in {file_label} for item {item_label}"
        )

def format_translation_pair(conv_a, conv_b):
    """Format a pair of translations with explicit judge-safe section tags."""
    output = StringIO()

    output.write("<item>\n")
    output.write("<name>\n")
    output.write(f"{conv_a.get('name', 'Unnamed')}\n")
    output.write("</name>\n\n")
    output.write("<source_text>\n")
    output.write(f"{conv_a.get('source_text', '')}\n")
    output.write("</source_text>\n\n")
    output.write("<translation_a>\n")
    output.write(f"{conv_a.get('translation', '')}\n")
    output.write("</translation_a>\n\n")
    output.write("<translation_b>\n")
    output.write(f"{conv_b.get('translation', '')}\n")
    output.write("</translation_b>\n")
    output.write("</item>\n")

    return output.getvalue()

def write_pair_settings(file_a, file_b, example_name):
    """Format the basic settings for the translation pair."""
    return {
        "id": compute_pair_id_v1(file_a, file_b, example_name),
        "pair_id_schema": PAIR_ID_SCHEMA_V1,
        "llm_a": os.path.splitext(file_a)[0],
        "llm_b": os.path.splitext(file_b)[0]
    }

def record_match_key(item: dict) -> str:
    return item.get("item_id") or item.get("name")


def task_slice_metadata(item: dict) -> dict:
    metadata = {}
    for key in TASK_SLICE_TAG_FIELDS:
        if key in item:
            metadata[key] = item[key]
    return metadata


def validate_task_identity_fields(conv_a: dict, conv_b: dict, file_a: str, file_b: str) -> None:
    candidate_fields = list(TASK_IDENTITY_FIELDS)
    candidate_fields.extend(
        key for key in TASK_SLICE_TAG_FIELDS if key in conv_a or key in conv_b
    )
    for key in candidate_fields:
        if conv_a.get(key) != conv_b.get(key):
            raise ValueError(
                f"Task-defining field mismatch for '{conv_a.get('item_id') or conv_a.get('name')}' "
                f"between {file_a} and {file_b}: {key}={conv_a.get(key)!r} vs {conv_b.get(key)!r}"
            )


def generate_translation_pairs(
    test_model_file=None,
    force=False,
    output_path=None,
    judge_model=None,
    judge_profile_id: str = "default",
    task=None,
    translations_dir: str = "translations",
):
    """Generate translation pairs comparing target file against all other models."""
    task_config = load_task_config(task)
    base_translations_dir = os.path.join(BASESET_SNAPSHOT_DIR, "translations")
    snapshot_version = os.path.basename(os.path.normpath(BASESET_SNAPSHOT_DIR))
    if output_path:
        output_file = output_path
    elif test_model_file:
        # Default per-model location, parallel-safe
        pretty_name = os.path.splitext(test_model_file)[0].replace("__", "/")
        output_file = default_pairs_path(
            pretty_name,
            judge_model,
            judge_profile_id=judge_profile_id,
        )
    else:
        output_file = "base_conversation_pairs.jsonl"
    
    # Add warning and confirmation for base_conversation_pairs.jsonl
    if not test_model_file and not force:
        if not click.confirm("\nWARNING: You are about to overwrite base_conversation_pairs.jsonl. These hold the pairs for all the models you'll be comparing against, and this could cause the program to stop working.\n\nAre you sure you want to continue?"):
            print("Operation cancelled.")
            return

    if test_model_file:
        # Check if test_model_file exists in translations directory
        if not os.path.exists(os.path.join(translations_dir, test_model_file)):
            raise click.BadParameter(f"File {test_model_file} not found in {translations_dir}")
        # Get all files from base_translations to compare against
        base_files = sorted(f for f in os.listdir(base_translations_dir) if f.endswith('.jsonl'))
        pairs = [(test_model_file, base_file) for base_file in base_files]
        print(f"Comparing {test_model_file} against {len(base_files)} files from {base_translations_dir}")
    else:
        # Get all files from base_translations for pairwise comparison
        jsonl_files = sorted(f for f in os.listdir(base_translations_dir) if f.endswith('.jsonl'))
        pairs = list(combinations(jsonl_files, 2))
        print(f"Generating all pairwise combinations from {len(jsonl_files)} files in {base_translations_dir}")
    
    # Process pairs and write to output
    total_pairs = 0
    file_lengths = {}
    
    # First pass to count items in each file
    for file_a, file_b in pairs:
        if test_model_file:
            file_a_path = os.path.join(translations_dir, file_a)
            file_b_path = os.path.join(base_translations_dir, file_b)
        else:
            file_a_path = os.path.join(base_translations_dir, file_a)
            file_b_path = os.path.join(base_translations_dir, file_b)
        
        if file_a not in file_lengths:
            file_lengths[file_a] = len(load_jsonl(file_a_path))
        if file_b not in file_lengths:
            file_lengths[file_b] = len(load_jsonl(file_b_path))
    
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_a, file_b in pairs:
            if test_model_file:
                # Load target file from translations and comparison file from base_translations
                convs_a = load_jsonl(os.path.join(translations_dir, file_a))
                convs_b = load_jsonl(os.path.join(base_translations_dir, file_b))
            else:
                # Load both files from the same working directory
                convs_a = load_jsonl(os.path.join(base_translations_dir, file_a))
                convs_b = load_jsonl(os.path.join(base_translations_dir, file_b))

            for item in convs_a:
                validate_translation_record(item, file_a)
            for item in convs_b:
                validate_translation_record(item, file_b)

            convs_a = [task_config.normalize_record(item) for item in convs_a]
            convs_b = [task_config.normalize_record(item) for item in convs_b]
            
            # Validate that files have equal length
            if len(convs_a) != len(convs_b):
                raise ValueError(f"Files have different lengths: {file_a} ({len(convs_a)} items) vs {file_b} ({len(convs_b)} items)")
            
            # Create dictionaries indexed by item_id when available, falling back to legacy name.
            convs_a_dict = {record_match_key(item): item for item in convs_a}
            convs_b_dict = {record_match_key(item): item for item in convs_b}
            
            # Validate that all items match between files
            names_a = set(convs_a_dict.keys())
            names_b = set(convs_b_dict.keys())
            
            if names_a != names_b:
                missing_in_a = names_b - names_a
                missing_in_b = names_a - names_b
                error_msg = f"Items don't match between files {file_a} and {file_b}."
                if missing_in_a:
                    error_msg += f" Missing in {file_a}: {sorted(missing_in_a)}"
                if missing_in_b:
                    error_msg += f" Missing in {file_b}: {sorted(missing_in_b)}"
                raise ValueError(error_msg)
            
            # For each translation pair, match by name
            for name in sorted(names_a):  # Sort for consistent ordering
                conv_a = convs_a_dict[name]
                conv_b = convs_b_dict[name]
                validate_task_identity_fields(conv_a, conv_b, file_a, file_b)
                # Create settings with unique ID and model names
                settings = write_pair_settings(file_a, file_b, conv_a['name'])
                
                # Format both translations into a single markdown document
                formatted_data = format_translation_pair(conv_a, conv_b)
                
                # Combine into final format
                comparison_data = {
                    "id": settings["id"],
                    "pair_id_schema": settings["pair_id_schema"],
                    "llm_a": settings["llm_a"],
                    "llm_b": settings["llm_b"],
                    "formatted_data": formatted_data,
                    "item_id": conv_a["item_id"],
                    "name" : conv_a["name"],
                    "task_id": conv_a["task_id"],
                    "task_type": conv_a["task_type"],
                    "task_version": conv_a["task_version"],
                    "snapshot_version": snapshot_version,
                    "source_language": conv_a["source_language"],
                    "target_language": conv_a["target_language"],
                    "difficulty" : conv_a["difficulty"],
                    "llm_a_low_context": conv_a.get("low_context", False),
                    "llm_a_ultra_low_context": conv_a.get("ultra_low_context", False),
                    "llm_a_temperature": conv_a.get("temperature"),
                    "llm_a_generation_config": conv_a.get("generation_config"),
                    "llm_b_low_context": conv_b.get("low_context", False),
                    "llm_b_ultra_low_context": conv_b.get("ultra_low_context", False),
                    "llm_b_temperature": conv_b.get("temperature"),
                    "llm_b_generation_config": conv_b.get("generation_config"),

                }
                if "english" in conv_a:
                    comparison_data["english"] = conv_a["english"]
                comparison_data.update(task_slice_metadata(conv_a))
                comparison_data["pair_fingerprint"] = compute_pair_fingerprint(comparison_data)
                
                # Write to output file
                out_f.write(json.dumps(comparison_data, ensure_ascii=False) + '\n')
                total_pairs += 1

    items_per_file = next(iter(file_lengths.values()))  # Get count from first file
    total_files = len(file_lengths)
    print(f"Generated {total_pairs} total pairs written to {output_file} (comparing {total_files} files with {items_per_file} items each)")


@click.command()
@click.option('--task', envvar='TASK_CONFIG', help='Task config path or name under benchmark_tasks/.')
@click.option('--test-model', help='Test model to generate pairs for. If not specified, pairs will be generated between all models.')
@click.option('--generate-base', is_flag=True, help='Generate base translation pairs. This will overwrite base_conversation_pairs.jsonl')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts for automation/CI')
@click.option('--output', help='Optional output path for pairs file (default per-model results path when --test-model).')
@click.option('--judge-model', help='Optional judge name for embedding in default results path.')
@click.option(
    '--translations-dir',
    default=os.getenv("TRANSLATIONS_DIR", "translations"),
    show_default=True,
    help='Directory containing the candidate model translation JSONL files.',
)
@click.option(
    '--judge-profile',
    envvar='JUDGE_PROFILE',
    default='default',
    show_default=True,
    help='Judge profile path or name under judge_profiles/. Used to scope the default results path.',
)
def main(task, test_model, generate_base, yes, output, judge_model, translations_dir, judge_profile):
    """Generate conversation pairs for evaluation."""
    judge_profile_config = load_judge_profile(judge_profile)
    if generate_base:
        # build all pairs in base_translations
        generate_translation_pairs(force=yes, output_path=output, task=task)
    else:
        if not test_model:
            raise click.UsageError("Either --test-model or --generate-base must be specified")
        # Transform the model name into the target file path
        test_model_file = test_model.replace('/', '__') + '.jsonl'
        generate_translation_pairs(
            test_model_file,
            force=yes,
            output_path=output,
            judge_model=judge_model,
            judge_profile_id=judge_profile_config.profile_id,
            task=task,
            translations_dir=translations_dir,
        )

if __name__ == "__main__":
    main()
