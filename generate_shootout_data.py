import os
import json
from itertools import combinations
import hashlib
from io import StringIO
from datasets import load_dataset
import click

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_translation_pair(conv_a, conv_b):
    """Format a pair of translations into a single markdown document."""
    output = StringIO()
    
    # Write the name and source text
    output.write(f"## Name: {conv_a.get('name', 'Unnamed')}\n\n")
    output.write("## Source Text:\n")
    output.write(f"{conv_a.get('source_text', '')}\n\n")
    
    # Write first translation
    output.write("## Translation A\n")
    output.write(f"{conv_a.get('translation', '')}\n\n")
    
    # Write second translation
    output.write("## Translation B\n")
    output.write(f"{conv_b.get('translation', '')}\n\n")
    
    # Add final separator
    output.write("---\n")
    
    return output.getvalue()

def write_pair_settings(file_a, file_b, example_name):
    """Format the basic settings for the translation pair."""
    return {
        "id": hashlib.md5(f"{file_a}_{file_b}_{example_name}".encode()).hexdigest(),
        "llm_a": os.path.splitext(file_a)[0],
        "llm_b": os.path.splitext(file_b)[0]
    }

def generate_translation_pairs(test_model_file=None):
    """Generate translation pairs comparing target file against all other models."""
    base_translations_dir = "base_translations"
    translations_dir = "translations"
    output_file = "latest_conversation_pairs.jsonl" if test_model_file else "base_conversation_pairs.jsonl"
    
    # Add warning and confirmation for base_conversation_pairs.jsonl
    if not test_model_file:
        print("\nWARNING: You are about to overwrite base_conversation_pairs.jsonl. These hold the pairs for all the models you'll be comparing against, and this could cause the program to stop working.")
        confirmation = input("Are you sure you want to continue? (yes/no): ")
        if confirmation.lower() != "yes":
            print("Operation cancelled.")
            return
    
    
    if test_model_file:
        # Check if test_model_file exists in translations directory
        if not os.path.exists(os.path.join(translations_dir, test_model_file)):
            raise click.BadParameter(f"File {test_model_file} not found in {translations_dir}")
        # Get all files from base_translations to compare against
        base_files = [f for f in os.listdir(base_translations_dir) if f.endswith('.jsonl')]
        pairs = [(test_model_file, base_file) for base_file in base_files]
        print(f"Comparing {test_model_file} against {len(base_files)} files from {base_translations_dir}")
    else:
        # Get all files from base_translations for pairwise comparison
        jsonl_files = [f for f in os.listdir(base_translations_dir) if f.endswith('.jsonl')]
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
            
            # Validate that files have equal length
            if len(convs_a) != len(convs_b):
                raise ValueError(f"Files have different lengths: {file_a} ({len(convs_a)} items) vs {file_b} ({len(convs_b)} items)")
            
            # For each translation pair in the files
            for conv_a, conv_b in zip(convs_a, convs_b):
                # Create settings with unique ID and model names
                settings = write_pair_settings(file_a, file_b, conv_a['name'])
                
                # Format both translations into a single markdown document
                formatted_data = format_translation_pair(conv_a, conv_b)
                
                # Combine into final format
                comparison_data = {
                    "id": settings["id"],
                    "llm_a": settings["llm_a"],
                    "llm_b": settings["llm_b"],
                    "formatted_data": formatted_data,
                    "name" : conv_a["name"],
                    "english": conv_a["english"],
                    "difficulty" : conv_a["difficulty"],

                }
                
                # Write to output file
                out_f.write(json.dumps(comparison_data, ensure_ascii=False) + '\n')
                total_pairs += 1

    items_per_file = next(iter(file_lengths.values()))  # Get count from first file
    total_files = len(file_lengths)
    print(f"Generated {total_pairs} total pairs written to {output_file} (comparing {total_files} files with {items_per_file} items each)")


@click.command()
@click.option('--test-model', help='Test model to generate pairs for. If not specified, pairs will be generated between all models.')
@click.option('--generate-base', is_flag=True, help='Generate base translation pairs. This will overwrite base_conversation_pairs.jsonl')
def main(test_model, generate_base):
    """Generate conversation pairs for evaluation."""
    if generate_base:
        # build all pairs in base_translations
        generate_translation_pairs()
    else:
        if not test_model:
            raise click.UsageError("Either --test-model or --generate-base must be specified")
        # Transform the model name into the target file path
        test_model_file = test_model.replace('/', '__') + '.jsonl'
        generate_translation_pairs(test_model_file)

if __name__ == "__main__":
    main()
