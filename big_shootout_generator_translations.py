import os
import json
from itertools import combinations
import hashlib
from io import StringIO
from datasets import load_dataset

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_conversation_pair(conv_a, conv_b):
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

def write_pair_settings(file_a, file_b):
    """Format the basic settings for the translation pair."""
    return {
        "id": hashlib.md5(f"{file_a}_{file_b}".encode()).hexdigest(),
        "llm_a": os.path.splitext(file_a)[0],
        "llm_b": os.path.splitext(file_b)[0]
    }

def generate_translation_pairs():
    translations_dir = "translations"
    output_file = "all_translation_pairs.jsonl"
    
    # Get all JSONL files
    jsonl_files = [f for f in os.listdir(translations_dir) if f.endswith('.jsonl')]
    
    # Generate unique pairs using combinations
    pairs = list(combinations(jsonl_files, 2))
    
    print(f"Found {len(jsonl_files)} files, generating {len(pairs)} unique pairs...")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_a, file_b in pairs:
            path_a = os.path.join(translations_dir, file_a)
            path_b = os.path.join(translations_dir, file_b)
            
            data_a = load_jsonl(path_a)
            data_b = load_jsonl(path_b)
            
            # For each translation pair in the files
            for conv_a, conv_b in zip(data_a, data_b):
                # Create settings with unique ID and translator names
                settings = write_pair_settings(file_a, file_b)
                
                # Format both translations into a single markdown document
                formatted_data = format_conversation_pair(conv_a, conv_b)
                
                # Combine into final format
                comparison_data = {
                    "id": settings["id"],
                    "llm_a": settings["llm_a"],
                    "llm_b": settings["llm_b"],
                    "formatted_data": formatted_data
                }
                
                # Write to output file
                out_f.write(json.dumps(comparison_data, ensure_ascii=False) + '\n')
                
    print(f"Generated translation pairs have been written to {output_file}")

if __name__ == "__main__":
    # Generate all possible pairs
    print("Generating all possible translation pairs...")
    generate_translation_pairs()
