import json
import os
import re
import click


def clean_file(file_path):
    """Reads a .jsonl file, filters for valid entries, and overwrites the original file."""
    cleaned_lines = []
    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if 'analysis' not in data:
                    print(f"  - Skipping line {i+1}: 'analysis' field missing.")
                    continue

                match = re.search(r'<answer>(.*?)</answer>', data['analysis'])
                if not match:
                    print(f"  - Skipping line {i+1}: <answer> tag missing.")
                    continue

                answer_content = match.group(1).strip().lower()
                if answer_content not in ['a', 'b']:
                    print(f"  - Skipping line {i+1}: Invalid answer content: '{match.group(1)}'")
                    continue

                cleaned_lines.append(line)

            except json.JSONDecodeError:
                print(f"  - Skipping line {i+1}: Invalid JSON.")
                continue

    # Write the cleaned data back to the original file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"Finished cleaning {file_path}. Found {len(cleaned_lines)} valid lines.")

@click.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
def main(file_path):
    """Cleans a specified .jsonl base set file by removing invalid entries."""
    clean_file(file_path)

if __name__ == "__main__":
    main()
