import click
import os
import json
from collections import Counter

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

@click.group()
def cli():
    """A utility to manage base translation sets."""
    pass

@cli.command("list-models")
@click.option('--judge', required=True, help='The judge model to inspect the base set of.')
def list_models(judge):
    """List models and their match counts in a judge's base set."""
    console = Console() if RICH_AVAILABLE else None
    base_set_file = find_base_set_file(judge)

    if not base_set_file:
        if console:
            console.print(f"[bold red]Error:[/] Base set for judge '{judge}' not found.")
        else:
            print(f"Error: Base set for judge '{judge}' not found.")
        return

    model_counts = Counter()
    with open(base_set_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'llm_a' in data:
                    model_counts[data['llm_a']] += 1
                if 'llm_b' in data:
                    model_counts[data['llm_b']] += 1
            except json.JSONDecodeError:
                if RICH_AVAILABLE:
                    console = Console()
                    console.print(f"[yellow]Warning:[/] Skipping malformed line in {base_set_file}")
                else:
                    print(f"Warning: Skipping malformed line in {base_set_file}")


    if not model_counts:
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"No models found in {base_set_file}")
        else:
            print(f"No models found in {base_set_file}")
        return

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title=f"Model counts in {base_set_file}")
        table.add_column("Model", justify="left", style="cyan", no_wrap=True)
        table.add_column("Matches", justify="right", style="magenta")

        for model, count in model_counts.most_common():
            table.add_row(safe_to_display_name(model), str(count))
        
        console.print(table)
    else:
        print(f"Model counts in {base_set_file}:")
        for model, count in model_counts.most_common():
            print(f"- {safe_to_display_name(model)}: {count} matches")

def find_base_set_file(judge):
    """Helper function to find the correct base set file for a judge."""
    # First, check for the direct match
    direct_match = f"base_sets/base_set.{judge}.jsonl"
    if os.path.exists(direct_match):
        return direct_match

    # If not found, search for files ending with the judge's name
    try:
        files = os.listdir('base_sets')
        for f in files:
            if f.endswith(f".{judge}.jsonl"):
                return os.path.join('base_sets', f)
    except FileNotFoundError:
        return None # base_sets directory doesn't exist

    return None

def safe_to_display_name(safe_name):
    """Convert safe storage format (with __) to user-friendly display format (with /)."""
    return safe_name.replace('__', '/')

def display_to_safe_name(display_name):
    """Convert user-friendly display format (with /) to safe storage format (with __)."""
    return display_name.replace('/', '__')

def find_similar_models(target_model, available_models, max_suggestions=3):
    """Find similar model names using fuzzy matching."""
    import difflib
    
    # Convert all models to display format for comparison
    display_models = [(safe_to_display_name(model), model) for model in available_models]
    display_names = [display_name for display_name, _ in display_models]
    
    # Find close matches
    close_matches = difflib.get_close_matches(target_model, display_names, n=max_suggestions, cutoff=0.3)
    
    # Return the original safe names for the matches
    suggestions = []
    for match in close_matches:
        for display_name, safe_name in display_models:
            if display_name == match:
                suggestions.append((display_name, safe_name))
                break
    
    return suggestions

def analyze_model_overlap(input_file, base_set_file):
    """Analyze overlap between models in input file and existing base set."""
    # Get existing model counts from base set
    existing_models = Counter()
    if os.path.exists(base_set_file):
        with open(base_set_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'llm_a' in data:
                        existing_models[data['llm_a']] += 1
                    if 'llm_b' in data:
                        existing_models[data['llm_b']] += 1
                except json.JSONDecodeError:
                    continue
    
    # Get model counts from input file
    input_models = Counter()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'llm_a' in data:
                    input_models[data['llm_a']] += 1
                if 'llm_b' in data:
                    input_models[data['llm_b']] += 1
            except json.JSONDecodeError:
                continue
    
    # Calculate overlap: for each existing model, count how many times it appears in input file
    overlap_data = []
    for existing_model in existing_models:
        shared_count = input_models.get(existing_model, 0)
        if shared_count > 0:
            overlap_data.append({
                'existing_model': existing_model,
                'existing_count': existing_models[existing_model],
                'shared_count': shared_count
            })
    
    return overlap_data, input_models, existing_models

@cli.command("add-model")
@click.option('--judge', required=True, help='The judge model for the base set.')
@click.option('--input-file', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='The .jsonl file with comparisons to add.')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt.')
def add(judge, input_file, yes):
    """Add new comparisons to a judge's base set."""
    console = Console() if RICH_AVAILABLE else None

    base_set_file = find_base_set_file(judge)

    if not base_set_file:
        if console:
            console.print(f"[bold red]Error:[/] Base set for judge '{judge}' not found. Cannot add new entries.")
        else:
            print(f"Error: Base set for judge '{judge}' not found. Cannot add new entries.")
        return

    # Analyze model overlap and display comparison
    try:
        overlap_data, input_models, existing_models = analyze_model_overlap(input_file, base_set_file)
        
        if console:
            # Display new models being added
            console.print(f"\n[bold cyan]Models in input file '{input_file}':[/]")
            new_models_table = Table()
            new_models_table.add_column("Model", justify="left", style="cyan")
            new_models_table.add_column("Matches in Input", justify="right", style="magenta")
            
            for model, count in input_models.most_common():
                new_models_table.add_row(safe_to_display_name(model), str(count))
            console.print(new_models_table)
            
            # Display overlap with existing models
            if overlap_data:
                console.print(f"\n[bold yellow]Overlap with existing models in '{base_set_file}':[/]")
                overlap_table = Table()
                overlap_table.add_column("Existing Model", justify="left", style="green")
                overlap_table.add_column("Current Matches", justify="right", style="blue")
                overlap_table.add_column("Shared with Input", justify="right", style="red")
                
                for item in sorted(overlap_data, key=lambda x: x['shared_count'], reverse=True):
                    overlap_table.add_row(
                        safe_to_display_name(item['existing_model']),
                        str(item['existing_count']),
                        str(item['shared_count'])
                    )
                console.print(overlap_table)
            else:
                console.print(f"\n[bold green]No overlap found with existing models in '{base_set_file}'[/]")
        else:
            # Plain text output
            print(f"\nModels in input file '{input_file}':")
            for model, count in input_models.most_common():
                print(f"- {safe_to_display_name(model)}: {count} matches")
            
            if overlap_data:
                print(f"\nOverlap with existing models in '{base_set_file}':")
                for item in sorted(overlap_data, key=lambda x: x['shared_count'], reverse=True):
                    print(f"- {safe_to_display_name(item['existing_model'])}: {item['existing_count']} current, {item['shared_count']} shared with input")
            else:
                print(f"\nNo overlap found with existing models in '{base_set_file}'")
    
    except Exception as e:
        if console:
            console.print(f"[yellow]Warning:[/] Could not analyze model overlap: {e}")
        else:
            print(f"Warning: Could not analyze model overlap: {e}")

    # Confirmation prompt
    if not yes:
        if not click.confirm(f"Are you sure you want to add entries from '{input_file}' to '{base_set_file}'?"):
            if console:
                console.print("[yellow]Operation cancelled.[/]")
            else:
                print("Operation cancelled.")
            return

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(base_set_file, 'a', encoding='utf-8') as outfile:
            count = 0
            for line in infile:
                # Basic validation to ensure it's a JSON line
                try:
                    json.loads(line)
                    outfile.write(line)
                    count += 1
                except json.JSONDecodeError:
                    if console:
                        console.print(f"[yellow]Warning:[/] Skipping malformed line in {input_file}: {line.strip()}")
                    else:
                        print(f"Warning: Skipping malformed line in {input_file}: {line.strip()}")
        
        if console:
            console.print(f"[bold green]Success:[/] Added {count} new entries to {base_set_file}")
        else:
            print(f"Success: Added {count} new entries to {base_set_file}")

    except Exception as e:
        if console:
            console.print(f"[bold red]An error occurred:[/] {e}")
        else:
            print(f"An error occurred: {e}")

@cli.command("remove-model")
@click.option('--judge', required=True, help='The judge model for the base set.')
@click.option('--model', required=True, help='The model to remove from the base set.')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt.')
def remove(judge, model, yes):
    """Remove all comparisons involving a specific model from a base set."""
    console = Console() if RICH_AVAILABLE else None
    
    # Convert user input to safe format for internal processing
    safe_model_name = display_to_safe_name(model)

    base_set_file = find_base_set_file(judge)

    if not base_set_file:
        if console:
            console.print(f"[bold red]Error:[/] Base set for judge '{judge}' not found.")
        else:
            print(f"Error: Base set for judge '{judge}' not found.")
        return

    # Display current model counts
    model_counts = Counter()
    with open(base_set_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'llm_a' in data:
                    model_counts[data['llm_a']] += 1
                if 'llm_b' in data:
                    model_counts[data['llm_b']] += 1
            except json.JSONDecodeError:
                continue

    if not model_counts:
        if console:
            console.print(f"No models found in {base_set_file}")
        else:
            print(f"No models found in {base_set_file}")
        return

    # Display the current model counts with highlighting for the model to be removed
    if console:
        table = Table(title=f"Current model counts in {base_set_file}")
        table.add_column("Model", justify="left", style="cyan", no_wrap=True)
        table.add_column("Matches", justify="right", style="magenta")

        for model_name, count in model_counts.most_common():
            display_name = safe_to_display_name(model_name)
            if model_name == safe_model_name:
                # Highlight the model being removed
                table.add_row(f"[bold red]{display_name}[/bold red] (TO BE REMOVED)", f"[bold red]{count}[/bold red]")
            else:
                table.add_row(display_name, str(count))
        
        console.print(table)
    else:
        print(f"Current model counts in {base_set_file}:")
        for model_name, count in model_counts.most_common():
            display_name = safe_to_display_name(model_name)
            if model_name == safe_model_name:
                print(f"- {display_name}: {count} matches (TO BE REMOVED)")
            else:
                print(f"- {display_name}: {count} matches")

    # Check if the model to be removed actually exists
    if safe_model_name not in model_counts:
        # Find similar models and suggest alternatives
        suggestions = find_similar_models(model, list(model_counts.keys()))
        
        if console:
            console.print(f"\n[bold red]Error:[/] Model '{model}' not found in {base_set_file}")
            if suggestions:
                console.print(f"\n[bold cyan]Did you mean one of these?[/]")
                for display_name, safe_name in suggestions:
                    console.print(f"  [green]•[/] {display_name}")
                    console.print(f"    [dim]Command: python base_set_manager.py remove-model --model \"{display_name}\" --judge {judge}[/]")
        else:
            print(f"\nError: Model '{model}' not found in {base_set_file}")
            if suggestions:
                print(f"\nDid you mean one of these?")
                for display_name, safe_name in suggestions:
                    print(f"  • {display_name}")
                    print(f"    Command: python base_set_manager.py remove-model --model \"{display_name}\" --judge {judge}")
        
        return  # Exit early instead of continuing

    # Confirmation prompt
    if not yes:
        if not click.confirm(f"Are you sure you want to remove all entries involving '{model}' from '{base_set_file}'?"):
            if console:
                console.print("[yellow]Operation cancelled.[/]")
            else:
                print("Operation cancelled.")
            return

    temp_file = base_set_file + '.tmp'
    removed_count = 0
    original_count = 0

    try:
        with open(base_set_file, 'r', encoding='utf-8') as infile, open(temp_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                original_count += 1
                try:
                    data = json.loads(line)
                    if data.get('llm_a') == safe_model_name or data.get('llm_b') == safe_model_name:
                        removed_count += 1
                    else:
                        outfile.write(line)
                except json.JSONDecodeError:
                    # Keep malformed lines as they are
                    outfile.write(line)

        # Replace the original file with the temporary one
        os.replace(temp_file, base_set_file)

        if console:
            if removed_count > 0:
                console.print(f"[bold green]Success:[/] Removed {removed_count} entries involving '{model}' from {base_set_file}.")
                console.print(f"The base set now contains {original_count - removed_count} entries.")
            else:
                console.print(f"[bold yellow]Info:[/] No entries involving '{model}' found in {base_set_file}.")
        else:
            if removed_count > 0:
                print(f"Success: Removed {removed_count} entries involving '{model}' from {base_set_file}.")
                print(f"The base set now contains {original_count - removed_count} entries.")
            else:
                print(f"Info: No entries involving '{model}' found in {base_set_file}.")

    except Exception as e:
        if console:
            console.print(f"[bold red]An error occurred:[/] {e}")
        else:
            print(f"An error occurred: {e}")
        # Clean up the temp file if an error occurs
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    cli()
