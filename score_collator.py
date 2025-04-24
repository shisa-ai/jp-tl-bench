import os
import json
import csv
import glob

def collate_scores():
    # Find all _tl_bench_scores.jsonl files in the scores folder
    score_files = glob.glob('scores/*_tl_bench_scores.jsonl')
    
    # Dictionary to store unique LLM entries
    unique_llms = {}
    
    # Process each file
    for file_path in score_files:
        print(file_path)
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Check if this is an "all/all" entry
                    if data.get('difficulty') == 'all' and data.get('language') == 'all':
                        llm_name = data.get('llm')
                        
                        # Clean up the LLM name by removing prefixes
                        if llm_name:
                            if llm_name.startswith('base//fsx2/outputs/'):
                                llm_name = llm_name[len('base//fsx2/outputs/'):]
                            elif llm_name.startswith('base/'):
                                llm_name = llm_name[len('base/'):]
                            
                            # Update the LLM name in the data
                            data['llm'] = llm_name
                            
                            # Only add if this LLM isn't already in our dictionary
                            if llm_name not in unique_llms:
                                unique_llms[llm_name] = data
                except json.JSONDecodeError:
                    continue
    
    # Write results to CSV
    with open('tl_bench_scores_summary.csv', 'w', newline='') as csvfile:
        # Define CSV headers based on the data structure
        fieldnames = ['llm', 'score', 'wins', 'total_matches', 'EN', 'LT']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        # Sort by score in descending order
        for llm_name, data in sorted(unique_llms.items(), key=lambda x: x[1]['score'], reverse=True):
            writer.writerow({
                'llm': llm_name,
                'score': data['score'],
                'wins': data['wins'],
                'total_matches': data['total_matches'],
                'EN': data['EN'],
                'LT': data['LT']
            })
    
    print(f"Successfully collated scores from {len(score_files)} files.")
    print(f"Found {len(unique_llms)} unique LLMs.")
    print(f"Results saved to tl_bench_scores_summary.csv")

if __name__ == "__main__":
    collate_scores() 
