import json
import os
from collections import defaultdict
import statistics

metrics = [
    "Roleplay Adherence",
    "Consistency",
    "Contextual Understanding",
    "Expressiveness",
    "Creativity",
    "Naturalness of Japanese",
    "Enjoyment of the Dialogue",
    "Appropriateness of Turn-Taking"
]

def process_file(filepath):
    scores = defaultdict(list)
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                for metric in metrics:
                    if metric in data:
                        scores[metric].append(float(data[metric]))
            except json.JSONDecodeError:
                continue
    
    # Calculate means for each metric
    metric_means = {}
    for metric in metrics:
        if scores[metric]:  # Only calculate if we have scores
            metric_means[metric] = statistics.mean(scores[metric])
    
    # Calculate total average
    if metric_means:
        total_average = statistics.mean(metric_means.values())
    else:
        total_average = 0
        
    return metric_means, total_average

def main():
    eval_dir = "/fsx/ubuntu/evals/Japanese-RP-Bench/evaluations"
    results = []
    
    for filename in os.listdir(eval_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(eval_dir, filename)
            model_name = filename.replace('.Aratako-Japanese-RP-Bench-testdata-SFW.jsonl', '')
            metric_means, total_average = process_file(filepath)
            
            results.append({
                'model': model_name,
                'metrics': metric_means,
                'total_average': total_average
            })
    
    # Sort by total average descending
    results.sort(key=lambda x: x['total_average'], reverse=True)
    
    # Print results
    print("\nResults sorted by total average score (highest to lowest):\n")
    print(f"{'Model':<50} | {'Total Avg':<10} | " + " | ".join(f"{m:<10}" for m in metrics))
    print("-" * (50 + 10 + (10 * len(metrics)) + (3 * (len(metrics) + 1))))
    
    for result in results:
        metrics_str = " | ".join(f"{result['metrics'].get(m, 'N/A'):10.2f}" if isinstance(result['metrics'].get(m), (int, float)) else f"{'N/A':<10}" for m in metrics)
        print(f"{result['model']:<50} | {result['total_average']:10.2f} | {metrics_str}")

if __name__ == "__main__":
    main()
