import choix
import numpy as np
from typing import List, Tuple
import pandas as pd

class LLMRanker:
    def __init__(self):
        self.llm_to_idx = {}  # Maps LLM names to consecutive integers
        self.idx_to_llm = {}  # Reverse mapping
        self.n_items = 0
        self.params = None
        self.wins_count = {}  # Track raw number of wins for each LLM

    def process_comparisons(self, comparisons: List[Tuple[str, str, str]]):
        """
        Process raw comparison data into format needed for choix.
        
        Args:
            comparisons: List of tuples (llm1, llm2, winner)
                        where winner is either "llm1" or "llm2"
        """
        # First, build the mapping of LLM names to integers
        unique_llms = set()
        for llm1, llm2, _ in comparisons:
            unique_llms.add(llm1)
            unique_llms.add(llm2)
        
        self.llm_to_idx = {llm: idx for idx, llm in enumerate(unique_llms)}
        self.idx_to_llm = {idx: llm for llm, idx in self.llm_to_idx.items()}
        self.n_items = len(self.llm_to_idx)
        
        # Initialize wins count for each LLM
        self.wins_count = {llm: 0 for llm in unique_llms}
        
        # Convert comparisons to format needed by choix
        processed_comparisons = []
        for llm1, llm2, winner in comparisons:
            idx1 = self.llm_to_idx[llm1]
            idx2 = self.llm_to_idx[llm2]
            # Track wins
            self.wins_count[winner] += 1
            if winner == llm1:
                processed_comparisons.append((idx1, idx2))
            else:
                processed_comparisons.append((idx2, idx1))
                
        return processed_comparisons

    def fit(self, comparisons: List[Tuple[str, str, str]]):
        """
        Fit the Bradley-Terry model to the comparison data.
        """
        processed_data = self.process_comparisons(comparisons)
        self.params = choix.opt_pairwise(self.n_items, processed_data)
        
    def get_rankings(self) -> pd.DataFrame:
        """
        Returns a DataFrame with LLMs ranked by their scores and win counts.
        """
        if self.params is None:
            raise ValueError("Must fit model before getting rankings")
        
        '''
        See more about EN & LT: https://chatgpt.com/share/67b34c25-61c8-8012-8667-17077284d92a
        '''

        # 1) Exponential & Normalize (EN)
        exp_params = np.exp(self.params)
        sum_exp = np.sum(exp_params)
        en_scores = exp_params / sum_exp  # 0-1 scale
        en_scores_0_10 = en_scores * 10   # 0-10 scale if desired

        # 2) Logistic Transform (LT)
        # Shift so the average model has param=0 => logistic transform is 0.5 on average
        mean_param = np.mean(self.params)
        shifted_params = self.params - mean_param
        lt_scores = 1.0 / (1.0 + np.exp(-shifted_params))  # 0-1 scale
        lt_scores_0_10 = lt_scores * 10                    # 0-10 scale if desired
            
        rankings = pd.DataFrame({
            'llm': [self.idx_to_llm[i] for i in range(self.n_items)],
            'score': self.params,
            'wins': [self.wins_count[self.idx_to_llm[i]] for i in range(self.n_items)],
            'EN': en_scores_0_10,
            'LT': lt_scores_0_10,
        })
        return rankings.sort_values('score', ascending=False).reset_index(drop=True)
    
    def predict_winner_probability(self, llm1: str, llm2: str) -> float:
        """
        Predict probability that llm1 will win against llm2.
        """
        if self.params is None:
            raise ValueError("Must fit model before making predictions")
            
        idx1 = self.llm_to_idx[llm1]
        idx2 = self.llm_to_idx[llm2]
        prob, _ = choix.probabilities((idx1, idx2), self.params)
        return prob

# Example usage:
if __name__ == "__main__":
    import json
    import re
    import os
    import glob

    # Read and process all JSONL files in the analysis directory
    comparisons = []
    analysis_files = glob.glob('analysis/*.jsonl')
    
    for file_path in analysis_files:
        print(f"Processing {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'llm_a' not in data or 'llm_b' not in data or 'analysis' not in data:
                        continue
                    
                    # Extract the winner from analysis field
                    match = re.search(r'<answer>(.*?)</answer>', data['analysis'])
                    if not match:
                        continue
                        
                    answer_content = match.group(1)
                    # Remove anything that isn't an ASCII letter
                    cleaned_answer = ''.join(c for c in answer_content if c.isalpha())
                    
                    if not cleaned_answer:
                        continue
                        
                    cleaned_answer = cleaned_answer.lower()
                    
                    llm1 = data['llm_a']
                    llm2 = data['llm_b']
                    
                    if cleaned_answer == 'a':
                        winner = llm1
                    elif cleaned_answer == 'b':
                        winner = llm2
                    else:
                        print(f"Error: Invalid answer content in <answer> tag: {answer_content}")
                        continue
                        
                    comparisons.append((llm1, llm2, winner))
                    
                except json.JSONDecodeError:
                    print("Error: Invalid JSON line encountered")
                    continue
                except Exception as e:
                    print(f"Error processing line: {str(e)}")
                    continue
    
    if not comparisons:
        print("No valid comparisons found in any files")
        exit(1)
        
    # Initialize and fit the model
    ranker = LLMRanker()
    ranker.fit(comparisons)
    
    # Get rankings
    rankings = ranker.get_rankings()
    
    # Convert __ back to / in model names
    rankings['llm'] = rankings['llm'].str.replace('__', '/')
    
    print("\nRankings:")
    print(rankings)
    
    # Print win counts
    print("\nRaw win counts:")
    for llm, wins in sorted(ranker.wins_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{llm}: {wins} wins")
    
    # Save rankings to scores/scores.jsonl
    os.makedirs('scores', exist_ok=True)
    with open('scores/scores.jsonl', 'w') as f:
        rankings_dict = rankings.to_dict(orient='records')
        for rank in rankings_dict:
            json.dump(rank, f)
            f.write('\n')
