import choix
import numpy as np
from typing import List, Tuple
import pandas as pd
import click
import glob
import json
import re
import os
import shutil 

class LLMRanker:
    def __init__(self):
        self.llm_to_idx = {}  # Maps LLM names to consecutive integers
        self.idx_to_llm = {}  # Reverse mapping
        self.n_items = 0
        self.params = None
        self.wins_count = {}  # Track raw number of wins for each LLM
        self.wins_by_difficulty = {'easy': {}, 'hard': {}}
        self.wins_by_language = {
            'english': {'all': {}, 'easy': {}, 'hard': {}},
            'japanese': {'all': {}, 'easy': {}, 'hard': {}}
        }
        self.total_matches = {
            'all': {},
            'easy': {},
            'hard': {},
            'english': {'all': {}, 'easy': {}, 'hard': {}},
            'japanese': {'all': {}, 'easy': {}, 'hard': {}}
        }

    def process_comparisons(self, comparisons: List[Tuple[str, str, str, str, bool]]):
        """
        Process raw comparison data into format needed for choix.
        
        Args:
            comparisons: List of tuples (llm1, llm2, winner, difficulty, is_english)
        """
        # First, build the mapping of LLM names to integers
        unique_llms = set()
        for llm1, llm2, _, _, _ in comparisons:
            unique_llms.add(llm1)
            unique_llms.add(llm2)
        
        self.llm_to_idx = {llm: idx for idx, llm in enumerate(unique_llms)}
        self.idx_to_llm = {idx: llm for llm, idx in self.llm_to_idx.items()}
        self.n_items = len(self.llm_to_idx)
        
        # Initialize counters
        self.wins_count = {llm: 0 for llm in unique_llms}
        self.wins_by_difficulty = {
            'easy': {llm: 0 for llm in unique_llms},
            'hard': {llm: 0 for llm in unique_llms}
        }
        self.wins_by_language = {
            'english': {
                'all': {llm: 0 for llm in unique_llms},
                'easy': {llm: 0 for llm in unique_llms},
                'hard': {llm: 0 for llm in unique_llms}
            },
            'japanese': {
                'all': {llm: 0 for llm in unique_llms},
                'easy': {llm: 0 for llm in unique_llms},
                'hard': {llm: 0 for llm in unique_llms}
            }
        }
        
        # Initialize total_matches for all categories
        self.total_matches = {
            'all': {llm: 0 for llm in unique_llms},
            'easy': {llm: 0 for llm in unique_llms},
            'hard': {llm: 0 for llm in unique_llms},
            'english': {
                'all': {llm: 0 for llm in unique_llms},
                'easy': {llm: 0 for llm in unique_llms},
                'hard': {llm: 0 for llm in unique_llms}
            },
            'japanese': {
                'all': {llm: 0 for llm in unique_llms},
                'easy': {llm: 0 for llm in unique_llms},
                'hard': {llm: 0 for llm in unique_llms}
            }
        }
        
        # Initialize processed_comparisons
        processed_comparisons = {
            'all': [],
            'easy': [],
            'hard': [],
            'english': {'all': [], 'easy': [], 'hard': []},
            'japanese': {'all': [], 'easy': [], 'hard': []}
        }
        
        for llm1, llm2, winner, difficulty, is_english in comparisons:
            idx1 = self.llm_to_idx[llm1]
            idx2 = self.llm_to_idx[llm2]
            lang = 'english' if is_english else 'japanese'
            
            # Track total matches for both LLMs in all relevant categories
            for model in [llm1, llm2]:
                # Overall
                self.total_matches['all'][model] += 1
                # By difficulty
                self.total_matches[difficulty][model] += 1
                # By language
                self.total_matches[lang]['all'][model] += 1
                self.total_matches[lang][difficulty][model] += 1
            
            # Track wins
            self.wins_count[winner] += 1
            self.wins_by_difficulty[difficulty][winner] += 1
            self.wins_by_language[lang]['all'][winner] += 1
            self.wins_by_language[lang][difficulty][winner] += 1
            
            if winner == llm1:
                comparison = (idx1, idx2)
            else:
                comparison = (idx2, idx1)
                
            processed_comparisons['all'].append(comparison)
            processed_comparisons[difficulty].append(comparison)
            processed_comparisons[lang]['all'].append(comparison)
            processed_comparisons[lang][difficulty].append(comparison)
                
        return processed_comparisons

    def fit(self, comparisons: List[Tuple[str, str, str, str, bool]]):
        """
        Fit the Bradley-Terry model to all comparison subsets.
        """
        processed_data = self.process_comparisons(comparisons)
        self.params = {
            'all': choix.opt_pairwise(self.n_items, processed_data['all']),
            'easy': choix.opt_pairwise(self.n_items, processed_data['easy']) if processed_data['easy'] else None,
            'hard': choix.opt_pairwise(self.n_items, processed_data['hard']) if processed_data['hard'] else None,
            'english': {
                'all': choix.opt_pairwise(self.n_items, processed_data['english']['all']) if processed_data['english']['all'] else None,
                'easy': choix.opt_pairwise(self.n_items, processed_data['english']['easy']) if processed_data['english']['easy'] else None,
                'hard': choix.opt_pairwise(self.n_items, processed_data['english']['hard']) if processed_data['english']['hard'] else None
            },
            'japanese': {
                'all': choix.opt_pairwise(self.n_items, processed_data['japanese']['all']) if processed_data['japanese']['all'] else None,
                'easy': choix.opt_pairwise(self.n_items, processed_data['japanese']['easy']) if processed_data['japanese']['easy'] else None,
                'hard': choix.opt_pairwise(self.n_items, processed_data['japanese']['hard']) if processed_data['japanese']['hard'] else None
            }
        }
        
    def get_rankings(self, difficulty: str = 'all', language: str = None) -> pd.DataFrame:
        """
        Returns a DataFrame with LLMs ranked by their scores and win counts.
        
        Args:
            difficulty: 'all', 'easy', or 'hard'
            language: None, 'english', or 'japanese'
        """
        '''
        See more about EN & LT: https://chatgpt.com/share/67b34c25-61c8-8012-8667-17077284d92a
        '''

        if language:
            if self.params is None or self.params[language] is None or self.params[language][difficulty] is None:
                raise ValueError(f"No data for {language} {difficulty} rankings")
            params = self.params[language][difficulty]
            wins_dict = self.wins_by_language[language][difficulty]
        else:
            if self.params is None or self.params[difficulty] is None:
                raise ValueError(f"No data for {difficulty} rankings")
            params = self.params[difficulty]
            wins_dict = self.wins_by_difficulty[difficulty] if difficulty != 'all' else self.wins_count

        # Calculate EN scores
        exp_params = np.exp(params)
        sum_exp = np.sum(exp_params)
        en_scores = exp_params / sum_exp
        en_scores_0_10 = en_scores * 10

        # Calculate LT scores
        mean_param = np.mean(params)
        shifted_params = params - mean_param
        lt_scores = 1.0 / (1.0 + np.exp(-shifted_params))
        lt_scores_0_10 = lt_scores * 10
            
        # Get the appropriate total_matches dictionary
        if language:
            total_matches_dict = self.total_matches[language][difficulty]
        else:
            total_matches_dict = self.total_matches[difficulty]
            
        rankings = pd.DataFrame({
            'llm': [self.idx_to_llm[i] for i in range(self.n_items)],
            'score': params,
            'wins': [wins_dict[self.idx_to_llm[i]] for i in range(self.n_items)],
            'total_matches': [total_matches_dict[self.idx_to_llm[i]] for i in range(self.n_items)],
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
        prob, _ = choix.probabilities((idx1, idx2), self.params['all'])
        return prob



def load_comparisons_from_file(file_path):
    """Helper function to load comparisons from a file."""
    comparisons = []
    # Check if this is a base set file
    is_base_file = 'base_set.' in file_path
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'llm_a' not in data or 'llm_b' not in data or 'analysis' not in data:
                    continue
                
                # Extract difficulty and language
                difficulty = data['difficulty']
                is_english = data['english']
                
                if difficulty not in ['easy', 'hard']:
                    raise ValueError(f"Invalid difficulty value '{difficulty}'. Must be 'easy' or 'hard'")
                
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
                
                '''
                In order to have proper handling of models that appear in the base set, we prefix base set models with a 'base__'

                The base model is always llm2 for our target model comparison file so we have to prefix 'base__' as well there
                '''
                llm1 = data['llm_a']
                llm2 = data['llm_b']

                
                if cleaned_answer == 'a':
                    winner = llm1
                elif cleaned_answer == 'b':
                    winner = llm2
                else:
                    print(f"Error: Invalid answer content in <answer> tag: {answer_content}")
                    continue
                        
                comparisons.append((llm1, llm2, winner, difficulty, is_english))
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON line encountered")
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    return comparisons



@click.command()
@click.option('--target-model', '-m', required=False, help='Name of the model being evaluated')
@click.option('--judge-model', '-j', required=False, help='Name of the model did the judging')

def main(target_model, judge_model):
    # Always load base set comparisons first
    comparisons = []
    base_files = [f for f in glob.glob('analysis/base_set.*.jsonl')]
    if base_files:
        print("\nProcessing base set comparisons...")
        for base_file in base_files:
            comparisons.extend(load_comparisons_from_file(base_file))
    
    # If target model and judge model are specified, load those comparisons too
    if target_model and judge_model:
        safe_model_name = target_model.replace("/", "__")
        safe_judge_name = judge_model.replace("/", "__")
        file_path = f'analysis/{safe_model_name}.{safe_judge_name}.jsonl'
        
        if os.path.exists(file_path):
            print(f"\nProcessing {file_path}...")
            comparisons.extend(load_comparisons_from_file(file_path))
        else:
            print(f"Analysis file not found: {file_path}")
            if not comparisons:  # If we don't even have base comparisons
                exit(1)
    
    if not comparisons:
        print("No valid comparisons found in any files")
        exit(1)
            
    # Initialize and fit the model
    ranker = LLMRanker()
    ranker.fit(comparisons)
    
    # First print overall rankings
    print("\n=== Overall Rankings ===")
    for diff in ['all', 'easy', 'hard']:
        try:
            rankings = ranker.get_rankings(diff)
            rankings['llm'] = rankings['llm'].str.replace('__', '/')
            
            print(f"\nRankings for {diff} questions:")
            print(rankings)
            
            print(f"\nRaw win counts for {diff} questions:")
            wins_dict = ranker.wins_by_difficulty[diff] if diff != 'all' else ranker.wins_count
            for llm, wins in sorted(wins_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"{llm}: {wins} wins")
        except ValueError as e:
            print(f"\nNo data available for {diff} difficulty")
            continue

    # Then print language-specific rankings
    for lang in ['english', 'japanese']:
        print(f"\n=== {lang.title()} Rankings ===")
        for diff in ['all', 'easy', 'hard']:
            try:
                rankings = ranker.get_rankings(diff, lang)
                rankings['llm'] = rankings['llm'].str.replace('__', '/')
                
                print(f"\nRankings for {lang.title()}({diff}) questions:")
                print(rankings)
                
                print(f"\nRaw win counts for {lang.title()}({diff}) questions:")
                wins_dict = ranker.wins_by_language[lang][diff]
                for llm, wins in sorted(wins_dict.items(), key=lambda x: x[1], reverse=True):
                    print(f"{llm}: {wins} wins")
            except ValueError as e:
                print(f"\nNo data available for {lang} {diff} difficulty")
                continue
    
    # Only save files if both model names are provided
    if target_model and judge_model:
        # Save rankings with safe model names
        safe_model_name = target_model.replace("/", "__")
        
        # Create scores directory if it doesn't exist
        os.makedirs('scores', exist_ok=True)
        
        # Save scores
        scores_file = f'scores/{safe_model_name}_tl_bench_scores.jsonl'
        
        with open(scores_file, 'w') as f:
            # Save overall rankings for each difficulty
            for diff in ['all', 'easy', 'hard']:
                try:
                    rankings = ranker.get_rankings(diff)
                    rankings['difficulty'] = diff
                    rankings['language'] = 'all'
                    rankings['llm'] = rankings['llm'].str.replace('__', '/')
                    rankings.to_json(f, orient='records', lines=True)
                except ValueError:
                    continue

            # Save language-specific rankings
            for lang in ['english', 'japanese']:
                for diff in ['all', 'easy', 'hard']:
                    try:
                        rankings = ranker.get_rankings(diff, lang)
                        rankings['difficulty'] = diff
                        rankings['language'] = lang
                        rankings['llm'] = rankings['llm'].str.replace('__', '/')
                        rankings.to_json(f, orient='records', lines=True)
                    except ValueError:
                        continue
        
        print(f"\nScores saved to: {scores_file}")
        
        # Save raw answers for analysis
        answers_file = f'scores/{safe_model_name}_tl_bench_answers.jsonl'
        shutil.move(file_path, answers_file)
        print(f"Results saved to: {answers_file}")

if __name__ == "__main__":
    main()