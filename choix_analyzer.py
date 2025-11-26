import choix
import numpy as np
from typing import List, Tuple
import pandas as pd
import click
import glob
import json
import re
import os
from datetime import datetime
from pathlib import Path
 
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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
    file_path = str(file_path)
    comparisons = []
    # Check if this is a base set file
    is_base_file = 'base_set.' in file_path
    skipped_missing_answer = 0
    
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
                    skipped_missing_answer += 1
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
                        
                comparisons.append((llm1, llm2, winner, difficulty, is_english))
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON line encountered")
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    
    if skipped_missing_answer > 0:
        print(f"Skipped {skipped_missing_answer} pairs due to missing <answer> tags in {file_path}")
    
    return comparisons



def display_rankings(console, rankings_df, title, target_model=None):
    """Displays rankings in a formatted table, highlighting the target model."""
    if RICH_AVAILABLE and console:
        table = Table(title=title)
        
        # Define columns
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("LLM", style="magenta")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Wins", justify="right", style="blue")
        table.add_column("Total Matches", justify="right", style="yellow")
        table.add_column("EN", justify="right", style="red")
        table.add_column("LT", justify="right", style="purple")

        # Add rows
        for index, row in rankings_df.iterrows():
            llm_name = row['llm']
            style = "on yellow" if target_model and target_model in llm_name else None
            
            table.add_row(
                str(index + 1),
                llm_name,
                f"{row['score']:.4f}",
                str(row['wins']),
                str(row['total_matches']),
                f"{row['EN']:.4f}",
                f"{row['LT']:.4f}",
                style=style
            )
            
        console.print(table)
    else:
        # Plain text fallback
        print(f"\n{title}")
        print("=" * len(title))
        print(f"{'Rank':<4} {'LLM':<30} {'Score':<8} {'Wins':<6} {'Matches':<8} {'EN':<8} {'LT':<8}")
        print("-" * 72)
        
        for index, row in rankings_df.iterrows():
            llm_name = row['llm']
            highlight = " <-- TARGET" if target_model and target_model in llm_name else ""
            
            print(f"{index + 1:<4} {llm_name:<30} {row['score']:<8.4f} {row['wins']:<6} {row['total_matches']:<8} {row['EN']:<8.4f} {row['LT']:<8.4f}{highlight}")


@click.command()
@click.option('--test-model', '-m', required=True, help='Name of the model being evaluated')
@click.option('--judge-model', '-j', required=True, help='Name of the model that did the judging')
@click.option('--judgments-file', help='Optional path to judged comparisons (default results/<baseset_version>/<model>/<judge>/judgments.jsonl).')
@click.option('--pairs-file', help='Optional path to pairs file for coverage accounting.')
@click.option('--baseset-version', help='Baseset version label (default derived from BASESET_SNAPSHOT_DIR).')
def main(test_model, judge_model, judgments_file, pairs_file, baseset_version):
    safe_judge_name = judge_model.replace("/", "__")
    safe_model_name = test_model.replace("/", "__")
    base_version = baseset_version or Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0")).name
    default_results_dir = Path("results") / base_version / safe_model_name / safe_judge_name
    if not judgments_file:
        judgments_file = default_results_dir / "judgments.jsonl"
    else:
        judgments_file = Path(judgments_file)
    if not pairs_file:
        pairs_file = default_results_dir / "pairs.jsonl"
    else:
        pairs_file = Path(pairs_file)

    # Always load base set comparisons first
    snapshot_base_file = Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0")) / f"base_set.{safe_judge_name}.jsonl"
    if not snapshot_base_file.exists():
        print(f"Base set file not found at {snapshot_base_file}. Set BASESET_SNAPSHOT_DIR or ensure the file exists.")
        exit(1)
    base_file = snapshot_base_file
    print(f"\nProcessing base set file: {base_file}...")
    comparisons = load_comparisons_from_file(base_file)
    base_comparisons_count = len(comparisons)

    if not os.path.exists(judgments_file):
        print(f"Target model judgments file not found: {judgments_file}")
        exit(1)
    print(f"\nProcessing test model file: {judgments_file}...")
    target_comparisons = load_comparisons_from_file(judgments_file)
    comparisons.extend(target_comparisons)
    judged_pairs = len(target_comparisons)

    expected_pairs = None
    if pairs_file and pairs_file.exists():
        expected_pairs = sum(1 for _ in pairs_file.open("r", encoding="utf-8"))
    missing_pairs = None
    if expected_pairs is not None:
        missing_pairs = max(expected_pairs - judged_pairs, 0)

    if not comparisons:
        print("No valid comparisons found in any files")
        exit(1)

    # Fit the model
    ranker = LLMRanker()
    ranker.fit(comparisons)

    def extract_slice(diff: str, language: str):
        try:
            rankings = ranker.get_rankings(diff, language)
        except ValueError:
            return None
        rankings['llm'] = rankings['llm'].str.replace('__', '/')
        row = rankings[rankings['llm'] == test_model]
        if row.empty:
            return None
        row = row.iloc[0]
        wins = int(row['wins'])
        total = int(row['total_matches'])
        lt = float(row['LT'])
        win_rate = wins / total if total else 0.0
        return {
            "difficulty": diff,
            "language": language,
            "lt": lt,
            "wins": wins,
            "total": total,
            "win_rate": win_rate,
        }

    en_ja = {
        "overall": extract_slice("all", "english"),
        "easy": extract_slice("easy", "english"),
        "hard": extract_slice("hard", "english"),
    }
    ja_en = {
        "overall": extract_slice("all", "japanese"),
        "easy": extract_slice("easy", "japanese"),
        "hard": extract_slice("hard", "japanese"),
    }

    output_dir = default_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_file = output_dir / "scores.json"

    summary = {
        "model": test_model,
        "judge_model": judge_model,
        "baseset_version": base_version,
        "pairs_file": str(pairs_file) if pairs_file else None,
        "judgments_file": str(judgments_file),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "expected_pairs": expected_pairs,
        "judged_pairs": judged_pairs,
        "missing_pairs": missing_pairs,
        "base_comparisons": base_comparisons_count,
        "en_ja": en_ja,
        "ja_en": ja_en,
    }

    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nScores saved to: {scores_file}")
    print(f"Raw answers can be found at: {judgments_file}")

if __name__ == "__main__":
    main()
