import choix
import numpy as np
from typing import List, Tuple
import pandas as pd
import click
import glob
import json
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from artifact_paths import candidate_results_dir, resolve_result_file_candidates
from artifact_metadata import write_score_metadata_sidecar
from baseset.legacy_boundary import legacy_candidate_paths
from benchmark_tasks import load_judge_profile, load_task_config
 
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class LLMRanker:
    def __init__(self, task=None):
        self.task_config = load_task_config(task)
        self.direction_aliases = {}
        for direction in self.task_config.directions:
            self.direction_aliases[direction.key] = direction.key
            self.direction_aliases[direction.summary_language_label] = direction.key
        self.llm_to_idx = {}  # Maps LLM names to consecutive integers
        self.idx_to_llm = {}  # Reverse mapping
        self.n_items = 0
        self.params = None
        self.wins_count = {}  # Track raw number of wins for each LLM
        self.wins_by_difficulty = {'easy': {}, 'hard': {}}
        self.wins_by_direction = {}
        self.total_matches = {
            'all': {},
            'easy': {},
            'hard': {},
            'directions': {},
        }

    def process_comparisons(self, comparisons: List[Tuple[str, str, str, str, str]]):
        """
        Process raw comparison data into format needed for choix.
        
        Args:
            comparisons: List of tuples (llm1, llm2, winner, difficulty, direction_key)
        """
        # First, build the mapping of LLM names to integers
        unique_llms = set()
        direction_keys = sorted({direction_key for _, _, _, _, direction_key in comparisons})
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
        self.wins_by_direction = {
            direction_key: {
                'all': {llm: 0 for llm in unique_llms},
                'easy': {llm: 0 for llm in unique_llms},
                'hard': {llm: 0 for llm in unique_llms}
            }
            for direction_key in direction_keys
        }
        
        # Initialize total_matches for all categories
        self.total_matches = {
            'all': {llm: 0 for llm in unique_llms},
            'easy': {llm: 0 for llm in unique_llms},
            'hard': {llm: 0 for llm in unique_llms},
            'directions': {
                direction_key: {
                    'all': {llm: 0 for llm in unique_llms},
                    'easy': {llm: 0 for llm in unique_llms},
                    'hard': {llm: 0 for llm in unique_llms}
                }
                for direction_key in direction_keys
            },
        }
        
        # Initialize processed_comparisons
        processed_comparisons = {
            'all': [],
            'easy': [],
            'hard': [],
            'directions': {
                direction_key: {'all': [], 'easy': [], 'hard': []}
                for direction_key in direction_keys
            },
        }
        
        for llm1, llm2, winner, difficulty, direction_key in comparisons:
            idx1 = self.llm_to_idx[llm1]
            idx2 = self.llm_to_idx[llm2]
            
            # Track total matches for both LLMs in all relevant categories
            for model in [llm1, llm2]:
                # Overall
                self.total_matches['all'][model] += 1
                # By difficulty
                self.total_matches[difficulty][model] += 1
                # By direction
                self.total_matches['directions'][direction_key]['all'][model] += 1
                self.total_matches['directions'][direction_key][difficulty][model] += 1
            
            # Track wins
            self.wins_count[winner] += 1
            self.wins_by_difficulty[difficulty][winner] += 1
            self.wins_by_direction[direction_key]['all'][winner] += 1
            self.wins_by_direction[direction_key][difficulty][winner] += 1
            
            if winner == llm1:
                comparison = (idx1, idx2)
            else:
                comparison = (idx2, idx1)
                
            processed_comparisons['all'].append(comparison)
            processed_comparisons[difficulty].append(comparison)
            processed_comparisons['directions'][direction_key]['all'].append(comparison)
            processed_comparisons['directions'][direction_key][difficulty].append(comparison)
                
        return processed_comparisons

    def fit(self, comparisons: List[Tuple[str, str, str, str, str]]):
        """
        Fit the Bradley-Terry model to all comparison subsets.
        """
        processed_data = self.process_comparisons(comparisons)
        self.params = {
            'all': choix.opt_pairwise(self.n_items, processed_data['all']),
            'easy': choix.opt_pairwise(self.n_items, processed_data['easy']) if processed_data['easy'] else None,
            'hard': choix.opt_pairwise(self.n_items, processed_data['hard']) if processed_data['hard'] else None,
            'directions': {
                direction_key: {
                    'all': choix.opt_pairwise(self.n_items, processed_data['directions'][direction_key]['all']) if processed_data['directions'][direction_key]['all'] else None,
                    'easy': choix.opt_pairwise(self.n_items, processed_data['directions'][direction_key]['easy']) if processed_data['directions'][direction_key]['easy'] else None,
                    'hard': choix.opt_pairwise(self.n_items, processed_data['directions'][direction_key]['hard']) if processed_data['directions'][direction_key]['hard'] else None,
                }
                for direction_key in processed_data['directions']
            },
        }
        
    def get_rankings(self, difficulty: str = 'all', language: str = None) -> pd.DataFrame:
        """
        Returns a DataFrame with LLMs ranked by their scores and win counts.
        
        Args:
            difficulty: 'all', 'easy', or 'hard'
            language: None, 'all', a legacy alias ('english'/'japanese'), or a task direction key
        """
        '''
        See more about EN & LT: https://chatgpt.com/share/67b34c25-61c8-8012-8667-17077284d92a
        '''

        if language and language != 'all':
            direction_key = self.direction_aliases.get(language, language)
            if (
                self.params is None
                or direction_key not in self.params['directions']
                or self.params['directions'][direction_key][difficulty] is None
            ):
                raise ValueError(f"No data for {language} {difficulty} rankings")
            params = self.params['directions'][direction_key][difficulty]
            wins_dict = self.wins_by_direction[direction_key][difficulty]
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
        if language and language != 'all':
            direction_key = self.direction_aliases.get(language, language)
            total_matches_dict = self.total_matches['directions'][direction_key][difficulty]
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



def load_comparisons_from_file(file_path, task=None):
    """Helper function to load comparisons from a file."""
    file_path = str(file_path)
    comparisons = []
    skipped_missing_answer = 0
    task_config = load_task_config(task)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'llm_a' not in data or 'llm_b' not in data or 'analysis' not in data:
                    continue
                
                normalized = task_config.normalize_record(data)
                difficulty = normalized['difficulty']
                direction = task_config.direction_for_record(normalized)
                
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
                        
                comparisons.append((llm1, llm2, winner, difficulty, direction.key))
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON line encountered")
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    
    if skipped_missing_answer > 0:
        print(f"Skipped {skipped_missing_answer} pairs due to missing <answer> tags in {file_path}")
    
    return comparisons


def extract_model_slice(ranker, test_model: str, difficulty: str, language: str, summary_language_label: str):
    try:
        rankings = ranker.get_rankings(difficulty, language)
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
        "difficulty": difficulty,
        "language": summary_language_label,
        "lt": lt,
        "wins": wins,
        "total": total,
        "win_rate": win_rate,
    }


def canonical_slice_label(direction_key: str | None, difficulty: str = "all") -> str:
    if not direction_key:
        return "overall"
    if difficulty == "all":
        return direction_key
    return f"{direction_key}_{difficulty}"


def iter_score_slice_specs(task=None):
    task_config = load_task_config(task)
    specs = [
        {
            "slice": "overall",
            "difficulty": "all",
            "direction_key": None,
            "title": "Overall (all directions)",
            "summary_language_label": "all",
            "source_language": None,
            "target_language": None,
            "direction_name": "Overall",
        }
    ]
    for direction_key in task_config.scoring_direction_order:
        direction = task_config.direction_by_key(direction_key)
        specs.append(
            {
                "slice": canonical_slice_label(direction.key, "all"),
                "difficulty": "all",
                "direction_key": direction.key,
                "title": f"{direction.display_name} Overall",
                "summary_language_label": direction.summary_language_label,
                "source_language": direction.source_language,
                "target_language": direction.target_language,
                "direction_name": direction.display_name,
            }
        )
        for difficulty in ("easy", "hard"):
            specs.append(
                {
                    "slice": canonical_slice_label(direction.key, difficulty),
                    "difficulty": difficulty,
                    "direction_key": direction.key,
                    "title": f"{direction.display_name} {difficulty.title()}",
                    "summary_language_label": direction.summary_language_label,
                    "source_language": direction.source_language,
                    "target_language": direction.target_language,
                    "direction_name": direction.display_name,
                }
            )
    return specs


def build_ranked_slice_rows(ranker, manifest_models=None, task=None):
    manifest_models = manifest_models or {}
    rows = []
    for spec in iter_score_slice_specs(task=task):
        language = spec["direction_key"]
        try:
            rankings = ranker.get_rankings(spec["difficulty"], language)
        except ValueError:
            continue
        for _, row in rankings.iterrows():
            safe = row["llm"]
            matches = int(row["total_matches"])
            wins = int(row["wins"])
            rows.append(
                {
                    "slice": spec["slice"],
                    "model": manifest_models.get(safe, safe.replace("__", "/")),
                    "safe_name": safe,
                    "score": float(row["score"]),
                    "wins": wins,
                    "matches": matches,
                    "win_rate": wins / matches * 100 if matches else 0.0,
                    "EN": float(row["EN"]),
                    "LT": float(row["LT"]),
                    "difficulty": spec["difficulty"],
                    "language": spec["summary_language_label"],
                    "direction_key": spec["direction_key"] or "overall",
                    "source_language": spec["source_language"],
                    "target_language": spec["target_language"],
                    "direction_name": spec["direction_name"],
                }
            )
    return rows


def build_score_summary(test_model, judge_model, base_version, base_file, judgments_file, pairs_file=None, task=None):
    base_file = Path(base_file)
    judgments_file = Path(judgments_file)
    pairs_file = Path(pairs_file) if pairs_file else None
    task_config = load_task_config(task)

    comparisons = load_comparisons_from_file(base_file, task=task)
    base_comparisons_count = len(comparisons)

    target_comparisons = load_comparisons_from_file(judgments_file, task=task)
    comparisons.extend(target_comparisons)
    judged_pairs = len(target_comparisons)

    expected_pairs = None
    if pairs_file and pairs_file.exists():
        with pairs_file.open("r", encoding="utf-8") as f:
            expected_pairs = sum(1 for _ in f)
    missing_pairs = None
    if expected_pairs is not None:
        missing_pairs = max(expected_pairs - judged_pairs, 0)

    if not comparisons:
        raise ValueError("No valid comparisons found in any files")

    ranker = LLMRanker(task=task)
    ranker.fit(comparisons)

    summary = {
        "model": test_model,
        "judge_model": judge_model,
        "baseset_version": base_version,
        "pairs_file": str(pairs_file) if pairs_file else None,
        "judgments_file": str(judgments_file),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "expected_pairs": expected_pairs,
        "judged_pairs": judged_pairs,
        "missing_pairs": missing_pairs,
        "base_comparisons": base_comparisons_count,
        "task_id": task_config.task_id,
        "task_type": task_config.task_type,
        "task_version": task_config.task_version,
        "task_config_digest": task_config.task_config_digest,
    }
    for direction_key in task_config.scoring_direction_order:
        direction = task_config.direction_by_key(direction_key)
        summary[direction.key] = {
            "overall": extract_model_slice(ranker, test_model, "all", direction.key, direction.summary_language_label),
            "easy": extract_model_slice(ranker, test_model, "easy", direction.key, direction.summary_language_label),
            "hard": extract_model_slice(ranker, test_model, "hard", direction.key, direction.summary_language_label),
        }
    return summary



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
@click.option('--judgments-file', help='Optional path to judged comparisons (default results/<baseset_version>/<model>/<judge_dir>/judgments.jsonl).')
@click.option('--pairs-file', help='Optional path to pairs file for coverage accounting.')
@click.option('--baseset-version', help='Baseset version label (default derived from BASESET_SNAPSHOT_DIR).')
@click.option('--task', envvar='TASK_CONFIG', help='Task config path or name under benchmark_tasks/.')
@click.option(
    '--judge-profile',
    envvar='JUDGE_PROFILE',
    default='default',
    show_default=True,
    help='Judge profile path or name under judge_profiles/. Used to scope the default results path.',
)
def main(test_model, judge_model, judgments_file, pairs_file, baseset_version, task, judge_profile):
    safe_judge_name = judge_model.replace("/", "__")
    base_version = baseset_version or Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0")).name
    judge_profile_config = load_judge_profile(judge_profile)
    default_results_dir = candidate_results_dir(
        base_version,
        test_model,
        judge_model,
        judge_profile_id=judge_profile_config.profile_id,
    )
    if not judgments_file:
        judgment_candidates = resolve_result_file_candidates(
            base_version,
            test_model,
            judge_model,
            "judgments.jsonl",
            judge_profile_id=judge_profile_config.profile_id,
        )
        judgments_file = next((path for path in judgment_candidates if path.exists()), judgment_candidates[0])
    else:
        judgments_file = Path(judgments_file)
    if not pairs_file:
        pair_candidates = resolve_result_file_candidates(
            base_version,
            test_model,
            judge_model,
            "pairs.jsonl",
            judge_profile_id=judge_profile_config.profile_id,
        )
        pairs_file = next((path for path in pair_candidates if path.exists()), pair_candidates[0])
    else:
        pairs_file = Path(pairs_file)

    # Always load base set comparisons first
    snapshot_dir = Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0"))
    snapshot_base_file = snapshot_dir / f"base_set.{safe_judge_name}.jsonl"
    base_file = None
    for candidate in legacy_candidate_paths(snapshot_base_file, snapshot_dir):
        if candidate.exists():
            base_file = candidate
            break
    if base_file is None:
        print(f"Base set file not found at {snapshot_base_file}. Set BASESET_SNAPSHOT_DIR or ensure the file exists.")
        exit(1)
    print(f"\nProcessing base set file: {base_file}...")
    if not os.path.exists(judgments_file):
        print(f"Target model judgments file not found: {judgments_file}")
        exit(1)
    print(f"\nProcessing test model file: {judgments_file}...")
    try:
        summary = build_score_summary(
            test_model=test_model,
            judge_model=judge_model,
            base_version=base_version,
            base_file=base_file,
            judgments_file=judgments_file,
            pairs_file=pairs_file,
            task=task,
        )
    except ValueError as exc:
        print(str(exc))
        exit(1)

    output_dir = default_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_file = output_dir / "scores.json"

    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    task_config = load_task_config(task)
    sidecar_path = write_score_metadata_sidecar(
        scores_file,
        test_model=test_model,
        task_config=task_config,
        judgments_file=judgments_file,
        pairs_file=pairs_file,
        workdir=Path.cwd(),
    )

    print(f"\nScores saved to: {scores_file}")
    print(f"Metadata sidecar saved to: {sidecar_path}")
    print(f"Raw answers can be found at: {judgments_file}")

if __name__ == "__main__":
    main()
