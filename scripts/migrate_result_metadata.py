#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_metadata import score_metadata_sidecar_path, write_score_metadata_sidecar
from benchmark_tasks import load_task_config


@click.command()
@click.option("--scores-file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--task", envvar="TASK_CONFIG", required=True, help="Task config path or name under benchmark_tasks/.")
@click.option("--judgments-file", type=click.Path(dir_okay=False, path_type=Path), help="Optional judgments JSONL path. Defaults to the sibling judgments.jsonl.")
@click.option("--pairs-file", type=click.Path(dir_okay=False, path_type=Path), help="Optional pairs JSONL path. Defaults to the sibling pairs.jsonl.")
def main(scores_file: Path, task: str, judgments_file: Path | None, pairs_file: Path | None) -> None:
    """Write an additive scores.metadata.json sidecar next to an existing scores.json file."""
    score_payload = json.loads(scores_file.read_text(encoding="utf-8"))
    test_model = score_payload.get("model")
    if not test_model:
        raise click.ClickException(f"Could not infer test model from {scores_file}")

    task_config = load_task_config(task)
    judgments_file = judgments_file or scores_file.with_name("judgments.jsonl")
    pairs_file = pairs_file or scores_file.with_name("pairs.jsonl")

    sidecar_path = write_score_metadata_sidecar(
        scores_file,
        test_model=test_model,
        task_config=task_config,
        judgments_file=judgments_file,
        pairs_file=pairs_file if pairs_file.exists() else None,
        workdir=REPO_ROOT,
    )

    click.echo(f"Wrote metadata sidecar to {sidecar_path}")
    click.echo(f"Legacy scores file left unchanged: {scores_file}")
    click.echo(f"Sidecar path: {score_metadata_sidecar_path(scores_file)}")


if __name__ == "__main__":
    main()
