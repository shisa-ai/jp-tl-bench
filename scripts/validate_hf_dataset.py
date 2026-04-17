import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_tools.translation_set_global import validate_translation_set_global


@click.command()
@click.option(
    "--task",
    required=True,
    type=click.Path(path_type=Path),
    help="Task config path under benchmark_tasks/ to validate against.",
)
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path),
    default=Path("hf_datasets") / "bt_translation_set_global",
    show_default=True,
)
@click.option(
    "--manifest-path",
    type=click.Path(path_type=Path),
    default=Path("docs") / "chinese_source_manifest.csv",
    show_default=True,
)
def main(task: Path, dataset_root: Path, manifest_path: Path) -> None:
    counts = validate_translation_set_global(
        dataset_root,
        task=task,
        manifest_path=manifest_path,
    )
    click.echo(f"Validated dataset export at {dataset_root}")
    for name, count in counts.items():
        click.echo(f"  - {name}: {count}")


if __name__ == "__main__":
    main()
