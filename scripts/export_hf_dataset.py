import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_tools.translation_set_global import (
    CHINESE_SOURCE_MANIFEST_COLUMNS,
    INVENTORY_COLUMNS,
    build_chinese_source_manifest,
    build_translation_set_global,
    build_translation_set_inventory,
    write_dataset_card,
    validate_translation_set_global,
    write_csv_rows,
)


@click.command()
@click.option(
    "--output-root",
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
@click.option(
    "--inventory-path",
    type=click.Path(path_type=Path),
    default=Path("docs") / "translation_set_inventory.csv",
    show_default=True,
)
def main(output_root: Path, manifest_path: Path, inventory_path: Path) -> None:
    paths = build_translation_set_global(output_root=output_root)
    manifest_rows = build_chinese_source_manifest()
    inventory_rows = build_translation_set_inventory(template_path=inventory_path)
    readme_path = write_dataset_card(
        output_root,
        manifest_rows=manifest_rows,
        inventory_rows=inventory_rows,
    )

    write_csv_rows(manifest_path, manifest_rows, CHINESE_SOURCE_MANIFEST_COLUMNS)
    write_csv_rows(inventory_path, inventory_rows, INVENTORY_COLUMNS)

    click.echo(f"Built dataset export at {output_root}")
    for key, path in paths.items():
        click.echo(f"  - {key}: {path}")
    click.echo(f"  - dataset_card: {readme_path}")

    for task in (
        Path("benchmark_tasks/translation_ja_en_bidirectional_v1.yaml"),
        Path("benchmark_tasks/translation_zh_en_bidirectional_v1.yaml"),
        Path("benchmark_tasks/translation_zh_ja_bidirectional_v1.yaml"),
    ):
        counts = validate_translation_set_global(
            output_root,
            task=task,
            manifest_path=manifest_path,
        )
        for name, count in counts.items():
            click.echo(f"  - {name}: {count}")


if __name__ == "__main__":
    main()
