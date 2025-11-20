#!/usr/bin/env python3
"""Interactive Textual TUI for browsing translation benchmark judgments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

BASE_DIR = Path(__file__).parent
SCORES_DIR = BASE_DIR / "scores"
BASESET_DIR = BASE_DIR / "baseset" / "v1.0"

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, ListItem, ListView, Static
except ImportError as exc:
    App = None
    ComposeResult = None
    Horizontal = Vertical = VerticalScroll = None
    reactive = None
    Footer = Header = ListItem = ListView = Static = None
    TEXTUAL_IMPORT_ERROR = exc
else:
    TEXTUAL_IMPORT_ERROR = None


def display_model_name(safe_name: str) -> str:
    """Convert safe model name back to display format."""
    return safe_name.replace("__", "/")


def safe_name_from_model(model: str) -> str:
    """Convert model name to safe filename format."""
    return model.replace("/", "__")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as src:
            return json.load(src)
    except json.JSONDecodeError:
        return None


def safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """Safely convert value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ComparisonRecord:
    """A single pairwise comparison judgment."""
    index: int
    name: str
    english: bool
    difficulty: str
    comparison_id: str
    llm_a: str
    llm_b: str
    source_text: str
    translation_a: str
    translation_b: str
    analysis: str
    answer: str  # "A" or "B"
    judge_model: str
    raw: Dict[str, Any]

    @property
    def winner(self) -> str:
        """Return the winning model name."""
        return self.llm_a if self.answer == "A" else self.llm_b

    @property
    def loser(self) -> str:
        """Return the losing model name."""
        return self.llm_b if self.answer == "A" else self.llm_a

    @property
    def is_test_model_win(self) -> bool:
        """Check if the test model (llm_b) won."""
        return self.answer == "B"


@dataclass
class ModelScores:
    """Overall scores for a model."""
    model_name: str
    safe_name: str
    judge_model: str
    score: float
    wins: int
    total_matches: int
    en_score: float
    lt_score: float
    win_rate: float

    @property
    def display_name(self) -> str:
        return display_model_name(self.safe_name)


@dataclass
class ModelData:
    """Complete data for a model including scores and comparisons."""
    safe_name: str
    display_name: str
    scores: Optional[ModelScores]
    comparisons: List[ComparisonRecord]
    judge_model: str


def parse_formatted_data(formatted_data: str) -> tuple[str, str, str]:
    """Parse formatted_data field to extract source text and translations."""
    source_text = ""
    translation_a = ""
    translation_b = ""

    # Split by sections
    lines = formatted_data.split('\n')
    current_section = None
    current_content = []

    for line in lines:
        if line.startswith("## Source Text:"):
            if current_section and current_content:
                content = '\n'.join(current_content).strip()
                if current_section == "source":
                    source_text = content
                elif current_section == "translation_a":
                    translation_a = content
                elif current_section == "translation_b":
                    translation_b = content
            current_section = "source"
            current_content = []
        elif line.startswith("## Translation A"):
            if current_section == "source" and current_content:
                source_text = '\n'.join(current_content).strip()
            current_section = "translation_a"
            current_content = []
        elif line.startswith("## Translation B"):
            if current_section == "translation_a" and current_content:
                translation_a = '\n'.join(current_content).strip()
            current_section = "translation_b"
            current_content = []
        elif current_section:
            current_content.append(line)

    # Don't forget the last section
    if current_section == "translation_b" and current_content:
        translation_b = '\n'.join(current_content).strip()

    return source_text, translation_a, translation_b


def load_comparisons(comparison_file: Path) -> List[ComparisonRecord]:
    """Load comparison records from a JSONL file."""
    records: List[ComparisonRecord] = []
    if not comparison_file.exists():
        return records

    with comparison_file.open("r", encoding="utf-8") as src:
        for idx, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                # Parse the formatted_data field
                formatted_data = data.get("formatted_data", "")
                source_text, translation_a, translation_b = parse_formatted_data(formatted_data)

                # Extract answer from analysis
                answer = data.get("answer", "")
                if not answer and "analysis" in data:
                    # Try to extract from analysis
                    analysis_text = data["analysis"]
                    if "<answer>A</answer>" in analysis_text:
                        answer = "A"
                    elif "<answer>B</answer>" in analysis_text:
                        answer = "B"

                record = ComparisonRecord(
                    index=idx,
                    name=data.get("name", "unknown"),
                    english=data.get("english", False),
                    difficulty=data.get("difficulty", "unknown"),
                    comparison_id=data.get("id", str(idx)),
                    llm_a=data.get("llm_a", ""),
                    llm_b=data.get("llm_b", ""),
                    source_text=source_text,
                    translation_a=translation_a,
                    translation_b=translation_b,
                    analysis=data.get("analysis", ""),
                    answer=answer,
                    judge_model=data.get("judge_model", "unknown"),
                    raw=data,
                )
                records.append(record)
            except json.JSONDecodeError:
                continue

    return records


def load_score_summary(score_file: Path, target_model: Optional[str] = None) -> Optional[ModelScores]:
    """Load score summary from *_tl_bench_scores.jsonl file.

    Args:
        score_file: Path to the scores JSONL file
        target_model: Optional model name to look for. If None, returns first "all" difficulty entry.
    """
    if not score_file.exists():
        return None

    # If target_model is specified, extract it from the filename
    if target_model is None:
        # Extract from filename: model_tl_bench_scores.jsonl -> model
        target_model = score_file.name.replace("_tl_bench_scores.jsonl", "")

    # Normalize target model name
    target_safe = safe_name_from_model(target_model) if "/" in target_model else target_model

    with score_file.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Look for the "all" difficulty summary
                if data.get("difficulty") != "all":
                    continue

                model_name = data.get("llm", "")
                safe_name = safe_name_from_model(model_name)

                # Check if this is the model we're looking for
                if safe_name == target_safe or model_name == target_model:
                    wins = safe_int(data.get("wins")) or 0
                    total = safe_int(data.get("total_matches")) or 1

                    return ModelScores(
                        model_name=model_name,
                        safe_name=safe_name,
                        judge_model="",  # Will be set later
                        score=safe_float(data.get("score")) or 0.0,
                        wins=wins,
                        total_matches=total,
                        en_score=safe_float(data.get("EN")) or 0.0,
                        lt_score=safe_float(data.get("LT")) or 0.0,
                        win_rate=wins / total if total > 0 else 0.0,
                    )
            except json.JSONDecodeError:
                continue

    return None


def find_model_files(scores_dir: Path) -> List[tuple[str, str, Path, Path]]:
    """Find all model comparison and score files.

    Returns:
        List of (safe_name, judge_model, comparison_file, score_file) tuples
    """
    results = []

    # Find all score files first to know which models we have
    score_files_map = {}
    for score_file in sorted(scores_dir.glob("*_tl_bench_scores.jsonl")):
        safe_model = score_file.name.replace("_tl_bench_scores.jsonl", "")
        score_files_map[safe_model] = score_file

    # Find all comparison files (format: model.judge.jsonl)
    for comparison_file in sorted(scores_dir.glob("*.jsonl")):
        filename = comparison_file.name

        # Skip if it's a scores file
        if "_tl_bench_scores.jsonl" in filename:
            continue

        # Remove .jsonl extension
        base_name = filename.replace(".jsonl", "")

        # Try to match against known models from score files
        matched_model = None
        judge_model = None

        for safe_model in score_files_map.keys():
            # Check if base_name starts with safe_model followed by a dot
            if base_name.startswith(safe_model + "."):
                matched_model = safe_model
                judge_model = base_name[len(safe_model) + 1:]  # Everything after "model."
                break

        if matched_model:
            results.append((matched_model, judge_model, comparison_file, score_files_map[matched_model]))

    return results


def load_models(scores_dir: Path) -> List[ModelData]:
    """Load all models with their scores and comparisons."""
    models: Dict[str, ModelData] = {}

    model_files = find_model_files(scores_dir)

    for safe_name, judge_model, comparison_file, score_file in model_files:
        display_name = display_model_name(safe_name)

        # Load comparisons
        comparisons = load_comparisons(comparison_file)

        # Load scores (pass the safe_name to find the right model)
        scores = load_score_summary(score_file, safe_name)
        if scores:
            scores.judge_model = judge_model

        # Create or update model data
        if safe_name not in models:
            models[safe_name] = ModelData(
                safe_name=safe_name,
                display_name=display_name,
                scores=scores,
                comparisons=comparisons,
                judge_model=judge_model,
            )
        else:
            # Merge comparisons if we have multiple judge models
            models[safe_name].comparisons.extend(comparisons)

    # Sort by display name
    return sorted(models.values(), key=lambda m: m.display_name.lower())


def format_model_title(model: ModelData) -> Text:
    """Format model title with scores."""
    text = Text()
    text.append(model.display_name, style="bold cyan")

    if model.scores:
        text.append(f" • LT: {model.scores.lt_score:.2f}")
        text.append(f" • EN: {model.scores.en_score:.2f}")
        text.append(f" • Win Rate: {model.scores.win_rate*100:.1f}%")
        text.append(f" ({model.scores.wins}/{model.scores.total_matches})")

    return text


def build_comparison_renderable(record: ComparisonRecord, test_model: str) -> Group:
    """Build rich renderable for a single comparison."""
    # Determine if test model won
    test_is_b = record.llm_b == test_model or safe_name_from_model(record.llm_b) == test_model
    won = (test_is_b and record.answer == "B") or (not test_is_b and record.answer == "A")

    # Header with result
    icon = "✅" if won else "❌"
    icon_style = "green" if won else "red"

    header = Text()
    header.append(icon + " ", style=icon_style)
    header.append(record.name, style="bold")
    header.append(f" • {'EN→JA' if record.english else 'JA→EN'}", style="cyan")
    header.append(f" • {record.difficulty}", style="yellow")
    header.append(f" • Winner: {record.winner}", style="green")

    # Comparison table
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=15, style="bold", no_wrap=True)
    table.add_column(justify="center", width=1, style="dim", no_wrap=True)
    table.add_column(ratio=1)

    # Source text
    lang_label = "English" if record.english else "Japanese"
    table.add_row(f"Source ({lang_label})", "|", Text(record.source_text[:500] + ("..." if len(record.source_text) > 500 else "")))

    # Translation A
    winner_a = " ✓ WINNER" if record.answer == "A" else ""
    style_a = "green" if record.answer == "A" else "dim"
    table.add_row(
        f"Translation A{winner_a}",
        "|",
        Text(f"[{record.llm_a}]\n{record.translation_a[:500] + ('...' if len(record.translation_a) > 500 else '')}", style=style_a)
    )

    # Translation B
    winner_b = " ✓ WINNER" if record.answer == "B" else ""
    style_b = "green" if record.answer == "B" else "dim"
    table.add_row(
        f"Translation B{winner_b}",
        "|",
        Text(f"[{record.llm_b}]\n{record.translation_b[:500] + ('...' if len(record.translation_b) > 500 else '')}", style=style_b)
    )

    # Analysis (truncated)
    analysis_preview = record.analysis[:1000] + ("..." if len(record.analysis) > 1000 else "")
    table.add_row("Judge Analysis", "|", Text(analysis_preview, style="italic"))

    return Group(header, table)


def build_comparisons_renderable(records: Sequence[ComparisonRecord], test_model: str) -> Group:
    """Build rich renderable for all comparisons."""
    if not records:
        return Group(Text("No comparisons found."))

    renderables: List[Any] = []

    for idx, record in enumerate(records):
        renderables.append(build_comparison_renderable(record, test_model))
        if idx != len(records) - 1:
            renderables.append(Text(""))

    return Group(*renderables)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive viewer for translation benchmark judgments.")
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR, help="Path to scores directory")
    parser.add_argument("--model", help="Model name to preselect (safe or display name)")
    args = parser.parse_args(argv)

    scores_dir = args.scores_dir

    if App is None or TEXTUAL_IMPORT_ERROR is not None:
        print("This viewer requires the 'textual' package. Install it with `pip install textual`.")
        if TEXTUAL_IMPORT_ERROR is not None:
            print(f"Import error: {TEXTUAL_IMPORT_ERROR}")
        return 1

    models = load_models(scores_dir)

    if not models:
        print(f"No models found in {scores_dir}")
        return 1

    app = TranslationBenchViewerApp(
        models=models,
        scores_dir=scores_dir,
        preselect_model=args.model,
    )
    app.run()
    return 0


if App is not None:

    class ModelListItem(ListItem):
        """List item for a model."""
        def __init__(self, model: ModelData):
            super().__init__(Static(format_model_title(model)))
            self.model = model


    class TranslationBenchViewerApp(App):
        """Main TUI application."""

        CSS = """
        Screen {
            layout: vertical;
        }
        #body {
            layout: horizontal;
            height: 1fr;
        }
        #sidebar {
            width: 60;
            min-width: 50;
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 1 0;
        }
        #model-list {
            height: 1fr;
            margin: 0 1 0 1;
            overflow: auto;
        }
        #main {
            layout: vertical;
            width: 1fr;
            height: 1fr;
            padding: 1;
        }
        #content-summary {
            min-height: 1;
            margin-bottom: 1;
        }
        #details-panel {
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("w", "toggle_wins", "Toggle wins only"),
            ("l", "toggle_losses", "Toggle losses only"),
        ]

        show_wins_only = reactive(False)
        show_losses_only = reactive(False)

        def __init__(
            self,
            models: List[ModelData],
            scores_dir: Path,
            preselect_model: Optional[str] = None,
        ) -> None:
            super().__init__()
            self.models = models
            self.scores_dir = scores_dir
            self.preselect_model = preselect_model
            self.selected_model: Optional[ModelData] = None

        def compose(self) -> ComposeResult:
            """Compose the UI."""
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                with Vertical(id="sidebar"):
                    yield ListView(id="model-list")
                with Vertical(id="main"):
                    yield Static("", id="content-summary")
                    with VerticalScroll(id="details-panel"):
                        yield Static("Select a model to view comparisons.", id="details-content")
            yield Footer()

        def on_mount(self) -> None:
            """Handle mount event."""
            self.title = "Translation Benchmark Viewer"
            self._populate_models()

        def action_toggle_wins(self) -> None:
            """Toggle showing only wins."""
            self.show_wins_only = not self.show_wins_only
            if self.show_wins_only:
                self.show_losses_only = False

        def action_toggle_losses(self) -> None:
            """Toggle showing only losses."""
            self.show_losses_only = not self.show_losses_only
            if self.show_losses_only:
                self.show_wins_only = False

        def watch_show_wins_only(self, _: bool) -> None:
            """React to wins filter change."""
            self._refresh_content()

        def watch_show_losses_only(self, _: bool) -> None:
            """React to losses filter change."""
            self._refresh_content()

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            """Handle model selection."""
            if isinstance(event.item, ModelListItem):
                self._select_model(event.item.model)

        def _populate_models(self) -> None:
            """Populate the model list."""
            model_list = self.query_one("#model-list", ListView)
            model_list.clear()

            if not self.models:
                model_list.append(ListItem(Static("No models found.")))
                return

            for model in self.models:
                model_list.append(ModelListItem(model))

            # Auto-select first model or preselected model
            index = self._find_model_index(self.preselect_model)
            index = max(0, min(index, len(self.models) - 1))
            model_list.index = index
            self._select_model(self.models[index])

        def _find_model_index(self, target: Optional[str]) -> int:
            """Find model index by name."""
            if not target:
                return 0
            target_lower = target.lower()
            for idx, model in enumerate(self.models):
                if model.safe_name.lower() == target_lower or model.display_name.lower() == target_lower:
                    return idx
            return 0

        def _select_model(self, model: Optional[ModelData]) -> None:
            """Select a model and display its comparisons."""
            self.selected_model = model
            self._refresh_content()

        def _refresh_content(self) -> None:
            """Refresh the content panel."""
            summary = self.query_one("#content-summary", Static)
            details = self.query_one("#details-content", Static)

            if not self.selected_model:
                summary.update("Select a model to view comparisons.")
                details.update(Text("Select a model to display comparisons."))
                return

            model = self.selected_model
            records = model.comparisons

            # Apply filters
            if self.show_wins_only:
                records = [r for r in records if r.is_test_model_win]
            elif self.show_losses_only:
                records = [r for r in records if not r.is_test_model_win]

            # Update summary
            wins = sum(1 for r in model.comparisons if r.is_test_model_win)
            total = len(model.comparisons)

            summary_text = f"Showing {len(records)} of {total} comparisons"
            if self.show_wins_only:
                summary_text += " (wins only)"
            elif self.show_losses_only:
                summary_text += " (losses only)"
            summary_text += f" • Wins: {wins}/{total} ({wins/total*100:.1f}%)" if total > 0 else ""

            if model.scores:
                summary_text += f" • LT: {model.scores.lt_score:.2f} • EN: {model.scores.en_score:.2f}"

            summary.update(summary_text)

            # Update details
            if not records:
                details.update(Text("No comparisons match the current filter."))
                return

            details.update(build_comparisons_renderable(records, model.safe_name))


if __name__ == "__main__":
    sys.exit(main())
