#!/usr/bin/env python3
"""Fast Interactive TUI for browsing translations and comparisons with lazy loading."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Group
from rich.table import Table
from rich.text import Text

BASE_DIR = Path(__file__).parent
SCORES_DIR = BASE_DIR / "scores"
BASESET_DIR = BASE_DIR / "baseset"
TRANSLATIONS_DIR = BASE_DIR / "translations"

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, ListItem, ListView, Select, Static
except ImportError as exc:
    App = None
    TEXTUAL_IMPORT_ERROR = exc
else:
    TEXTUAL_IMPORT_ERROR = None


def display_model_name(safe_name: str) -> str:
    """Convert safe model name back to display format."""
    return safe_name.replace("__", "/")


def safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ===== Translation Data Structures =====

@dataclass
class TranslationMetadata:
    """Lightweight translation metadata for list view."""
    index: int
    line_num: int
    name: str
    english: bool
    difficulty: str
    source_preview: str  # First 100 chars of source


@dataclass
class TranslationDetail:
    """Full translation details loaded on-demand."""
    metadata: TranslationMetadata
    source_text: str
    translation: str
    full_response: str
    prompt: str
    temperature: Optional[float]
    generation_config: Dict[str, Any]


# ===== Comparison Data Structures =====

@dataclass
class ComparisonMetadata:
    """Lightweight comparison metadata for list view."""
    index: int
    line_num: int
    name: str
    english: bool
    difficulty: str
    comparison_id: str
    llm_a: str
    llm_b: str
    answer: str

    @property
    def winner(self) -> str:
        return self.llm_a if self.answer == "A" else self.llm_b

    @property
    def is_test_model_win(self) -> bool:
        return self.answer == "B"


@dataclass
class ComparisonDetail:
    """Full comparison details loaded on-demand."""
    metadata: ComparisonMetadata
    source_text: str
    translation_a: str
    translation_b: str
    analysis: str


@dataclass
class ModelScores:
    """Overall scores for a model."""
    model_name: str
    safe_name: str
    score: float
    wins: int
    total_matches: int
    en_score: float
    lt_score: float
    win_rate: float


@dataclass
class ModelData:
    """Model with scores and metadata."""
    safe_name: str
    display_name: str
    scores: Optional[ModelScores]
    category: str
    # For comparisons mode
    comparison_file: Optional[Path]
    comparisons_metadata: List[ComparisonMetadata]
    # For translations mode
    translation_file: Optional[Path]
    translations_metadata: List[TranslationMetadata]


# ===== Translation Loading Functions =====

def load_translation_metadata(translation_file: Path) -> List[TranslationMetadata]:
    """Load only lightweight metadata from translations file."""
    metadata_list: List[TranslationMetadata] = []

    if not translation_file.exists():
        return metadata_list

    with translation_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                source_text = data.get("source_text", "")
                source_preview = source_text[:100] if len(source_text) > 100 else source_text

                metadata = TranslationMetadata(
                    index=idx,
                    line_num=idx,
                    name=data.get("name", "unknown"),
                    english=data.get("english", False),
                    difficulty=data.get("difficulty", "unknown"),
                    source_preview=source_preview,
                )
                metadata_list.append(metadata)
            except json.JSONDecodeError:
                continue

    return metadata_list


def load_translation_detail(translation_file: Path, metadata: TranslationMetadata) -> Optional[TranslationDetail]:
    """Lazy-load full translation details for a specific record."""
    if not translation_file.exists():
        return None

    try:
        with translation_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == metadata.line_num:
                    data = json.loads(line)

                    return TranslationDetail(
                        metadata=metadata,
                        source_text=data.get("source_text", ""),
                        translation=data.get("translation", ""),
                        full_response=data.get("full_response", ""),
                        prompt=data.get("prompt", ""),
                        temperature=safe_float(data.get("temperature")),
                        generation_config=data.get("generation_config", {}),
                    )

    except (json.JSONDecodeError, IOError):
        return None

    return None


# ===== Comparison Loading Functions (from previous version) =====

def parse_formatted_data(formatted_data: str) -> Tuple[str, str, str]:
    """Parse formatted_data field to extract source text and translations."""
    source_text = ""
    translation_a = ""
    translation_b = ""

    lines = formatted_data.split('\n')
    current_section = None
    current_content = []

    def save_section():
        nonlocal source_text, translation_a, translation_b
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if current_section == "source":
                source_text = content
            elif current_section == "translation_a":
                translation_a = content
            elif current_section == "translation_b":
                translation_b = content

    for line in lines:
        if line.startswith("## Source Text:"):
            save_section()
            current_section = "source"
            current_content = []
        elif line.startswith("## Translation A"):
            save_section()
            current_section = "translation_a"
            current_content = []
        elif line.startswith("## Translation B"):
            save_section()
            current_section = "translation_b"
            current_content = []
        elif line.startswith("##"):
            save_section()
            current_section = None
            current_content = []
        elif current_section:
            current_content.append(line)

    save_section()
    return source_text, translation_a, translation_b


def load_comparison_metadata(comparison_file: Path) -> List[ComparisonMetadata]:
    """Load only lightweight metadata from comparisons file."""
    metadata_list: List[ComparisonMetadata] = []

    if not comparison_file.exists():
        return metadata_list

    with comparison_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                answer = data.get("answer", "")
                if not answer and "analysis" in data:
                    analysis_text = data["analysis"]
                    if "<answer>A</answer>" in analysis_text:
                        answer = "A"
                    elif "<answer>B</answer>" in analysis_text:
                        answer = "B"

                metadata = ComparisonMetadata(
                    index=idx,
                    line_num=idx,
                    name=data.get("name", "unknown"),
                    english=data.get("english", False),
                    difficulty=data.get("difficulty", "unknown"),
                    comparison_id=data.get("id", str(idx)),
                    llm_a=data.get("llm_a", ""),
                    llm_b=data.get("llm_b", ""),
                    answer=answer,
                )
                metadata_list.append(metadata)
            except json.JSONDecodeError:
                continue

    return metadata_list


def load_comparison_detail(comparison_file: Path, metadata: ComparisonMetadata) -> Optional[ComparisonDetail]:
    """Lazy-load full comparison details for a specific record."""
    if not comparison_file.exists():
        return None

    try:
        with comparison_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == metadata.line_num:
                    data = json.loads(line)
                    formatted_data = data.get("formatted_data", "")
                    source_text, translation_a, translation_b = parse_formatted_data(formatted_data)

                    return ComparisonDetail(
                        metadata=metadata,
                        source_text=source_text,
                        translation_a=translation_a,
                        translation_b=translation_b,
                        analysis=data.get("analysis", ""),
                    )

    except (json.JSONDecodeError, IOError):
        return None

    return None


# ===== Score Loading =====

def load_score_summary(score_file: Path, target_model: Optional[str] = None) -> Optional[ModelScores]:
    """Load score summary from *_tl_bench_scores.jsonl file."""
    if not score_file.exists():
        return None

    if target_model is None:
        target_model = score_file.name.replace("_tl_bench_scores.jsonl", "")

    target_safe = target_model.replace("/", "__") if "/" in target_model else target_model

    with score_file.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("difficulty") != "all":
                    continue

                model_name = data.get("llm", "")
                safe_name = model_name.replace("/", "__")

                if safe_name == target_safe or model_name == target_model:
                    wins = int(data.get("wins", 0))
                    total = int(data.get("total_matches", 1))

                    return ModelScores(
                        model_name=model_name,
                        safe_name=safe_name,
                        score=safe_float(data.get("score")) or 0.0,
                        wins=wins,
                        total_matches=total,
                        en_score=safe_float(data.get("EN")) or 0.0,
                        lt_score=safe_float(data.get("LT")) or 0.0,
                        win_rate=wins / total if total > 0 else 0.0,
                    )
            except (json.JSONDecodeError, ValueError):
                continue

    return None


# ===== Model Loading Functions =====

def load_test_models(scores_dir: Path, translations_dir: Path) -> List[ModelData]:
    """Load test models from scores/ and translations/ directories."""
    models: Dict[str, ModelData] = {}

    # Find score files
    score_files_map = {}
    for score_file in sorted(scores_dir.glob("*_tl_bench_scores.jsonl")):
        safe_model = score_file.name.replace("_tl_bench_scores.jsonl", "")
        score_files_map[safe_model] = score_file

    # Find translation files
    translation_files_map = {}
    for trans_file in sorted(translations_dir.glob("*.jsonl")):
        safe_model = trans_file.name.replace(".jsonl", "")
        translation_files_map[safe_model] = trans_file

    # Find comparison files
    for comparison_file in sorted(scores_dir.glob("*.jsonl")):
        if "_tl_bench_scores.jsonl" in comparison_file.name:
            continue

        base_name = comparison_file.name.replace(".jsonl", "")

        matched_model = None
        for safe_model in score_files_map.keys():
            if base_name.startswith(safe_model + "."):
                matched_model = safe_model
                break

        if not matched_model:
            continue

        display_name = display_model_name(matched_model)

        # Load comparison metadata
        comparisons_metadata = load_comparison_metadata(comparison_file)

        # Load translation metadata if available
        translation_file = translation_files_map.get(matched_model)
        translations_metadata = []
        if translation_file:
            translations_metadata = load_translation_metadata(translation_file)

        # Load scores
        scores = load_score_summary(score_files_map[matched_model], matched_model)

        if matched_model not in models:
            models[matched_model] = ModelData(
                safe_name=matched_model,
                display_name=display_name,
                scores=scores,
                category="test_models",
                comparison_file=comparison_file,
                comparisons_metadata=comparisons_metadata,
                translation_file=translation_file,
                translations_metadata=translations_metadata,
            )
        else:
            models[matched_model].comparisons_metadata.extend(comparisons_metadata)

    return sorted(models.values(), key=lambda m: m.display_name.lower())


def load_baseset_models(baseset_dir: Path, version: str) -> List[ModelData]:
    """Load base set models from baseset/vX.X/ directory."""
    version_dir = baseset_dir / version
    if not version_dir.exists():
        return []

    models: List[ModelData] = []

    # Find base_set comparison files
    for comparison_file in sorted(version_dir.glob("base_set.*.jsonl")):
        judge_name = comparison_file.name.replace("base_set.", "").replace(".jsonl", "")
        comparisons_metadata = load_comparison_metadata(comparison_file)

        if not comparisons_metadata:
            continue

        display_name = f"Base Set {version} ({judge_name})"

        models.append(ModelData(
            safe_name=f"baseset_{version}_{judge_name}",
            display_name=display_name,
            scores=None,
            category=f"baseset_{version}",
            comparison_file=comparison_file,
            comparisons_metadata=comparisons_metadata,
            translation_file=None,
            translations_metadata=[],
        ))

    # Load individual base set translations
    translations_dir = version_dir / "translations"
    if translations_dir.exists():
        for trans_file in sorted(translations_dir.glob("*.jsonl")):
            safe_model = trans_file.name.replace(".jsonl", "")
            display_name = display_model_name(safe_model)

            translations_metadata = load_translation_metadata(trans_file)

            if not translations_metadata:
                continue

            models.append(ModelData(
                safe_name=f"baseset_{version}_{safe_model}",
                display_name=f"{display_name} [{version}]",
                scores=None,
                category=f"baseset_{version}",
                comparison_file=None,
                comparisons_metadata=[],
                translation_file=trans_file,
                translations_metadata=translations_metadata,
            ))

    return models


# ===== Rendering Functions =====

def format_model_title(model: ModelData, view_mode: str) -> Text:
    """Format model title with scores."""
    text = Text()
    text.append(model.display_name, style="bold cyan")

    if model.scores:
        text.append(f" • LT: {model.scores.lt_score:.2f}")
        text.append(f" • EN: {model.scores.en_score:.2f}")
        text.append(f" • WR: {model.scores.win_rate*100:.1f}%")

    if view_mode == "translations":
        count = len(model.translations_metadata)
        text.append(f" • {count} translations", style="dim")
    else:
        count = len(model.comparisons_metadata)
        text.append(f" • {count} comparisons", style="dim")

    return text


def build_translation_renderable(detail: TranslationDetail) -> Group:
    """Build rich renderable for a single translation."""
    meta = detail.metadata

    # Header
    header = Text()
    header.append(meta.name, style="bold")
    header.append(f" • {'EN→JA' if meta.english else 'JA→EN'}", style="cyan")
    header.append(f" • {meta.difficulty}", style="yellow")

    # Translation table
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=15, style="bold", no_wrap=True)
    table.add_column(justify="center", width=1, style="dim", no_wrap=True)
    table.add_column(ratio=1)

    # Source text
    lang_label = "English" if meta.english else "Japanese"
    source_preview = detail.source_text[:800] + ("..." if len(detail.source_text) > 800 else "")
    table.add_row(f"Source ({lang_label})", "|", Text(source_preview))

    # Translation
    translation_preview = detail.translation[:800] + ("..." if len(detail.translation) > 800 else "")
    table.add_row("Translation", "|", Text(translation_preview, style="cyan"))

    # Generation settings
    settings = []
    if detail.temperature is not None:
        settings.append(f"temp={detail.temperature}")
    if detail.generation_config:
        for k, v in list(detail.generation_config.items())[:3]:
            if k not in ["model"]:
                settings.append(f"{k}={v}")
    if settings:
        table.add_row("Settings", "|", Text(", ".join(settings), style="dim"))

    # Full response preview (if different from translation)
    if detail.full_response and detail.full_response != detail.translation:
        response_preview = detail.full_response[:500] + ("..." if len(detail.full_response) > 500 else "")
        table.add_row("Full Response", "|", Text(response_preview, style="italic dim"))

    return Group(header, table)


def build_comparison_renderable(detail: ComparisonDetail, test_model: str) -> Group:
    """Build rich renderable for a single comparison."""
    meta = detail.metadata

    test_is_b = meta.llm_b == test_model or meta.llm_b.replace("/", "__") == test_model
    won = (test_is_b and meta.answer == "B") or (not test_is_b and meta.answer == "A")

    icon = "✅" if won else "❌"
    icon_style = "green" if won else "red"

    header = Text()
    header.append(icon + " ", style=icon_style)
    header.append(meta.name, style="bold")
    header.append(f" • {'EN→JA' if meta.english else 'JA→EN'}", style="cyan")
    header.append(f" • {meta.difficulty}", style="yellow")
    header.append(f" • Winner: {meta.winner}", style="green")

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=15, style="bold", no_wrap=True)
    table.add_column(justify="center", width=1, style="dim", no_wrap=True)
    table.add_column(ratio=1)

    lang_label = "English" if meta.english else "Japanese"
    source_preview = detail.source_text[:500] + ("..." if len(detail.source_text) > 500 else "")
    table.add_row(f"Source ({lang_label})", "|", Text(source_preview))

    winner_a = " ✓ WINNER" if meta.answer == "A" else ""
    style_a = "green" if meta.answer == "A" else "dim"
    trans_a_preview = detail.translation_a[:500] + ("..." if len(detail.translation_a) > 500 else "")
    table.add_row(
        f"Translation A{winner_a}",
        "|",
        Text(f"[{meta.llm_a}]\n{trans_a_preview}", style=style_a)
    )

    winner_b = " ✓ WINNER" if meta.answer == "B" else ""
    style_b = "green" if meta.answer == "B" else "dim"
    trans_b_preview = detail.translation_b[:500] + ("..." if len(detail.translation_b) > 500 else "")
    table.add_row(
        f"Translation B{winner_b}",
        "|",
        Text(f"[{meta.llm_b}]\n{trans_b_preview}", style=style_b)
    )

    analysis_preview = detail.analysis[:1000] + ("..." if len(detail.analysis) > 1000 else "")
    table.add_row("Judge Analysis", "|", Text(analysis_preview, style="italic"))

    return Group(header, table)


# ===== Main Application =====

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast interactive viewer for translations and comparisons.")
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR, help="Path to scores directory")
    parser.add_argument("--baseset-dir", type=Path, default=BASESET_DIR, help="Path to baseset directory")
    parser.add_argument("--translations-dir", type=Path, default=TRANSLATIONS_DIR, help="Path to translations directory")
    args = parser.parse_args(argv)

    if App is None or TEXTUAL_IMPORT_ERROR is not None:
        print("This viewer requires the 'textual' package. Install it with `pip install textual`.")
        if TEXTUAL_IMPORT_ERROR is not None:
            print(f"Import error: {TEXTUAL_IMPORT_ERROR}")
        return 1

    print("Loading models...")
    test_models = load_test_models(args.scores_dir, args.translations_dir)
    baseset_v10_models = load_baseset_models(args.baseset_dir, "v1.0")
    baseset_v09_models = load_baseset_models(args.baseset_dir, "v0.9")

    all_models = test_models + baseset_v10_models + baseset_v09_models

    if not all_models:
        print(f"No models found")
        return 1

    trans_count = sum(len(m.translations_metadata) for m in all_models)
    comp_count = sum(len(m.comparisons_metadata) for m in all_models)
    print(f"Loaded {len(test_models)} test models, {len(baseset_v10_models)} v1.0 models, {len(baseset_v09_models)} v0.9 models")
    print(f"Total: {trans_count} translations, {comp_count} comparisons")

    app = TranslationBenchViewerApp(
        all_models=all_models,
        test_models=test_models,
        baseset_v10_models=baseset_v10_models,
        baseset_v09_models=baseset_v09_models,
    )
    app.run()
    return 0


if App is not None:

    class ModelListItem(ListItem):
        """List item for a model."""
        def __init__(self, model: ModelData, view_mode: str):
            super().__init__(Static(format_model_title(model, view_mode)))
            self.model = model


    class TranslationBenchViewerApp(App):
        """Main TUI application with lazy loading."""

        CSS = """
        Screen {
            layout: vertical;
        }
        #body {
            layout: horizontal;
            height: 1fr;
        }
        #sidebar {
            width: 70;
            min-width: 50;
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 1 0;
        }
        #view-mode-select {
            margin: 0 1 0 1;
        }
        #category-select {
            margin: 0 1 1 1;
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
            ("w", "toggle_wins", "Toggle wins"),
            ("l", "toggle_losses", "Toggle losses"),
            ("n", "next_item", "Next"),
            ("p", "prev_item", "Previous"),
            ("v", "toggle_view", "Switch view"),
        ]

        show_wins_only = reactive(False)
        show_losses_only = reactive(False)
        current_index = reactive(0)
        view_mode = reactive("comparisons")  # "comparisons" or "translations"

        def __init__(
            self,
            all_models: List[ModelData],
            test_models: List[ModelData],
            baseset_v10_models: List[ModelData],
            baseset_v09_models: List[ModelData],
        ) -> None:
            super().__init__()
            self.all_models = all_models
            self.test_models = test_models
            self.baseset_v10_models = baseset_v10_models
            self.baseset_v09_models = baseset_v09_models
            self.current_category = "test_models"
            self.selected_model: Optional[ModelData] = None

        def compose(self) -> ComposeResult:
            """Compose the UI."""
            view_mode_options = [
                ("Comparisons (A vs B)", "comparisons"),
                ("Translations (Individual)", "translations"),
            ]

            category_options = [
                ("Test Models", "test_models"),
                ("Base Set v1.0", "baseset_v1.0"),
                ("Base Set v0.9", "baseset_v0.9"),
            ]

            yield Header(show_clock=True)
            with Horizontal(id="body"):
                with Vertical(id="sidebar"):
                    yield Select(options=view_mode_options, value="comparisons", id="view-mode-select")
                    yield Select(options=category_options, value="test_models", id="category-select")
                    yield ListView(id="model-list")
                with Vertical(id="main"):
                    yield Static("", id="content-summary")
                    with VerticalScroll(id="details-panel"):
                        yield Static("Select a model to view data.", id="details-content")
            yield Footer()

        def on_mount(self) -> None:
            """Handle mount event."""
            self.title = "Translation Benchmark Viewer"
            self._populate_models()

        def action_toggle_wins(self) -> None:
            """Toggle showing only wins."""
            if self.view_mode == "comparisons":
                self.show_wins_only = not self.show_wins_only
                if self.show_wins_only:
                    self.show_losses_only = False

        def action_toggle_losses(self) -> None:
            """Toggle showing only losses."""
            if self.view_mode == "comparisons":
                self.show_losses_only = not self.show_losses_only
                if self.show_losses_only:
                    self.show_wins_only = False

        def action_next_item(self) -> None:
            """Navigate to next item."""
            if self.selected_model:
                filtered = self._get_filtered_items()
                if filtered and self.current_index < len(filtered) - 1:
                    self.current_index += 1
                    self._refresh_content()

        def action_prev_item(self) -> None:
            """Navigate to previous item."""
            if self.current_index > 0:
                self.current_index -= 1
                self._refresh_content()

        def action_toggle_view(self) -> None:
            """Toggle between comparisons and translations view."""
            self.view_mode = "translations" if self.view_mode == "comparisons" else "comparisons"
            self.current_index = 0
            self._populate_models()

        def watch_view_mode(self, _: str) -> None:
            """React to view mode change."""
            view_select = self.query_one("#view-mode-select", Select)
            view_select.value = self.view_mode
            self._populate_models()

        def watch_show_wins_only(self, _: bool) -> None:
            """React to wins filter change."""
            self.current_index = 0
            self._refresh_content()

        def watch_show_losses_only(self, _: bool) -> None:
            """React to losses filter change."""
            self.current_index = 0
            self._refresh_content()

        def on_select_changed(self, event: Select.Changed) -> None:
            """Handle selection changes."""
            if event.select.id == "category-select" and event.value:
                self._change_category(str(event.value))
            elif event.select.id == "view-mode-select" and event.value:
                self.view_mode = str(event.value)

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            """Handle model selection."""
            if isinstance(event.item, ModelListItem):
                self._select_model(event.item.model)

        def _change_category(self, category: str) -> None:
            """Change the category filter."""
            if category == self.current_category:
                return

            self.current_category = category
            self.selected_model = None
            self.current_index = 0
            self._populate_models()

        def _get_models_for_category(self) -> List[ModelData]:
            """Get models for the current category."""
            if self.current_category == "test_models":
                return self.test_models
            elif self.current_category == "baseset_v1.0":
                return self.baseset_v10_models
            elif self.current_category == "baseset_v0.9":
                return self.baseset_v09_models
            return []

        def _populate_models(self) -> None:
            """Populate the model list for current category and view mode."""
            model_list = self.query_one("#model-list", ListView)
            model_list.clear()

            models = self._get_models_for_category()

            # Filter models based on view mode
            if self.view_mode == "translations":
                models = [m for m in models if m.translations_metadata]
            else:
                models = [m for m in models if m.comparisons_metadata]

            if not models:
                model_list.append(ListItem(Static(f"No {self.view_mode} data for {self.current_category}")))
                return

            for model in models:
                model_list.append(ModelListItem(model, self.view_mode))

            model_list.index = 0
            self._select_model(models[0])

        def _select_model(self, model: Optional[ModelData]) -> None:
            """Select a model and display its data."""
            self.selected_model = model
            self.current_index = 0
            self._refresh_content()

        def _get_filtered_items(self):
            """Get filtered items based on current view mode."""
            if not self.selected_model:
                return []

            if self.view_mode == "translations":
                return self.selected_model.translations_metadata
            else:
                comparisons = self.selected_model.comparisons_metadata
                if self.show_wins_only:
                    comparisons = [c for c in comparisons if c.is_test_model_win]
                elif self.show_losses_only:
                    comparisons = [c for c in comparisons if not c.is_test_model_win]
                return comparisons

        def _refresh_content(self) -> None:
            """Refresh the content panel."""
            summary = self.query_one("#content-summary", Static)
            details = self.query_one("#details-content", Static)

            if not self.selected_model:
                summary.update("Select a model to view data.")
                details.update(Text("Select a model to display data."))
                return

            model = self.selected_model
            filtered_items = self._get_filtered_items()

            if not filtered_items:
                summary.update(f"No {self.view_mode} match the current filter")
                details.update(Text(f"No {self.view_mode} match the current filter."))
                return

            # Update summary
            if self.view_mode == "translations":
                total = len(model.translations_metadata)
                summary_text = f"Showing translation {self.current_index + 1}/{len(filtered_items)}"
                summary_text += f" • Total: {total} translations"
            else:
                wins = sum(1 for c in model.comparisons_metadata if c.is_test_model_win)
                total = len(model.comparisons_metadata)
                summary_text = f"Showing comparison {self.current_index + 1}/{len(filtered_items)}"
                if self.show_wins_only:
                    summary_text += " (wins only)"
                elif self.show_losses_only:
                    summary_text += " (losses only)"
                summary_text += f" • Total: {len(filtered_items)}/{total}"

            if model.scores:
                summary_text += f" • LT: {model.scores.lt_score:.2f} • EN: {model.scores.en_score:.2f}"

            summary.update(summary_text)

            # Lazy-load the current item detail
            current_meta = filtered_items[self.current_index]

            if self.view_mode == "translations":
                detail = load_translation_detail(model.translation_file, current_meta)
                if detail:
                    details.update(build_translation_renderable(detail))
                else:
                    details.update(Text(f"Failed to load translation details for {current_meta.name}"))
            else:
                detail = load_comparison_detail(model.comparison_file, current_meta)
                if detail:
                    details.update(build_comparison_renderable(detail, model.safe_name))
                else:
                    details.update(Text(f"Failed to load comparison details for {current_meta.name}"))


if __name__ == "__main__":
    sys.exit(main())
