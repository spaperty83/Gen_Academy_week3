"""MOCKED: export commit boundaries, repair, idempotency and confinement."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import export


@pytest.fixture
def destination(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    monkeypatch.setattr(export, "OUTPUTS_DIR", root)
    return root / "Lisbon_abcdefgh_v2"


@pytest.fixture
def itinerary():
    return {
        "city": "Lisbon", "days": [{"day": 1, "stops": [{
            "name": "MAAT", "note": "Architecture by the river.",
            "source_url": "https://example.org/maat",
        }]}],
    }


def filename(base, suffix):
    return Path(str(base) + suffix)


def test_partial_write_has_no_manifest_and_retry_repairs(destination, itinerary, monkeypatch):
    original = export.os.replace
    def fail_markdown(source, target):
        if target.suffix == ".md":
            raise OSError("injected disk failure")
        return original(source, target)
    with monkeypatch.context() as patcher:
        patcher.setattr(export.os, "replace", fail_markdown)
        with pytest.raises(export.ExportError):
            export.write_export(itinerary, destination)
    assert filename(destination, ".json").exists()
    assert not filename(destination, ".manifest.json").exists()
    assert not export.export_is_complete(destination)
    export.write_export(itinerary, destination)
    assert export.export_is_complete(destination)


def test_double_export_writes_once(destination, itinerary, monkeypatch):
    writer = Mock(wraps=export.os.replace)
    monkeypatch.setattr(export.os, "replace", writer)
    first = export.write_export(itinerary, destination)
    second = export.write_export(itinerary, destination)
    assert first == second
    assert writer.call_count == 3  # JSON, Markdown, then manifest, once each.


def test_json_equals_approved_itinerary(destination, itinerary):
    result = export.write_export(itinerary, destination)
    assert json.loads(filename(destination, ".json").read_text("utf-8")) == itinerary
    assert result["manifest"]["draft_version"] == 2
    assert Path(result["path"]).is_file()
    markdown = filename(destination, ".md").read_text("utf-8")
    assert "MAAT" in markdown and "Architecture by the river." in markdown
    assert "https://example.org/maat" in markdown and export.DISCLAIMER in markdown


@pytest.mark.parametrize("suffix", [".json", ".md", ".manifest.json"])
def test_corruption_or_missing_file_is_repaired(suffix, destination, itinerary):
    export.write_export(itinerary, destination)
    filename(destination, suffix).write_text("corrupted", encoding="utf-8")
    assert not export.export_is_complete(destination)
    export.write_export(itinerary, destination)
    assert export.export_is_complete(destination)


@pytest.mark.parametrize("suffix", [".json", ".md"])
def test_lone_file_is_incomplete(suffix, destination):
    destination.parent.mkdir()
    filename(destination, suffix).write_text("{}")
    assert not export.export_is_complete(destination)


def test_same_base_different_itinerary_replaces_export(destination, itinerary):
    first = export.write_export(itinerary, destination)
    changed = itinerary | {"city": "Porto"}
    second = export.write_export(changed, destination)
    assert first["manifest"]["itinerary_hash"] != second["manifest"]["itinerary_hash"]
    assert json.loads(filename(destination, ".json").read_text()) == changed
    assert export.export_is_complete(destination)


def test_repair_removes_old_manifest_before_writing(destination, itinerary, monkeypatch):
    export.write_export(itinerary, destination)
    monkeypatch.setattr(export, "_atomic_write", Mock(side_effect=OSError("disk failed")))
    with pytest.raises(export.ExportError):
        export.write_export(itinerary | {"city": "Porto"}, destination)
    assert not filename(destination, ".manifest.json").exists()


@pytest.mark.parametrize("base", [
    "../escape_v2", "outputs/../../escape_v2", "outside/escape_v2",
])
def test_relative_path_escape_rejected(base, destination, itinerary):
    with pytest.raises(export.ExportError, match="inside outputs"):
        export.write_export(itinerary, base)


def test_absolute_path_escape_rejected(destination, itinerary):
    with pytest.raises(export.ExportError, match="inside outputs"):
        export.write_export(itinerary, destination.parent.parent / "escape_v2")


def test_export_complete_also_rejects_path_escape(destination):
    with pytest.raises(export.ExportError):
        export.export_is_complete(destination.parent.parent / "escape_v2")


def test_symlink_escape_rejected(destination, itinerary, tmp_path):
    destination.parent.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = destination.parent / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode or privilege")
    with pytest.raises(export.ExportError, match="inside outputs"):
        export.write_export(itinerary, link / "escape_v2")


def test_temp_file_cleaned_on_rename_failure(destination, itinerary, monkeypatch):
    monkeypatch.setattr(export.os, "replace", Mock(side_effect=OSError("rename failed")))
    with pytest.raises(export.ExportError):
        export.write_export(itinerary, destination)
    assert list(destination.parent.iterdir()) == []


def test_both_files_staged_before_first_rename(destination, itinerary, monkeypatch):
    original = export.os.replace
    def check_staging(source, target):
        if target.suffix == ".json" and not target.name.endswith(".manifest.json"):
            assert len(list(destination.parent.glob("*.tmp"))) == 2
        return original(source, target)
    monkeypatch.setattr(export.os, "replace", check_staging)
    export.write_export(itinerary, destination)
    assert export.export_is_complete(destination)


def test_failed_file_staging_leaves_no_manifest(destination, itinerary, monkeypatch):
    original = export._stage_file
    def fail_markdown(target, content):
        if target.suffix == ".md":
            raise OSError("staging failed")
        return original(target, content)
    monkeypatch.setattr(export, "_stage_file", fail_markdown)
    with pytest.raises(export.ExportError):
        export.write_export(itinerary, destination)
    assert not export.export_is_complete(destination)
    assert list(destination.parent.iterdir()) == []


@pytest.mark.parametrize("as_model", [True, False])
def test_exports_new_itinerary_schema(destination, as_model):
    from src.state import Itinerary

    itinerary = Itinerary(
        city="Lisbon", budget_tips=["Use transit.", "Consider free hours."],
        days=[{
            "day_number": 2, "theme": "Art", "stops": [{
                "time_slot": "afternoon", "place_name": "MAAT",
                "note": "Art by the river.", "source_url": "https://example.org/maat",
            }],
        }],
    )
    approved = itinerary.model_dump(mode="json")
    export.write_export(itinerary if as_model else approved, destination)
    assert json.loads(filename(destination, ".json").read_text("utf-8")) == approved
    markdown = filename(destination, ".md").read_text("utf-8")
    assert "## Day 2" in markdown
    assert "MAAT: Art by the river." in markdown
    assert "https://example.org/maat" in markdown
    assert export.export_is_complete(destination)


def test_manifest_version_must_match_export_base(destination, itinerary):
    export.write_export(itinerary, destination)
    path = filename(destination, ".manifest.json")
    manifest = json.loads(path.read_text("utf-8"))
    manifest["draft_version"] = 999
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not export.export_is_complete(destination)
    repaired = export.write_export(itinerary, destination)
    assert repaired["manifest"]["draft_version"] == 2
    assert export.export_is_complete(destination)


def test_markdown_includes_approved_weather_note_without_changing_json(destination, itinerary):
    note = "Forecast for 2026-11-15: Precipitation probability: 2%; Low: 20 C; High: 29 C."
    itinerary["days"][0]["weather_note"] = note
    export.write_export(itinerary, destination)
    assert "Weather: " + note in filename(destination, ".md").read_text(encoding="utf-8")
    assert json.loads(filename(destination, ".json").read_text(encoding="utf-8")) == itinerary
    assert export.export_is_complete(destination)
