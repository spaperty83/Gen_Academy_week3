"""Manifest-last JSON and Markdown exports of approved itineraries."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.state import Itinerary

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
DISCLAIMER = (
    "Places were checked against their source pages; "
    "opening hours and prices may have changed."
)


class ExportError(Exception):
    """An export could not be safely completed."""


def _base_path(base_name: str | Path) -> Path:
    """Resolve export destinations strictly inside outputs, including symlinks."""
    base = Path(base_name)
    if not base.is_absolute():
        base = OUTPUTS_DIR.parent / base
    base = base.resolve()
    if not base.is_relative_to(OUTPUTS_DIR.resolve()) or base == OUTPUTS_DIR.resolve():
        raise ExportError("Export path must be inside outputs/")
    for suffix in (".json", ".md", ".manifest.json"):
        target = Path(str(base) + suffix)
        if not target.resolve().is_relative_to(OUTPUTS_DIR.resolve()):
            raise ExportError("Export path must be inside outputs/")
    return base


def _sha256(data: bytes) -> str:
    """Hash the exact bytes written to disk."""
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    """Render deterministic JSON without changing the itinerary."""
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n").encode("utf-8")


def _read_manifest(base: Path) -> dict[str, Any]:
    """Read a manifest without trusting file paths from it."""
    manifest = json.loads(Path(str(base) + ".manifest.json").read_text("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Invalid manifest")
    return manifest


def export_is_complete(base_name: str | Path) -> bool:
    """Require the manifest and matching JSON and Markdown hashes."""
    base = _base_path(base_name)
    try:
        manifest = _read_manifest(base)
        if not isinstance(manifest.get("itinerary_hash"), str):
            return False
        if type(manifest.get("draft_version")) is not int:
            return False
        if manifest["draft_version"] != int(base.name.rsplit("_v", 1)[1]):
            return False
        if manifest["itinerary_hash"] != manifest["files"][".json"]:
            return False
        return all(
            manifest["files"][suffix] == _sha256(
                Path(str(base) + suffix).read_bytes()
            )
            for suffix in (".json", ".md")
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def _markdown(itinerary: dict[str, Any]) -> bytes:
    """Render each day's stops from the same approved itinerary as JSON."""
    lines = [f"# {itinerary.get('city', 'Travel itinerary')}", ""]
    for index, day in enumerate(itinerary.get("days", []), 1):
        lines += [f"## Day {day.get('day_number', day.get('day', index))}", ""]
        for stop in day.get("stops", []):
            name = str(stop.get("place_name", stop.get("name", "")))
            note = str(stop.get("note", ""))
            url = str(stop.get("source_url", ""))
            lines.append(f"- {name}: {note} ([source](<{url}>))")
        lines.append("")
    lines += [DISCLAIMER, ""]
    return "\n".join(lines).encode("utf-8")


def _stage_file(target: Path, content: bytes) -> Path:
    """Create a complete flushed sibling temp file before any rename."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write(target: Path, content: bytes) -> None:
    """Atomically replace one destination, used for the final manifest."""
    temporary = _stage_file(target, content)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_pair(base: Path, json_data: bytes, markdown_data: bytes) -> None:
    """Stage both content files, then rename both; always clean temp files."""
    staged: list[tuple[Path, Path]] = []
    try:
        for suffix, content in ((".json", json_data), (".md", markdown_data)):
            target = Path(str(base) + suffix)
            staged.append((_stage_file(target, content), target))
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def write_export(
    itinerary: Itinerary | dict[str, Any], base_name: str | Path
) -> dict[str, Any]:
    """Commit JSON and Markdown before the manifest; repair partial exports."""
    try:
        base = _base_path(base_name)
        if isinstance(itinerary, Itinerary):
            itinerary = itinerary.model_dump(mode="json")
        json_data = _json_bytes(itinerary)
        itinerary_hash = _sha256(json_data)
        if export_is_complete(base):
            manifest = _read_manifest(base)
            if manifest["itinerary_hash"] == itinerary_hash:
                return {"path": str(base) + ".json", "manifest": manifest}
        markdown_data = _markdown(itinerary)
        # Version comes from the mandated filename, keeping JSON exactly approved.
        version = int(base.name.rsplit("_v", 1)[1])
        manifest = {
            "draft_version": version,
            "itinerary_hash": itinerary_hash,
            "files": {".json": _sha256(json_data), ".md": _sha256(markdown_data)},
        }
        base.parent.mkdir(parents=True, exist_ok=True)
        # An old manifest must not commit a partially replaced pair of files.
        Path(str(base) + ".manifest.json").unlink(missing_ok=True)
        _write_pair(base, json_data, markdown_data)
        _atomic_write(Path(str(base) + ".manifest.json"), _json_bytes(manifest))
        return {"path": str(base) + ".json", "manifest": manifest}
    except OSError:
        raise ExportError("Export write failed") from None
    except (ValueError, TypeError, IndexError):
        raise ExportError("Invalid itinerary or export base name") from None
