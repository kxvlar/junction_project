from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ARTIFACT_SUBDIRS = ["raw", "processed", "reports", "metrics", "models"]


def get_archive_root(project_root: Path) -> Path:
    return project_root.parent / f"{project_root.name}_archives"


def _infer_git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _infer_date_window(project_root: Path) -> dict[str, str | None]:
    raw_payload = project_root / "artifacts" / "raw" / "junction_sleep_payload.json"
    if not raw_payload.exists():
        return {"start_date": None, "end_date": None, "source": None}
    payload = json.loads(raw_payload.read_text())
    return {
        "start_date": payload.get("start_date"),
        "end_date": payload.get("end_date"),
        "source": payload.get("source"),
    }


def _write_checksums(snapshot_root: Path) -> None:
    lines: list[str] = []
    for path in sorted(
        p
        for p in snapshot_root.rglob("*")
        if p.is_file() and p.name != "checksums.sha256"
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(snapshot_root).as_posix()}")
    (snapshot_root / "checksums.sha256").write_text("\n".join(lines) + "\n")


def archive_run(
    project_root: Path,
    version: str,
    command_lines: list[str] | None = None,
    notes: str | None = None,
) -> Path:
    archive_root = get_archive_root(project_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    snapshot_root = archive_root / version / timestamp
    snapshot_root.mkdir(parents=True, exist_ok=True)

    for name in ARTIFACT_SUBDIRS:
        source = project_root / "artifacts" / name
        destination = snapshot_root / "artifacts" / name
        if source.exists():
            shutil.copytree(source, destination, dirs_exist_ok=True)

    commands = command_lines or []
    (snapshot_root / "commands.txt").write_text("\n".join(commands) + ("\n" if commands else ""))

    git_commit = _infer_git_commit(project_root)
    (snapshot_root / "git_commit.txt").write_text(git_commit + "\n")

    dataset_window = _infer_date_window(project_root)
    manifest = {
        "version": version,
        "timestamp_utc": timestamp,
        "source_repo": str(project_root),
        "git_commit": git_commit,
        "date_window": dataset_window,
        "dataset_provenance": notes or "Junction portfolio snapshot",
        "artifact_inventory": sorted(
            p.relative_to(snapshot_root).as_posix()
            for p in snapshot_root.rglob("*")
            if p.is_file()
        ),
    }
    (snapshot_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    readme = (
        f"# {version} Snapshot\n\n"
        f"Timestamp: `{timestamp}`\n\n"
        f"Source repo: `{project_root}`\n\n"
        f"Notes: {notes or 'No additional notes recorded.'}\n"
    )
    (snapshot_root / "README.md").write_text(readme)
    _write_checksums(snapshot_root)

    archive_root.mkdir(parents=True, exist_ok=True)
    root_readme = archive_root / "README.md"
    if not root_readme.exists():
        root_readme.write_text(
            "# Junction Portfolio Archives\n\n"
            "Immutable run snapshots for the Playground Junction portfolio project. "
            "Each version folder contains timestamped artifacts, manifests, and checksums.\n"
        )

    return snapshot_root
