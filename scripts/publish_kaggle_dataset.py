from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


DEFAULT_DATASET = "timurakimoff/gemma-grpo-source"


def _run(*args: str, cwd: Path, capture: bool = False) -> str:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the committed project as a private Kaggle Dataset."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    status = _run("git", "status", "--porcelain", cwd=root, capture=True)
    if status:
        raise SystemExit(
            "Refusing to publish a dirty worktree. Commit and test changes first.\n"
            + status
        )

    commit = _run("git", "rev-parse", "HEAD", cwd=root, capture=True)
    publish_dir = root / "dist" / "kaggle-source-dataset"
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    publish_dir.mkdir(parents=True)

    archive = publish_dir / "gemma-grpo-kaggle.zip"
    _run(
        "git",
        "archive",
        "--format=zip",
        f"--output={archive}",
        "HEAD",
        cwd=root,
    )
    (publish_dir / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    metadata = {
        "title": "Gemma GRPO source",
        "id": args.dataset,
        "licenses": [{"name": "other"}],
        "description": (
            "Private source snapshot for offline Gemma 4 Developer Agent "
            "training notebooks."
        ),
    }
    (publish_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    exists = subprocess.run(
        ["kaggle", "datasets", "status", args.dataset],
        cwd=root,
        text=True,
        capture_output=True,
    ).returncode == 0
    if exists:
        _run(
            "kaggle",
            "datasets",
            "version",
            "--path",
            str(publish_dir),
            "--message",
            f"Source commit {commit[:12]}",
            "--quiet",
            cwd=root,
        )
        action = "versioned"
    else:
        _run(
            "kaggle",
            "datasets",
            "create",
            "--path",
            str(publish_dir),
            "--quiet",
            cwd=root,
        )
        action = "created"

    print(f"{action}: https://www.kaggle.com/datasets/{args.dataset}")
    print(f"commit: {commit}")


if __name__ == "__main__":
    main()
