from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DEFAULT_DATASET = "timurakimoff/gemma-grpo-source"
VLLM_BNB_PLUGIN_REPOSITORY = "https://github.com/vllm-project/vllm-bnb-plugin.git"
# Pin the official plugin so an offline Kaggle run is reproducible even when
# the repository's main branch moves independently of the competition image.
VLLM_BNB_PLUGIN_COMMIT = "d4914a9e3c5927b33b7ef941385a4fda2187e2b9"


def _run(*args: str, cwd: Path, capture: bool = False) -> str:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def _build_vllm_bnb_plugin_wheel(publish_dir: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="gemma-vllm-bnb-") as temporary:
        checkout = Path(temporary) / "source"
        checkout.mkdir()
        _run("git", "init", "--quiet", cwd=checkout)
        _run("git", "remote", "add", "origin", VLLM_BNB_PLUGIN_REPOSITORY, cwd=checkout)
        _run(
            "git",
            "fetch",
            "--depth=1",
            "origin",
            VLLM_BNB_PLUGIN_COMMIT,
            cwd=checkout,
        )
        _run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=checkout)
        _run(
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(publish_dir),
            ".",
            cwd=checkout,
        )
        wheels = sorted(publish_dir.glob("vllm_bnb_plugin-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one vllm-bnb-plugin wheel, found {wheels}")
        shutil.copy2(checkout / "LICENSE", publish_dir / "vllm-bnb-plugin-LICENSE")
        return wheels[0]


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
    plugin_wheel = _build_vllm_bnb_plugin_wheel(publish_dir)
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
    print(f"offline NF4 plugin: {plugin_wheel.name} ({VLLM_BNB_PLUGIN_COMMIT[:12]})")


if __name__ == "__main__":
    main()
