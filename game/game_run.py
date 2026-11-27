"""Run identity: seeding and the run manifest (issue #27).

A run (GLOSSARY.md) can be given a seed, which fixes every source of
randomness, and a run directory, which holds its logs, models and
`manifest.json`. The manifest records what produced the run and whether it
has finished, so a runner can skip finished runs and resume the others.
"""
import datetime
import json
import os
import random
import subprocess

import numpy as np
import torch

MANIFEST_NAME = "manifest.json"
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seed_everything(seed):
    """Seed Python `random`, numpy and torch (CPU and every CUDA device), and
    make torch pick deterministic kernels. Exploration and PER sampling draw
    from numpy's global generator, so they are covered too."""
    # cuBLAS is only deterministic with a fixed workspace; it must be set
    # before the first cuBLAS call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: an op without a deterministic kernel warns instead of
    # stopping a long run.
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    """Every generator seed_everything seeds, so a resumed run continues the
    same random streams instead of replaying them from the seed."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=...) may have moved the tensors to the GPU;
    # the setters want CPU byte tensors.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def git_commit():
    """The checked-out commit of this repo, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True,
            text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def manifest_path(run_dir):
    return os.path.join(run_dir, MANIFEST_NAME)


def read_manifest(run_dir):
    """The run's manifest as a dict, or None if it has none yet."""
    path = manifest_path(run_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_manifest(run_dir, manifest):
    """Write atomically, so a kill mid-write keeps the previous manifest."""
    os.makedirs(run_dir, exist_ok=True)
    path = manifest_path(run_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def new_manifest(seed, budget, track):
    return {
        "seed": seed,
        "budget": budget,
        "track": track,
        "git_commit": git_commit(),
        "status": STATUS_RUNNING,
        "started_at": _now(),
        "finished_at": None,
        "stop_reason": None,
    }


def mark_finished(manifest, stop_reason):
    manifest.update(status=STATUS_FINISHED, finished_at=_now(),
                    stop_reason=stop_reason)
    return manifest
