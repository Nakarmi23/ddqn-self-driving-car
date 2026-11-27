import os

# Seeded runs need cuBLAS's fixed workspace (game_run.seed_everything sets
# it), and cuBLAS reads it on first use. Tests share one process, so set it
# before any test touches the GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
