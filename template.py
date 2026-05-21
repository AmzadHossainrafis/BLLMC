import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


project_name = "BLLMC"
list_of_files = [
    # ── GitHub ──
    ".github/workflows/.gitkeep",
    # ── Package root ──
    f"src/{project_name}/__init__.py",
    # ── Components: core configs & models ──
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/config.py",
    f"src/{project_name}/components/models.py",
    f"src/{project_name}/components/trainer.py",
    # ── Components: attention mechanisms ──
    f"src/{project_name}/components/attention/__init__.py",
    f"src/{project_name}/components/attention/multi_head.py",
    f"src/{project_name}/components/attention/grouped_query.py",
    # ── Components: reusable layers ──
    f"src/{project_name}/components/layers/__init__.py",
    f"src/{project_name}/components/layers/normalization.py",
    f"src/{project_name}/components/layers/activations.py",
    f"src/{project_name}/components/layers/feedforward.py",
    f"src/{project_name}/components/layers/embeddings.py",
    # ── Components: transformer blocks ──
    f"src/{project_name}/components/blocks/__init__.py",
    f"src/{project_name}/components/blocks/gpt2_block.py",
    f"src/{project_name}/components/blocks/llama_block.py",
    # ── Data: dataset, loader, ingestion ──
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/dataset.py",
    f"src/{project_name}/data/loader.py",
    f"src/{project_name}/data/ingestion.py",
    # ── Pipeline ──
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    f"src/{project_name}/pipeline/inference_pipeline.py",
    # ── Utils ──
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/exception.py",
    # ── Project root files ──
    "artifacts/model_ckpt/.gitkeep",
    "fig/.gitkeep",
    "dataset/.gitkeep",
    "tests/.gitkeep",
    "config/config.yaml",
    "requirements.txt",
    "LICENSE",
    "setup.py",
    ".gitignore",
    "README.md",
    "app.py",
    "notebook/trials.ipynb",
    "notebook/EDA.ipynb",
    "templates/.gitkeep",
    "static/.gitkeep",
    "docs/.gitkeep",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")
