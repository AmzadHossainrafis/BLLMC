"""Creates the Bangala_LLM project structure."""

from pathlib import Path

BASE_DIR = Path("src/BLLMC")

STRUCTURE = {
    "components": {
        "attention": [
            "__init__.py",
            "multi_head.py",
            "grouped_query.py",
        ],
        "layers": [
            "__init__.py",
            "normalization.py",
            "activations.py",
            "feedforward.py",
            "embeddings.py",
        ],
        "blocks": [
            "__init__.py",
            "gpt2_block.py",
            "llama_block.py",
        ],
        "__files__": [
            "__init__.py",
            "config.py",
            "models.py",
            "trainer.py",
        ],
    },
    "data": [
        "__init__.py",
        "dataset.py",
        "loader.py",
        "ingestion.py",
    ],
    "pipeline": [
        "__init__.py",
        "train_pipeline.py",
        "inference_pipeline.py",
    ],
    "utils": [
        "__init__.py",
        "common.py",
        "logger.py",
        "exception.py",
    ],
    "__files__": [
        "__init__.py",
    ],
}


def create(base: Path, tree: dict | list):
    if isinstance(tree, list):
        base.mkdir(parents=True, exist_ok=True)
        for filename in tree:
            (base / filename).touch()
        return

    for name, subtree in tree.items():
        if name == "__files__":
            base.mkdir(parents=True, exist_ok=True)
            for filename in subtree:
                (base / filename).touch()
        elif isinstance(subtree, list):
            create(base / name, subtree)
        elif isinstance(subtree, dict):
            create(base / name, subtree)


if __name__ == "__main__":
    create(BASE_DIR, STRUCTURE)
    print(f"✅ Structure created under {BASE_DIR.resolve()}")
