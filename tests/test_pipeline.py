from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from hyperag.ingest.pipeline import create_metadata_pipeline


def main():
    # snippet from the internet
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    while not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:
            raise FileNotFoundError("Project root (pyproject.toml) not found.")
    data_dir = project_root / "data" / "paul_graham"

    documents = SimpleDirectoryReader(str(data_dir)).load_data()

    pipeline = create_metadata_pipeline()
    pipeline.run(documents=documents, show_progress=True)


if __name__ == "__main__":
    main()
