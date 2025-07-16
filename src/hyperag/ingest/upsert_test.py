from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

from ingest.pipeline import create_metadata_pipeline

load_dotenv()


async def main():
    # snippet from the internet
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    while not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:
            raise FileNotFoundError("Project root (pyproject.toml) not found.")
    data_dir = project_root / "data" / "red_hat_build_of_keycloak"

    documents = SimpleDirectoryReader(str(data_dir)).load_data()

    pipeline = create_metadata_pipeline()
    nodes = await pipeline.arun(documents=documents)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
