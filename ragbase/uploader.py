import shutil
from pathlib import Path
from typing import List

from streamlit.runtime.uploaded_file_manager import UploadedFile

from ragbase.config import Config


def upload_files(
    files: List[UploadedFile], remove_old_files: bool = True
) -> List[Path]:
    if remove_old_files:
        shutil.rmtree(Config.Path.DATABASE_DIR, ignore_errors=True)
        shutil.rmtree(Config.Path.DOCUMENTS_DIR, ignore_errors=True)
    Config.Path.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    file_paths = []
    for file in files:
        file_path = Config.Path.DOCUMENTS_DIR / file.name
        with file_path.open("wb") as f:
            f.write(file.getvalue())
        file_paths.append(file_path)
    return file_paths
