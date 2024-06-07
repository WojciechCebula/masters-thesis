from pathlib import Path
from typing import Sequence, Dict, Any

import yaml

class DataPathsCollector:
    def __init__(self, root_dir: str | Path, split_path: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.split_files = self.load_split_file(split_path)
    
    def get_paths(self, split: str | None = None) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError

    def load_split_file(self, split_path: str | Path) -> Dict[str, Any]:
        raise NotImplementedError
