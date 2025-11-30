import os
from lightning.pytorch.loggers import CSVLogger
from typing_extensions import override


class CustomCSVLogger(CSVLogger):
    def __init__(self, *args, version_dir: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.version_dir = version_dir

    @property
    @override
    def log_dir(self) -> str:
        return os.path.join(self.version_dir)

