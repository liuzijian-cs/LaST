import os
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

class CustomWandbLogger(WandbLogger):
    def __init__(self, *args, version_id: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.version_id = version_id

    @property
    @override
    def version(self):
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if the experiment exists else the id given to the constructor.

        """
        # don't create an experiment if we don't have one
        return self.version_id