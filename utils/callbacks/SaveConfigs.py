import os
import yaml
import lightning as L
from typing_extensions import override
from lightning.pytorch.callbacks import Callback


class SaveConfigs(Callback):
    def __init__(self, args, save_path=None):
        self.args = args
        self.save_path = save_path if save_path is not None else self.args.version_path

    @property
    def save_dir(self) -> str:
        return os.path.join(str(self.save_path))

    @property
    def args_dir(self) -> str:
        return os.path.join(str(self.save_dir), 'args.yaml')

    @property
    def conf_dir(self) -> str:
        return os.path.join(str(self.save_dir), 'conf.yaml')

    def _save_args(self) -> None:
        with open(self.args_dir, 'w') as f:
            yaml.dump(vars(self.args), f, indent=4, default_flow_style=False)

    def _save_conf(self) -> None:
        conf_dict = {
            'data': self.args.data,
            'metrics': self.args.metrics,
            'data_back': self.args.data_back,
            'data_pred': self.args.data_pred,
            'data_val_rate': self.args.data_val_rate,
            'data_test_rate': self.args.data_test_rate,
        }
        with open(self.conf_dir, 'w') as f:
            yaml.dump(conf_dict, f, indent=4, default_flow_style=False)

    @override
    def on_sanity_check_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Called when the validation sanity check ends."""
        self._save_args()
        self._save_conf()
