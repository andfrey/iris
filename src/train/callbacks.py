import sys
from pathlib import Path

import numpy as np
from lightning.pytorch.callbacks import Callback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.utils import evaluate_regression


class PlotCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log scatter plot of true vs predicted values at the end of validation epoch"""
        evaluate_regression(
            np.concat(pl_module.val_targets, axis=0),
            np.concat(pl_module.val_predictions, axis=0),
            "val",
            trainer.logger.experiment,
            plot=True,
        )

        pl_module.val_predictions = []
        pl_module.val_targets = []
