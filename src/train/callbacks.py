import sys
from pathlib import Path

import numpy as np
from lightning.pytorch.callbacks import Callback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.utils import log_plot, log_regression_plots


class PlotCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log different visualization plots of true vs predicted values at the end of validation epoch"""
        log_regression_plots(
            np.concat(pl_module.val_targets, axis=0),
            np.concat(pl_module.val_predictions, axis=0),
            trainer.logger.experiment,
            "val",
        )
        pl_module.val_predictions = []
        pl_module.val_targets = []

    def on_train_epoch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        log_regression_plots(
            np.concat(pl_module.train_targets, axis=0),
            np.concat(pl_module.train_predictions, axis=0),
            trainer.logger.experiment,
            "train",
        )
        pl_module.train_predictions = []
        pl_module.train_targets = []


class DebugCallback(Callback):
    def on_sanity_check_start(self, trainer, pl_module):
        import matplotlib.pyplot as plt
        import wandb

        batch = next(iter(trainer.datamodule.train_dataloader()))
        X, y = batch
        batch_size = X.shape[0]
        channel_length = X.shape[1]
        fig, axes = plt.subplots(min(10, batch_size), min(10, channel_length), figsize=(12, 8))
        for i, ax in enumerate(axes):
            for j in range(len(ax)):
                ax[j].imshow(X[i, j].squeeze(), cmap="gray")
                ax[j].axis("off")
        trainer.logger.experiment.log({"sanity_check_images": wandb.Image(plt)})
        plt.close(fig)
