import logging
import torch
from lightning import LightningModule
from typing import Any, Dict, Optional, Union
from torch import Tensor
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.utilities import grad_norm

from src.utils.utils import state_norm


logger = logging.getLogger(__name__)


class BaseModule(LightningModule):
    def __init__(
        self, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        skip_nan: bool = False,
        prog_bar_names: Optional[list] = None,
        lr: float = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metrics = None
        self.configure_metrics()

    def compute_loss_preds(self, batch, *args, **kwargs):
        """Compute losses and predictions."""

    def configure_metrics(self):
        """Configure task-specific metrics."""

    def get_batch_size(self, batch):
        """Get batch size."""

    def extract_targets_and_probas_for_metric(self, preds, batch):
        """Extract preds and targets from batch."""

    def update_metrics(self, prefix, preds, batch):
        """Update train metrics."""
        if self.metrics is None or prefix not in self.metrics:
            return
        y, y_proba = self.extract_targets_and_probas_for_metric(preds, batch)
        for metric in self.metrics[prefix].values():
            metric.update(y_proba, y)

    def on_train_epoch_start(self) -> None:
        """Called in the training loop at the very beginning of the epoch."""
        # Unfreeze all layers if freeze period is over
        if self.hparams.finetuning is not None:
            # TODO change to >= somehow
            if self.current_epoch == self.hparams.finetuning['unfreeze_before_epoch']:
                self.unfreeze()

    def unfreeze_only_selected(self):
        """
        Unfreeze only layers selected by 
        model.finetuning.unfreeze_layer_names_*.
        """
        if self.hparams.finetuning is not None:
            for name, param in self.named_parameters():
                selected = False

                if 'unfreeze_layer_names_startswith' in self.hparams.finetuning:
                    selected = selected or any(
                        name.startswith(pattern) 
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_startswith']
                    )

                if 'unfreeze_layer_names_contains' in self.hparams.finetuning:
                    selected = selected or any(
                        pattern in name
                        for pattern in self.hparams.finetuning['unfreeze_layer_names_contains']
                    )
                logger.info(f'Param {name}\'s requires_grad == {selected}.')
                param.requires_grad = selected

    def training_step(self, batch, batch_idx, **kwargs):
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        for loss_name, loss in losses.items():
            self.log(
                f'train_loss_{loss_name}', 
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.get_batch_size(batch),
            )
        self.update_metrics('train', preds, batch)

        # Handle nan in loss
        has_nan = False
        if torch.isnan(total_loss):
            has_nan = True
            logger.warning(
                f'Loss is nan at epoch {self.current_epoch} '
                f'step {self.global_step}.'
            )
        for loss_name, loss in losses.items():
            if torch.isnan(loss):
                has_nan = True
                logger.warning(
                    f'Loss {loss_name} is nan at epoch {self.current_epoch} '
                    f'step {self.global_step}.'
                )
        if has_nan:
            return None
        
        return total_loss
    
    def validation_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one val dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'val_loss_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=self.get_batch_size(batch),
            )
        self.update_metrics('val', preds, batch)
        return total_loss
    
    def test_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        total_loss, losses, preds = self.compute_loss_preds(batch, **kwargs)
        assert dataloader_idx is None or dataloader_idx == 0, 'Only one test dataloader is supported.'
        for loss_name, loss in losses.items():
            self.log(
                f'test_loss_{loss_name}', 
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                batch_size=self.get_batch_size(batch),
            )
        self.update_metrics('test', preds, batch)
        return total_loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None, **kwargs) -> Tensor:
        _, _, preds = self.compute_loss_preds(batch, **kwargs)
        return preds

    def log_metrics_and_reset(
        self, 
        prefix, 
        on_step=False, 
        on_epoch=True, 
        prog_bar_names=None,
        reset=True,
    ):
        if self.metrics is None or prefix not in self.metrics:
            return
        
        # Calculate and log metrics
        for name, metric in self.metrics[prefix].items():
            metric_value = metric.compute()
            if reset:
                metric.reset()
            
            if metric_value is not None:
                if not isinstance(metric_value, dict):
                    metric_value = {name: metric_value}
                
                for inner_name, value in metric_value.items():
                    prog_bar = False
                    if prog_bar_names is not None:
                        prog_bar = (inner_name in prog_bar_names)
                    self.log(
                        f'{prefix}_{inner_name}',
                        value,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=prog_bar,
                    )

    def on_train_epoch_end(self) -> None:
        """Called in the training loop at the very end of the epoch."""
        if self.metrics is None:
            return

        self.log_metrics_and_reset(
            'train',
            on_step=False,
            on_epoch=True,
            prog_bar_names=self.hparams.prog_bar_names,
            reset=True,
        )
    
    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        if self.metrics is None:
            return

        self.log_metrics_and_reset(
            'val',
            on_step=False,
            on_epoch=True,
            prog_bar_names=self.hparams.prog_bar_names,
            reset=True,
        )

    def on_test_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        if self.metrics is None:
            return

        self.log_metrics_and_reset(
            'test',
            on_step=False,
            on_epoch=True,
            prog_bar_names=self.hparams.prog_bar_names,
            reset=True,
        )

    def get_lr_decayed(self, lr, layer_index, layer_name):
        """
        Get lr decayed by 
            - layer index as (self.hparams.lr_layer_decay ** layer_index) if
              self.hparams.lr_layer_decay is float 
              (useful e. g. when new parameters are in classifer head)
            - layer name as self.hparams.lr_layer_decay[layer_name] if
              self.hparams.lr_layer_decay is dict
              (useful e. g. when pretrained parameters are at few start layers 
              and new parameters are the most part of the model)
        """
        if isinstance(self.hparams.lr_layer_decay, dict):
            for key in self.hparams.lr_layer_decay:
                if layer_name.startswith(key):
                    return lr * self.hparams.lr_layer_decay[key]
            return lr
        elif isinstance(self.hparams.lr_layer_decay, float):
            if self.hparams.lr_layer_decay == 1.0:
                return lr
            else:
                return lr * (self.hparams.lr_layer_decay ** layer_index)

    def build_parameter_groups(self):
        """Get parameter groups for optimizer."""
        names, params = list(zip(*self.named_parameters()))
        num_layers = len(params)
        
        if self.hparams.lr_layer_decay == 1.0:
            grouped_parameters = [
                {
                    'params': params, 
                    'lr': self.hparams.optimizer_init['init_args']['lr']
                }
            ]
        else:
            grouped_parameters = [
                {
                    'params': param, 
                    'lr': self.get_lr_decayed(
                        self.hparams.optimizer_init['init_args']['lr'], 
                        num_layers - layer_index - 1,
                        name
                    )
                } for layer_index, (name, param) in enumerate(self.named_parameters())
            ]
        
        logger.info(
            f'Number of layers: {num_layers}, '
            f'min lr: {names[0]}, {grouped_parameters[0]["lr"]}, '
            f'max lr: {names[-1]}, {grouped_parameters[-1]["lr"]}'
        )
        return grouped_parameters

    def configure_optimizer(self):
        optimizer = instantiate_class(args=self.build_parameter_groups(), init=self.hparams.optimizer_init)
        return optimizer

    def configure_lr_scheduler(self, optimizer):
        # Convert milestones from total persents to steps
        # for PiecewiceFactorsLRScheduler
        if (
            'PiecewiceFactorsLRScheduler' in self.hparams.lr_scheduler_init['class_path'] and
            self.hparams.pl_lrs_cfg['interval'] == 'step'
        ):
            total_steps = len(self.trainer.fit_loop._data_source.dataloader()) * self.trainer.max_epochs
            grad_accum_steps = self.trainer.accumulate_grad_batches
            self.hparams.lr_scheduler_init['init_args']['milestones'] = [
                int(milestone * total_steps / grad_accum_steps) 
                for milestone in self.hparams.lr_scheduler_init['init_args']['milestones']
            ]
        
        scheduler = instantiate_class(args=optimizer, init=self.hparams.lr_scheduler_init)
        scheduler = {
            "scheduler": scheduler,
            **self.hparams.pl_lrs_cfg,
        }

        return scheduler

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        if self.hparams.lr_scheduler_init is None:
            return optimizer

        scheduler = self.configure_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms."""
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'grad_2.0_norm_total' in norms:
                self.log('grad_2.0_norm_total', norms['grad_2.0_norm_total'])

        norms = state_norm(self, norm_type=2)
        if self.hparams.log_norm_verbose:
            self.log_dict(norms)
        else:
            if 'state_2.0_norm_total' in norms:
                self.log('state_2.0_norm_total', norms['state_2.0_norm_total'])
