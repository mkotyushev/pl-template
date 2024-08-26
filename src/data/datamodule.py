from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, default_collate


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        debug: bool = False,
        batch_size: int = 32,
        num_workers: int = 10,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

    def build_trainsforms(self) -> None:
        """Build transforms for train, val, and test datasets."""

    def setup(self, stage: str = None) -> None:
        """Setup train, val, and test datasets."""
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=None,
            shuffle=False,
            drop_last=True,
            collate_fn=default_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=None,
            shuffle=False,
            drop_last=False,
            collate_fn=default_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=None,
            shuffle=False,
            drop_last=False,
            collate_fn=default_collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
