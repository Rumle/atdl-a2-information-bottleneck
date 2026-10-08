from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch

from shared import IBData, IBNetwork, Settings


def train(
    settings: Settings,
    run_dir: str | Path,
    accelerator: str = "auto",
    resume: str | Path | None = None,
    callbacks: tuple[Callback, ...] = (),
) -> pl.Trainer:
    run_dir = Path(run_dir).resolve()
    if resume is not None:
        model = IBNetwork.load_from_checkpoint(resume, map_location="cpu")
        # Keep the saved setup. Only the total epoch target can change on resume.
        settings = replace(model.settings, epochs=settings.epochs)
    settings = replace(settings, data_path=str(Path(settings.data_path).resolve()))
    pl.seed_everything(settings.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = IBData(settings)
    if resume is None:
        model = IBNetwork(data.dataset.tensors[0].shape[1], settings)
    else:
        model.settings = settings
        model.save_hyperparameters({"settings": asdict(settings)})
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "split.npz", train=data.train_indices, test=data.test_indices)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        save_last="link",
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=settings.epochs,
        precision="32-true",
        deterministic=True,
        callbacks=[*callbacks, checkpoint],
        logger=CSVLogger(str(run_dir), name="logs"),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume is not None else None)
    return trainer


def train_runs(
    settings: Settings,
    num_runs: int,
    run_dir: str | Path,
    accelerator="auto",
    callback=Callback,
):
    random = np.random.default_rng(settings.seed)
    for run_index in range(num_runs):
        seed, split_seed = random.integers(0, 2**32, size=2).tolist()
        run_settings = replace(settings, seed=seed, split_seed=split_seed, random_split=True)
        print(f"Run {run_index + 1}/{num_runs}")
        current_run_dir = Path(run_dir) / f"run-{run_index + 1}"
        checkpoint = current_run_dir / "checkpoints/last.ckpt"
        resume = checkpoint if checkpoint.is_file() else None
        if resume is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            finished_epochs = saved["loops"]["fit_loop"]["epoch_progress"]["total"]["processed"]
            if finished_epochs >= settings.epochs:
                print("Already complete, skipping.")
                continue
            print("Resuming from the last checkpoint.")
        elif current_run_dir.exists():
            print("No checkpoint yet, restarting this run.")
        # A fresh callback keeps measurements from different runs separate.
        train(
            run_settings,
            current_run_dir,
            accelerator=accelerator,
            resume=resume,
            callbacks=(callback(),),
        )


if __name__ == "__main__":
    settings = Settings(epochs=8000)
    num_runs = 50
    train_runs(settings, num_runs, "runs/randomized", accelerator="auto")