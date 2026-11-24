from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import get_config
from train import Trainer

from utils.data_utils import (
    load_data,
    create_target,
    preprocess_lsb_features,
    make_full_loader,
    refresh_reproducibility,
)

from utils.models_utils import build_model

from utils.sampling import (
    nested_stratified_indices,
    make_indexed_loader,
    check_binary_indices_distribution,
)

from utils.plot_utils import plot_train_val_loss
# Se invece il file si chiama plot_utils.py, usa:
# from utils.plot_utils import plot_train_val_loss


def main():
    # ============================================================
    # Load config
    # ============================================================

    config = get_config()

    # ============================================================
    # Data config
    # ============================================================

    data_folder = config["data"]["data_folder"]
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]

    legit_folder = config["data"]["legit_folder"]
    stego_folders = config["data"]["stego_folders"]

    k_lsb = config["data"]["k_lsb"]

    # ============================================================
    # Experiment config
    # ============================================================

    seed = config["experiment"]["seed"]
    cwd = Path(config["experiment"]["cwd"])
    model_space = Path(config["experiment"]["model_space"])

    # ============================================================
    # Model config
    # ============================================================

    model_type = config["model"]["model_type"]
    dl_size = config["model"]["dl_size"]
    drop_pcg = config["model"]["drop_pcg"]

    # ============================================================
    # Training config
    # ============================================================

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    threshold = config["training"]["threshold"]
    criterion_name = config["training"]["criterion"]

    # ============================================================
    # Sampling config
    # ============================================================

    use_sampling = config["sampling"]["use_sampling"]
    fractions = config["sampling"]["fractions"]
    sampling_k = config["sampling"]["sampling_k"]

    # ============================================================
    # Reproducibility
    # ============================================================

    refresh_reproducibility(seed)

    # ============================================================
    # Device
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ============================================================
    # Load training set
    # ============================================================

    print("\nLoading training set...")

    training_set, num_training_legit, num_training_stego_for_class = load_data(
        main_data_folder=data_folder,
        usage_folder_name=train_split,
        legit_folder_name=legit_folder,
        stego_folder_names=stego_folders,
    )

    num_training_stego = np.sum(num_training_stego_for_class)

    y_train = create_target(
        legits=num_training_legit,
        stegos=num_training_stego,
    )

    print("Raw training shape:", training_set.shape)
    print("y_train shape:", y_train.shape)

    # ============================================================
    # Load validation set
    # ============================================================

    print("\nLoading validation set...")

    validation_set, num_val_legit, num_val_stego_for_class = load_data(
        main_data_folder=data_folder,
        usage_folder_name=val_split,
        legit_folder_name=legit_folder,
        stego_folder_names=stego_folders,
    )

    num_val_stego = np.sum(num_val_stego_for_class)

    y_val = create_target(
        legits=num_val_legit,
        stegos=num_val_stego,
    )

    print("Raw validation shape:", validation_set.shape)
    print("y_val shape:", y_val.shape)

    # ============================================================
    # LSB feature extraction
    # ============================================================

    print("\nExtracting LSB features...")

    training_set = preprocess_lsb_features(
        X=training_set,
        k_lsb=k_lsb,
    )

    validation_set = preprocess_lsb_features(
        X=validation_set,
        k_lsb=k_lsb,
    )

    input_dim = training_set.shape[1]

    print("Processed training shape:", training_set.shape)
    print("Processed validation shape:", validation_set.shape)
    print("Input dim:", input_dim)

    # ============================================================
    # DataLoaders
    # ============================================================

    if not use_sampling:
        print("\nUsing full training and validation sets...")

        train_loader = make_full_loader(
            X=training_set,
            y=y_train,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = make_full_loader(
            X=validation_set,
            y=y_val,
            batch_size=batch_size,
            shuffle=False,
        )

        checkpoint_dir = cwd / model_space
        plot_dir = cwd / config["paths"]["plot_dir"]

    else:
        print("\nUsing nested stratified sampling...")

        if fractions is None:
            raise ValueError(
                "Sampling is enabled but config['sampling']['fractions'] is None."
            )

        if sampling_k is None:
            raise ValueError(
                "Sampling is enabled but config['sampling']['sampling_k'] is None."
            )

        if sampling_k not in fractions:
            raise ValueError(
                f"sampling_k={sampling_k} is not in fractions={fractions}."
            )

        train_indices_by_fraction = nested_stratified_indices(
            y_stratify=y_train,
            fractions=fractions,
            seed=seed,
        )

        val_indices_by_fraction = nested_stratified_indices(
            y_stratify=y_val,
            fractions=fractions,
            seed=seed,
        )

        train_indices = train_indices_by_fraction[sampling_k]
        val_indices = val_indices_by_fraction[sampling_k]

        check_binary_indices_distribution(
            y=y_train,
            indices=train_indices,
            name=f"train fraction {sampling_k}",
        )

        check_binary_indices_distribution(
            y=y_val,
            indices=val_indices,
            name=f"validation fraction {sampling_k}",
        )

        train_loader = make_indexed_loader(
            X=training_set,
            y=y_train,
            indices=train_indices,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = make_indexed_loader(
            X=validation_set,
            y=y_val,
            indices=val_indices,
            batch_size=batch_size,
            shuffle=False,
        )

        run_name = f"sampling_{str(sampling_k).replace('.', '_')}"

        checkpoint_dir = cwd / model_space / run_name
        plot_dir = cwd / config["paths"]["plot_dir"] / run_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Checkpoint directory:", checkpoint_dir)
    print("Plot directory:", plot_dir)

    # ============================================================
    # Model
    # ============================================================

    detector = build_model(
        model_type=model_type,
        input_dim=input_dim,
        dl_size=dl_size,
        drop_pcg=drop_pcg,
        device=device,
    )

    print("Model:", model_type)

    # ============================================================
    # Criterion
    # ============================================================

    if criterion_name == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")

    print("Criterion:", criterion_name)

    # ============================================================
    # Optimizer
    # ============================================================

    optimizer = torch.optim.RMSprop(
        detector.parameters(),
        lr=learning_rate,
    )

    # ============================================================
    # Scheduler
    # ============================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.01,
        min_lr=1e-9,
    )

    # ============================================================
    # Trainer
    # ============================================================

    trainer = Trainer(
        model=detector,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        input_dim=input_dim,
        checkpoint_dir=checkpoint_dir,
        best_loss_filename=config["paths"]["best_loss_filename"],
        best_auc_filename=config["paths"]["best_auc_filename"],
        threshold=threshold,
    )

    # ============================================================
    # Fit
    # ============================================================

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
    )

    # ============================================================
    # Save history
    # ============================================================

    history_path = checkpoint_dir / config["paths"]["history_filename"]
    np.save(history_path, history)

    print("Saved history to:", history_path)

    # ============================================================
    # Plot train vs validation loss
    # ============================================================

    plot_train_val_loss(
        history=history,
        output_dir=plot_dir,
        filename="train_val_loss.pdf",
        title="Training and Validation Loss",
        log_scale=True,
    )


if __name__ == "__main__":
    main()