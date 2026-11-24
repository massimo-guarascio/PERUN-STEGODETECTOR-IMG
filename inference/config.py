from pathlib import Path


CONFIG = {
    # ============================================================
    # Data configuration
    # ============================================================
    "data": {
        "data_folder": "./dataset_stego_fake_n/fake_news/real",

        "train_split": "train",
        "val_split": "val",
        "test_split": "test",

        "legit_folder": "legit",
        "stego_folders": ["attack"],

        # Number of least significant bits to extract.
        # k_lsb = 1 means we use only 1 LSB.
        "k_lsb": 1,
    },

    # ============================================================
    # Experiment configuration
    # ============================================================
    "experiment": {
        "seed": 230782,
        "cwd": "./",
        "model_space": "output/models-separate_k1_residual_real",
    },

    # ============================================================
    # Model configuration
    # ============================================================
    "model": {
        # Options could be:
        #   "original"
        #   "residual"
        "model_type": "residual",

        "dl_size": 128,
        "drop_pcg": 0.025,
    },

    # ============================================================
    # Training configuration
    # ============================================================
    "training": {
        "num_epochs": 20,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "threshold": 0.5,
        "criterion": "bce_with_logits",
    },



    "inference": {
    "image_folder": "./dataset_stego_fake_n/test_sample",
    "output_csv_path": "./output/inference_results/inference_results.csv",

    # Dimensione attesa nel formato:
    # (height, width)
    "image_size": (330, 600),

    },

    # ============================================================
    # Sampling configuration
    # ============================================================
    "sampling": {
        # If False, use the full training and validation sets.
        # If True, use nested stratified sampling.
        "use_sampling": False,

        # Example:
        #   [0.01, 0.02, 0.04, 0.06]
        #
        # If use_sampling is False, keep this as None.
        "fractions": None,
        "sampling_k": None,
    },

    # ============================================================
    # Output paths
    # ============================================================
    "paths": {
        "plot_dir": "plot_train",
        "test_metrics_dir": "output/test_metrics",

        "best_loss_filename": "best_model_stego_detection_by_loss.pt",
        "best_auc_filename": "best_model_stego_detection_by_auc.pt",

        "history_filename": "history.npy",
    },
}


def get_config():
    """
    Returns the experiment configuration.
    """

    config = CONFIG.copy()

    cwd = Path(config["experiment"]["cwd"])
    model_space = config["experiment"]["model_space"]

    config["paths"]["checkpoint_dir"] = str(cwd / model_space)
    config["paths"]["history_path"] = str(
        cwd / model_space / config["paths"]["history_filename"]
    )

    return config