from pathlib import Path

import torch

from config import get_config

from utils.data_utils import make_inference_loader_from_folder
from utils.models_utils import build_model

from inference import predict_folder_binary


def compute_input_dim_from_image_size(image_size, k_lsb=1):
    """
    Calcola input_dim a partire dalla dimensione immagine.

    image_size:
        tuple nel formato (height, width)

    Se k_lsb = 1:
        input_dim = height * width * 3

    Se k_lsb > 1:
        input_dim = height * width * 3 * k_lsb
    """

    height, width = image_size

    input_dim = height * width * 3 * k_lsb

    return input_dim


def main():
    # ============================================================
    # Load config
    # ============================================================

    config = get_config()

    # ============================================================
    # Device
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ============================================================
    # Read inference config
    # ============================================================

    image_folder = config["inference"]["image_folder"]
    output_csv_path = config["inference"]["output_csv_path"]
    image_size = config["inference"]["image_size"]

    # ============================================================
    # Read training/data/model config
    # ============================================================

    batch_size = config["training"]["batch_size"]
    threshold = config["training"]["threshold"]

    k_lsb = config["data"]["k_lsb"]

    model_type = config["model"]["model_type"]
    dl_size = config["model"]["dl_size"]
    drop_pcg = config["model"]["drop_pcg"]

    # ============================================================
    # Compute input dimension
    # ============================================================

    input_dim = compute_input_dim_from_image_size(
        image_size=image_size,
        k_lsb=k_lsb,
    )

    print("Image size:", image_size)
    print("k_lsb:", k_lsb)
    print("Input dim:", input_dim)

    # ============================================================
    # Checkpoint path
    # ============================================================

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])

    checkpoint_path = checkpoint_dir / config["paths"]["best_loss_filename"]

    # Se vuoi usare il best model by validation loss, usa invece:
    # checkpoint_path = checkpoint_dir / config["paths"]["best_loss_filename"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Loading checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # ============================================================
    # Build model
    # ============================================================

    detector = build_model(
        model_type=model_type,
        input_dim=input_dim,
        dl_size=dl_size,
        drop_pcg=drop_pcg,
        device=device,
    )

    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.to(device)
    detector.eval()

    print("Model loaded:", model_type)

    # ============================================================
    # Create inference loader
    # ============================================================

    inference_loader = make_inference_loader_from_folder(
        image_folder=image_folder,
        batch_size=batch_size,
        k_lsb=k_lsb,
        image_size=image_size,
        shuffle=False,
    )

    # ============================================================
    # Inference
    # ============================================================

    results_df = predict_folder_binary(
        model=detector,
        dataloader=inference_loader,
        device=device,
        threshold=threshold,
        output_csv_path=output_csv_path,
    )

    # ============================================================
    # Print output table
    # ============================================================

    print("\nInference results")
    print(results_df)

    print("\nSaved inference table to:", output_csv_path)


if __name__ == "__main__":
    main()