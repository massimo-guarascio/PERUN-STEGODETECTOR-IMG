from models.stego_detector_extended_torch import StegoDetectorExtendedTorch
from models.stego_detector_residual import StegoDetectorResidualTorch


def build_model(model_type, input_dim, dl_size, drop_pcg, device):
    """
    Costruisce il modello scelto nel config.

    model_type:
        "original"
        "residual"
    """

    if model_type == "original":
        model = StegoDetectorExtendedTorch(
            input_dim=input_dim,
            dl_size=dl_size,
            drop_pcg=drop_pcg,
        )

    elif model_type == "residual":
        model = StegoDetectorResidualTorch(
            input_dim=input_dim,
            dl_size=dl_size,
            drop_pcg=drop_pcg,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    return model