from __future__ import annotations
import os
import glob
import random as rn
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import torch
from PIL import Image


def extract_1_lsb_features(dataset: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Estrae 1 solo Least Significant Bit da ogni canale RGB di ogni immagine.

    Input:
        dataset:
            numpy array di shape [N_images, H, W, 3],
            con valori interi in [0, 255].

        dtype:
            tipo finale dell'array.
            Usa np.float32 se vuoi passare direttamente i dati al modello.
            Usa np.uint8 se vuoi risparmiare memoria e convertire a float nel DataLoader.

    Output:
        lsb_features:
            numpy array di shape [N_images, H * W * 3].
            Ogni immagine diventa un vettore binario.

    Esempio:
        154 = 10011010 -> LSB = 0
        155 = 10011011 -> LSB = 1
    """

    if dataset.ndim != 4:
        raise ValueError(
            f"Expected dataset with shape [N, H, W, 3], got {dataset.shape}"
        )

    if dataset.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB images with 3 channels, got shape {dataset.shape}"
        )

    lsb_dataset = dataset & 1
    lsb_dataset = lsb_dataset.reshape(dataset.shape[0], -1)

    return lsb_dataset.astype(dtype, copy=False)


def message_to_binary(message):
    """
    Converte un messaggio in rappresentazione binaria a 8 bit.

    Input possibile:
        - stringa
        - bytes
        - numpy array
        - intero
        - np.uint8

    Output:
        - se input è una stringa: stringa binaria lunga 8 * len(message)
        - se input è bytes/array: lista di stringhe binarie
        - se input è un intero: stringa binaria lunga 8 bit

    Esempio:
        message_to_binary(np.uint8(155)) -> '10011011'
    """

    if isinstance(message, str):
        return "".join(format(ord(i), "08b") for i in message)

    if isinstance(message, (bytes, np.ndarray)):
        return [format(int(i), "08b") for i in message]

    if isinstance(message, (int, np.uint8)):
        return format(int(message), "08b")

    raise TypeError(f"Input type not supported: {type(message)}")


# Alias per compatibilità con il notebook originale
messageToBinary = message_to_binary


def extract_k_lsb_features(data: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Estrae k Least Significant Bits da ogni canale RGB di ogni immagine.

    Input:
        data:
            numpy array di shape [N_images, H, W, 3]

        k:
            numero di bit meno significativi da estrarre per ogni canale.

    Output:
        lsb_features:
            numpy array di shape [N_images, H * W * 3 * k]

    Nota:
        Questa implementazione è fedele al notebook originale, ma è lenta
        perché cicla pixel per pixel. Per k=1 usare extract_1_lsb_features.
    """

    if k <= 0 or k > 8:
        raise ValueError("k must be an integer in [1, 8].")

    if data.ndim != 4:
        raise ValueError(f"Expected data with shape [N, H, W, 3], got {data.shape}")

    if data.shape[-1] != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {data.shape}")

    lsb_features = []

    for img_idx, img in enumerate(data, start=1):
        if img_idx % 50 == 0:
            print(f"Processed images: {img_idx}")

        binary_data = []

        for row in img:
            for pixel in row:
                r, g, b = message_to_binary(pixel)

                for i in range(1, k + 1):
                    binary_data.append(int(r[-i]))
                    binary_data.append(int(g[-i]))
                    binary_data.append(int(b[-i]))

        lsb_features.append(np.array(binary_data, dtype=np.float32))

    return np.array(lsb_features, dtype=np.float32)


def load_images(
    image_path: str | Path,
    image_size: Optional[Tuple[int, int]] = None,
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> List[np.ndarray]:
    """
    Carica tutte le immagini contenute in una cartella.

    Input:
        image_path:
            path della cartella contenente immagini.

        image_size:
            se non None, ridimensiona ogni immagine a questa size.
            Esempio: image_size=(32, 32)

        extensions:
            estensioni accettate.

    Output:
        images:
            lista di immagini RGB, ognuna di shape [H, W, 3].
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_path}")

    image_files = []

    for ext in extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        image_files.extend(image_path.glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    if len(image_files) == 0:
        raise ValueError(f"No images found in folder: {image_path}")

    images = []

    for file_path in image_files:
        img = Image.open(file_path).convert("RGB")

        if image_size is not None:
            img = img.resize(image_size)

        img = np.asarray(img, dtype=np.uint8)
        images.append(img)

    return images


def convert_np_array(vector: Sequence[np.ndarray]) -> np.ndarray:
    """
    Converte una lista di immagini in un numpy array.

    Input:
        vector:
            lista di immagini.

    Output:
        numpy array di shape [N_images, H, W, 3].

    Nota:
        Usa np.stack invece di np.array per fallire chiaramente se le immagini
        hanno dimensioni diverse.
    """

    if len(vector) == 0:
        raise ValueError("Cannot convert an empty image list to numpy array.")

    return np.stack(vector, axis=0)


def load_data(
    main_data_folder: str | Path,
    usage_folder_name: str,
    legit_folder_name: str,
    stego_folder_names: Sequence[str],
    image_size: Optional[Tuple[int, int]] = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Carica uno split del dataset.

    Input:
        main_data_folder:
            cartella principale del dataset.
            Esempio:
                "./dataset_stego_fake_n/real"

        usage_folder_name:
            nome dello split.
            Esempi:
                "train", "val", "test",
                oppure "training", "validation", "test_plain".

        legit_folder_name:
            nome della cartella con immagini pulite.
            Esempio:
                "legit"

        stego_folder_names:
            lista delle cartelle contenenti immagini stego.
            Esempio:
                ["attack"]
                oppure ["LSB_stego_url", "LSB_stego_eth", ...]

        image_size:
            se non None, ridimensiona tutte le immagini.

    Output:
        data_to_load:
            array con tutte le immagini concatenate.
            Prima le immagini legit, poi le immagini stego.

        num_legit_images:
            numero di immagini pulite.

        num_stego_images_for_class:
            array con il numero di immagini per ogni classe stego.
    """

    main_data_folder = Path(main_data_folder)
    data_path = main_data_folder / usage_folder_name

    if not data_path.exists():
        raise FileNotFoundError(f"Split folder does not exist: {data_path}")

    data_to_load = []
    num_stego_images_for_class = []

    legit_path = data_path / legit_folder_name
    legit_images = convert_np_array(
        load_images(legit_path, image_size=image_size)
    )

    data_to_load.append(legit_images)

    for stego_folder_name in stego_folder_names:
        stego_path = data_path / stego_folder_name

        stego_images = convert_np_array(
            load_images(stego_path, image_size=image_size)
        )

        num_stego_images_for_class.append(stego_images.shape[0])
        data_to_load.append(stego_images)

    num_legit_images = data_to_load[0].shape[0]

    print("# legit images:", num_legit_images)
    print("# stego images for class:", num_stego_images_for_class)

    data_to_load = np.concatenate(data_to_load, axis=0)

    print("data shape:", data_to_load.shape)
    print("done")

    return data_to_load, num_legit_images, np.array(num_stego_images_for_class)


def create_target_labels(legits: int, stego_type_number: Sequence[int]) -> np.ndarray:
    """
    Crea le label per il task di classificazione multi-classe.

    Label:
        0 = legit
        1 = prima classe stego
        2 = seconda classe stego
        ...
    """

    target = []

    target.append(np.zeros(legits, dtype=np.int64))

    class_id = 1

    for current_stego in stego_type_number:
        target.append(np.full(int(current_stego), class_id, dtype=np.int64))
        class_id += 1

    return np.concatenate(target)


def create_target(legits: int, stegos: int) -> np.ndarray:
    """
    Crea le label per detection binaria.

    Label:
        0 = immagine pulita
        1 = immagine stego / attack
    """

    y_legit = np.zeros(int(legits), dtype=np.float32)
    y_stego = np.ones(int(stegos), dtype=np.float32)

    return np.concatenate([y_legit, y_stego])


def refresh_reproducibility(seed: int) -> None:
    """
    Fissa i seed principali per rendere l'esperimento più riproducibile.

    Nota:
        La riproducibilità perfetta su GPU non è sempre garantita,
        ma questa configurazione è lo standard pratico in PyTorch.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    rn.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed fixed to: {seed}")




def make_full_loader(X, y, batch_size=256, shuffle=True):
    """
    Crea un DataLoader usando tutto il dataset.

    Input:
        X:
            numpy array di shape [N, input_dim]

        y:
            numpy array di shape [N]

        batch_size:
            dimensione del batch

        shuffle:
            se True, mescola i dati

    Output:
        DataLoader PyTorch
    """

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader



def preprocess_lsb_features(X, k_lsb):
    """
    Estrae le feature LSB.

    Se k_lsb = 1 usa la funzione veloce.
    Se k_lsb > 1 usa la funzione generale.
    """

    if k_lsb == 1:
        X = extract_1_lsb_features(X)
    else:
        X = extract_k_lsb_features(X, k=k_lsb)

    return X



class InferenceFolderDataset(Dataset):
    """
    Dataset PyTorch per inference su una cartella di immagini non etichettate.

    Ogni elemento restituisce:
        x:
            feature LSB dell'immagine.

        image_path:
            path dell'immagine originale.

    Non restituisce label perché in inference non sappiamo se l'immagine
    sia legit oppure attack.
    """

    def __init__(self, X, image_paths):
        self.X = X
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        x = self.X[idx]
        image_path = self.image_paths[idx]

        x = torch.tensor(x, dtype=torch.float32)

        return x, str(image_path)


def load_images_with_paths(
    image_folder,
    image_size=(330, 600),
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".webp"),
):
    """
    Carica immagini da una cartella e restituisce sia le immagini sia i path.

    Input:
        image_folder:
            cartella contenente immagini da analizzare.

        image_size:
            dimensione attesa delle immagini nel formato:
                (height, width)

            Default:
                (330, 600)

            Se una immagine ha dimensione diversa, viene sollevato errore.

        extensions:
            estensioni accettate.

    Output:
        images:
            numpy array di shape [N, H, W, 3].

        image_paths:
            lista dei path delle immagini caricate.

    Nota:
        Non facciamo resize.
        Il controllo serve a garantire che il modello riceva immagini con
        la stessa dimensione usata durante il training.
    """

    image_folder = Path(image_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

    image_paths = []

    for ext in extensions:
        image_paths.extend(image_folder.glob(f"*{ext}"))
        image_paths.extend(image_folder.glob(f"*{ext.upper()}"))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in folder: {image_folder}")

    images = []

    expected_height, expected_width = image_size

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img = np.asarray(img, dtype=np.uint8)

        if img.shape != (expected_height, expected_width, 3):
            raise ValueError(
                f"Invalid image size for {image_path}. "
                f"Expected {(expected_height, expected_width, 3)}, "
                f"got {img.shape}."
            )

        images.append(img)

    images = convert_np_array(images)

    return images, image_paths


def make_inference_loader_from_folder(
    image_folder,
    batch_size=256,
    k_lsb=1,
    image_size=(330, 600),
    shuffle=False,
):
    """
    Crea un DataLoader per inference partendo da una cartella di immagini.

    Input:
        image_folder:
            cartella contenente immagini da analizzare.

        batch_size:
            dimensione del batch.

        k_lsb:
            numero di LSB usati.
            Deve essere lo stesso usato nel training.

        image_size:
            dimensione attesa delle immagini nel formato:
                (height, width)

            Default:
                (330, 600)

        shuffle:
            normalmente False in inference.

    Output:
        inference_loader:
            DataLoader che restituisce:
                x_batch, path_batch
    """

    images, image_paths = load_images_with_paths(
        image_folder=image_folder,
        image_size=image_size,
    )

    X = preprocess_lsb_features(
        X=images,
        k_lsb=k_lsb,
    )

    dataset = InferenceFolderDataset(
        X=X,
        image_paths=image_paths,
    )

    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return inference_loader