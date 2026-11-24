import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


def nested_stratified_indices(y_stratify, fractions, seed=123, min_per_class=1):
    """
    Crea subset stratificati annidati salvando SOLO gli indici.

    Garantisce:
        indices[0.02] ⊂ indices[0.04] ⊂ indices[0.06] ...

    Input:
        y_stratify:
            label rispetto a cui stratificare.
            Può essere binaria:
                0 = legit
                1 = attack

            oppure multi-classe:
                0 = legit
                1 = attack_type_A
                2 = attack_type_B
                ...

        fractions:
            lista di frazioni, ad esempio:
                [0.02, 0.04, 0.06]

        seed:
            seed per riproducibilità.

        min_per_class:
            numero minimo di esempi da prendere per classe,
            se la classe esiste.

    Output:
        indices_by_fraction:
            dizionario:
                fraction -> selected_indices
    """

    if fractions is None:
        raise ValueError("fractions cannot be None when sampling is enabled.")

    y_stratify = np.asarray(y_stratify)
    fractions = sorted(fractions)

    if len(fractions) == 0:
        raise ValueError("fractions cannot be empty.")

    for frac in fractions:
        if frac <= 0 or frac > 1:
            raise ValueError(f"Invalid fraction {frac}. Fractions must be in (0, 1].")

    rng = np.random.default_rng(seed)

    indices_by_fraction = {}
    shuffled_indices_by_class = {}

    unique_classes = np.unique(y_stratify)

    for cls in unique_classes:
        cls_indices = np.flatnonzero(y_stratify == cls).astype(np.int32)
        rng.shuffle(cls_indices)
        shuffled_indices_by_class[cls] = cls_indices

    for frac in fractions:
        selected_parts = []

        for cls in unique_classes:
            cls_indices = shuffled_indices_by_class[cls]
            n_cls = len(cls_indices)

            n_select = int(np.floor(frac * n_cls))

            if n_cls > 0:
                n_select = max(min_per_class, n_select)

            n_select = min(n_select, n_cls)

            selected_parts.append(cls_indices[:n_select])

        selected_indices = np.concatenate(selected_parts).astype(np.int32)
        rng.shuffle(selected_indices)

        indices_by_fraction[frac] = selected_indices

    return indices_by_fraction


class NumpyIndexedDataset(Dataset):
    """
    Dataset PyTorch che usa un numpy array e una lista di indici.

    Vantaggio:
        non crea copie di X.

    Invece di fare:
        X_reduced = X[indices]

    usa:
        Dataset(X, y, indices)

    Così il campione viene estratto solo quando serve nel batch.
    """

    def __init__(self, X, y, indices):
        self.X = X
        self.y = np.asarray(y)
        self.indices = np.asarray(indices, dtype=np.int32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        x = self.X[idx]
        y = self.y[idx]

        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


def make_indexed_loader(X, y, indices, batch_size=256, shuffle=True):
    """
    Crea un DataLoader a partire da X, y e una lista di indici.

    Input:
        X:
            numpy array con i dati.

        y:
            numpy array con le label.

        indices:
            indici selezionati dal sampling.

        batch_size:
            dimensione del batch.

        shuffle:
            se True, mescola i batch.

    Output:
        DataLoader PyTorch.
    """

    dataset = NumpyIndexedDataset(
        X=X,
        y=y,
        indices=indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader


def check_binary_indices_distribution(y, indices, name="sample"):
    """
    Controlla la distribuzione 0/1 di un subset selezionato tramite indici.

    Label attese:
        0 = legit / normal
        1 = attack / stego
    """

    y = np.asarray(y)
    indices = np.asarray(indices)

    y_subset = y[indices]

    total = len(y_subset)
    n_normal = int(np.sum(y_subset == 0))
    n_attack = int(np.sum(y_subset == 1))

    normal_ratio = n_normal / total if total > 0 else 0.0
    attack_ratio = n_attack / total if total > 0 else 0.0

    print(f"\nDistribution in {name}")
    print("Total:", total)
    print(f"Normal / legit: {n_normal} ({normal_ratio:.4f})")
    print(f"Attack / stego: {n_attack} ({attack_ratio:.4f})")

    return {
        "total": total,
        "normal": n_normal,
        "attack": n_attack,
        "normal_ratio": normal_ratio,
        "attack_ratio": attack_ratio,
    }