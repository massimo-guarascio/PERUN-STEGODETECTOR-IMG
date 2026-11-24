from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc



def plot_train_val_loss(
    history,
    output_dir,
    filename="train_val_loss.pdf",
    title="Training and Validation Loss",
    log_scale=True,
    show=True,
):
    """
    Plotta training loss e validation loss.

    Input:
        history:
            dizionario con almeno:
                history["loss"]
                history["val_loss"]

        output_dir:
            cartella dove salvare il plot.

        filename:
            nome del file di output.

        title:
            titolo del grafico.

        log_scale:
            se True usa scala logaritmica sulla y.

        show:
            se True mostra il grafico.
    """

    if "loss" not in history:
        raise KeyError("history must contain key: 'loss'")

    if "val_loss" not in history:
        raise KeyError("history must contain key: 'val_loss'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    if log_scale:
        plt.yscale("log")

    plt.title(title)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if log_scale:
        plt.grid(True, which="both")
    else:
        plt.grid(True)

    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / filename
    plt.savefig(plot_path)

    if show:
        plt.show()
    else:
        plt.close()

    print("Saved plot to:", plot_path)

    return plot_path


def plot_roc_curve_from_df(
    roc_df,
    roc_auc,
    test_name,
    output_dir,
    filename=None,
    show=True,
):
    """
    Plotta la ROC curve su test set.

    Input:
        roc_df:
            DataFrame con colonne:
                - fpr
                - tpr
                - threshold

        roc_auc:
            valore della ROC-AUC.

        test_name:
            nome del test set.

        output_dir:
            cartella dove salvare il plot.

        filename:
            nome del file di output.
            Se None, viene creato automaticamente.

        show:
            se True mostra il grafico.
    """

    if "fpr" not in roc_df.columns:
        raise KeyError("roc_df must contain column: 'fpr'")

    if "tpr" not in roc_df.columns:
        raise KeyError("roc_df must contain column: 'tpr'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{test_name}_roc_curve.pdf"

    plt.figure(figsize=(7, 6))

    plt.plot(
        roc_df["fpr"],
        roc_df["tpr"],
        label=f"ROC-AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.title(f"ROC Curve - {test_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / filename
    plt.savefig(plot_path)

    if show:
        plt.show()
    else:
        plt.close()

    print("Saved ROC curve to:", plot_path)

    return plot_path


def plot_pr_curve_from_df(
    pr_df,
    pr_auc,
    test_name,
    output_dir,
    filename=None,
    show=True,
):
    """
    Plotta la Precision-Recall curve su test set.

    Input:
        pr_df:
            DataFrame con colonne:
                - precision
                - recall
                - threshold

        pr_auc:
            valore della PR-AUC.

        test_name:
            nome del test set.

        output_dir:
            cartella dove salvare il plot.

        filename:
            nome del file di output.
            Se None, viene creato automaticamente.

        show:
            se True mostra il grafico.
    """

    if "precision" not in pr_df.columns:
        raise KeyError("pr_df must contain column: 'precision'")

    if "recall" not in pr_df.columns:
        raise KeyError("pr_df must contain column: 'recall'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{test_name}_precision_recall_curve.pdf"

    plt.figure(figsize=(7, 6))

    plt.plot(
        pr_df["recall"],
        pr_df["precision"],
        label=f"PR-AUC = {pr_auc:.4f}",
    )

    plt.title(f"Precision-Recall Curve - {test_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / filename
    plt.savefig(plot_path)

    if show:
        plt.show()
    else:
        plt.close()

    print("Saved Precision-Recall curve to:", plot_path)

    return plot_path


def compute_auc_metrics(y_true, y_score):
    """
    Calcola ROC-AUC e PR-AUC.

    Input:
        y_true:
            label vere, 0/1.

        y_score:
            probabilità predette per la classe positiva, cioè attack/stego.

    Output:
        roc_auc:
            area sotto la ROC curve.

        pr_auc:
            area sotto la Precision-Recall curve.

    Nota:
        Se nel batch/epoch compare una sola classe, ROC-AUC e PR-AUC
        non sono ben definite. In quel caso restituiamo np.nan.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    roc_auc = roc_auc_score(y_true, y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc