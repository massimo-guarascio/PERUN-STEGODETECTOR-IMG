from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
)


def make_test_loader(X_test, y_test, batch_size=256):
    """
    Crea un DataLoader PyTorch per il test set.

    Input:
        X_test:
            numpy array di shape [N, input_dim].

        y_test:
            numpy array di shape [N].

        batch_size:
            dimensione del batch.

    Output:
        test_loader:
            DataLoader che restituisce batch (x_batch, y_batch).
    """

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader


def predict_proba_binary(model, dataloader, device, threshold=0.5):
    """
    Calcola le probabilità predette dal modello per un task binario.

    Input:
        model:
            modello PyTorch.

        dataloader:
            DataLoader del test set.

        device:
            cpu oppure cuda.

        threshold:
            soglia per trasformare probabilità in label binarie.

    Output:
        y_true:
            array numpy con label vere 0/1.

        y_score:
            array numpy con probabilità predette per la classe positiva.

        y_pred:
            array numpy con label predette usando la soglia scelta.
    """

    model.eval()

    all_true = []
    all_scores = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)
            probs = torch.sigmoid(logits)

            all_scores.extend(probs.detach().cpu().numpy().ravel())
            all_true.extend(y_batch.detach().cpu().numpy().ravel())

    y_true = np.asarray(all_true).astype(int)
    y_score = np.asarray(all_scores)
    y_pred = (y_score >= threshold).astype(int)

    return y_true, y_score, y_pred


def evaluate_binary_test_set(
    model,
    test_loader,
    device,
    test_name,
    output_dir,
    threshold=0.5,
):
    """
    Valuta il modello su un test set binario.

    Metriche calcolate:
        - accuracy
        - precision per classe positiva
        - recall per classe positiva
        - f1 per classe positiva
        - macro precision
        - macro recall
        - macro f1
        - ROC-AUC
        - PR-AUC
        - FPR alla soglia scelta

    Inoltre salva:
        1. summary metrics in CSV;
        2. classification report in CSV;
        3. ROC curve in CSV;
        4. Precision-Recall curve in CSV.

    Input:
        model:
            modello PyTorch.

        test_loader:
            DataLoader del test set.

        device:
            cpu oppure cuda.

        test_name:
            nome del test, ad esempio "test_plain", "test_real", "test_fake".

        output_dir:
            cartella dove salvare i risultati.

        threshold:
            soglia per trasformare probabilità in label.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Prediction
    # ----------------------------

    y_true, y_score, y_pred = predict_proba_binary(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=threshold,
    )

    # ----------------------------
    # Classification report
    # ----------------------------

    print(f"\nClassification report - {test_name}")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["legit", "stego"],
        digits=4,
        zero_division=0,
    )

    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["legit", "stego"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).T

    report_path = output_dir / f"{test_name}_classification_report.csv"
    report_df.to_csv(report_path)

    # ----------------------------
    # Confusion matrix
    # ----------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    # ----------------------------
    # Scalar metrics
    # ----------------------------

    acc_score = accuracy_score(y_true, y_pred)

    precision_pos = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall_pos = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1_pos = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ROC-AUC e PR-AUC sono definite solo se ci sono entrambe le classi
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan

        roc_df = pd.DataFrame({
            "fpr": [],
            "tpr": [],
            "threshold": [],
        })

        pr_df = pd.DataFrame({
            "precision": [],
            "recall": [],
            "threshold": [],
        })

    else:
        roc_auc = roc_auc_score(y_true, y_score)

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_true,
            y_score,
        )

        pr_auc = auc(recall_curve, precision_curve)

        fpr_curve, tpr_curve, roc_thresholds = roc_curve(
            y_true,
            y_score,
        )

        roc_df = pd.DataFrame({
            "fpr": fpr_curve,
            "tpr": tpr_curve,
            "threshold": roc_thresholds,
        })

        # precision_curve e recall_curve hanno lunghezza len(thresholds)+1.
        # Per salvare in CSV aggiungiamo NaN all'ultimo threshold.
        pr_thresholds_padded = np.append(pr_thresholds, np.nan)

        pr_df = pd.DataFrame({
            "precision": precision_curve,
            "recall": recall_curve,
            "threshold": pr_thresholds_padded,
        })

    # ----------------------------
    # Summary CSV
    # ----------------------------

    metrics_row = {
        "test_name": test_name,
        "threshold": threshold,

        "accuracy": acc_score,

        "precision_positive": precision_pos,
        "recall_positive": recall_pos,
        "f1_positive": f1_pos,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "fpr_at_threshold": fpr_at_threshold,
    }

    metrics_df = pd.DataFrame([metrics_row])

    metrics_path = output_dir / f"{test_name}_summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # ----------------------------
    # Curve CSV
    # ----------------------------

    roc_path = output_dir / f"{test_name}_roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)

    pr_path = output_dir / f"{test_name}_precision_recall_curve.csv"
    pr_df.to_csv(pr_path, index=False)

    # ----------------------------
    # Print sintetico
    # ----------------------------

    print("Summary metrics")
    print(metrics_df)

    print("\nSaved files:")
    print("Classification report:", report_path)
    print("Summary metrics:", metrics_path)
    print("ROC curve:", roc_path)
    print("Precision-Recall curve:", pr_path)

    return metrics_df, report_df, roc_df, pr_df


def predict_folder_binary(
    model,
    dataloader,
    device,
    threshold=0.5,
    output_csv_path=None,
):
    """
    Esegue inference binaria su immagini non etichettate.

    Output:
        DataFrame con:
            image_path
            attack_probability
            predicted_class
            predicted_label
            threshold
    """

    model.eval()

    all_paths = []
    all_scores = []

    with torch.no_grad():
        for x_batch, path_batch in dataloader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)
            probs = torch.sigmoid(logits)

            scores = probs.detach().cpu().numpy().ravel()

            all_scores.extend(scores)
            all_paths.extend(list(path_batch))

    all_scores = np.asarray(all_scores)

    predicted_class = (all_scores >= threshold).astype(int)

    predicted_label = np.where(
        predicted_class == 1,
        "attack",
        "legit",
    )

    results_df = pd.DataFrame({
        "image_path": all_paths,
        "attack_probability": all_scores,
        "predicted_label": predicted_label

    })

    results_df = results_df.sort_values(
        by="attack_probability",
        ascending=False,
    ).reset_index(drop=True)

    results_json = results_df.to_json(
    orient="records",
    indent=4,
    force_ascii=False,
)

    # Print in formato JSON
    print("\nInference results JSON")
    print(results_json)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        print("Saved inference results to:", output_csv_path)

    return results_df