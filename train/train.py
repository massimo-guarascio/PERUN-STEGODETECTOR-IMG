from pathlib import Path
import time

import numpy as np
import torch
from utils.plot_utils import compute_auc_metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc




class Trainer:
    """
    Classe per gestire training e validation del detector binario.

    Il modello deve restituire logits, non probabilità.
    Quindi la loss corretta è:

        nn.BCEWithLogitsLoss()

    Le probabilità vengono calcolate solo per metriche e predizioni tramite:

        probs = torch.sigmoid(logits)
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        input_dim=None,
        checkpoint_dir="./output/models",
        best_loss_filename="best_model_by_loss.pt",
        best_auc_filename="best_model_by_auc.pt",
        threshold=0.5,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.input_dim = input_dim
        self.threshold = threshold

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss_model_path = self.checkpoint_dir / best_loss_filename
        self.best_auc_model_path = self.checkpoint_dir / best_auc_filename

        self.best_val_loss = float("inf")
        self.best_val_auc = -float("inf")

        self.history = {
            "loss": [],
            "accuracy": [],
            "auc": [],
            "pr_auc": [],

            "val_loss": [],
            "val_accuracy": [],
            "val_auc": [],
            "val_pr_auc": [],

            "lr": [],
        }

    def train_one_epoch(self, dataloader):
        """
        Esegue una singola epoca di training.

        Output:
            average_loss
            average_accuracy
            roc_auc
            pr_auc
        """

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        all_true = []
        all_scores = []

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)

            loss.backward()
            self.optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()

            total_correct += (preds == y_batch).sum().item()
            total_examples += batch_size

            all_true.extend(y_batch.detach().cpu().numpy().ravel())
            all_scores.extend(probs.detach().cpu().numpy().ravel())

        average_loss = total_loss / total_examples
        average_accuracy = total_correct / total_examples

        roc_auc, pr_auc = compute_auc_metrics(all_true, all_scores)

        return average_loss, average_accuracy, roc_auc, pr_auc

    def validate_one_epoch(self, dataloader):
        """
        Esegue una singola epoca di validation.

        Differenze rispetto al training:
            - model.eval()
            - torch.no_grad()
            - nessuna backpropagation
            - nessun optimizer.step()
        """

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        all_true = []
        all_scores = []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).view(-1, 1)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size

                probs = torch.sigmoid(logits)
                preds = (probs >= self.threshold).float()

                total_correct += (preds == y_batch).sum().item()
                total_examples += batch_size

                all_true.extend(y_batch.detach().cpu().numpy().ravel())
                all_scores.extend(probs.detach().cpu().numpy().ravel())

        average_loss = total_loss / total_examples
        average_accuracy = total_correct / total_examples

        roc_auc, pr_auc = compute_auc_metrics(all_true, all_scores)

        return average_loss, average_accuracy, roc_auc, pr_auc

    def _save_checkpoint(
        self,
        path,
        epoch,
        val_loss,
        val_acc,
        val_auc,
        val_pr_auc,
    ):
        """
        Salva checkpoint del modello.

        Uso float(...) e int(...) per evitare problemi di serializzazione
        con numpy scalar in versioni recenti di PyTorch.
        """

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_auc": float(val_auc),
            "val_pr_auc": float(val_pr_auc),
            "threshold": float(self.threshold),
        }

        if self.input_dim is not None:
            checkpoint["input_dim"] = int(self.input_dim)

        torch.save(checkpoint, path)

    def fit(self, train_loader, val_loader, num_epochs):
        """
        Esegue il training completo.

        Salva:
            - best model by validation loss
            - best model by validation AUC

        Output:
            history:
                dizionario con loss, accuracy, AUC, PR-AUC e learning rate.
        """

        start_time = int(round(time.time() * 1000))

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, train_auc, train_pr_auc = self.train_one_epoch(
                dataloader=train_loader
            )

            val_loss, val_acc, val_auc, val_pr_auc = self.validate_one_epoch(
                dataloader=val_loader
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)
            self.history["auc"].append(train_auc)
            self.history["pr_auc"].append(train_pr_auc)

            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_auc"].append(val_auc)
            self.history["val_pr_auc"].append(val_pr_auc)

            self.history["lr"].append(current_lr)

            print(
                f"Epoch [{epoch:02d}/{num_epochs}] "
                f"loss={train_loss:.6f} "
                f"acc={train_acc:.4f} "
                f"auc={train_auc:.4f} "
                f"pr_auc={train_pr_auc:.4f} | "
                f"val_loss={val_loss:.6f} "
                f"val_acc={val_acc:.4f} "
                f"val_auc={val_auc:.4f} "
                f"val_pr_auc={val_pr_auc:.4f} | "
                f"lr={current_lr:.2e}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                self._save_checkpoint(
                    path=self.best_loss_model_path,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    val_auc=val_auc,
                    val_pr_auc=val_pr_auc,
                )

                print(f"Saved best-loss model with val_loss={val_loss:.6f}")

            if not np.isnan(val_auc) and val_auc > self.best_val_auc:
                self.best_val_auc = val_auc

                self._save_checkpoint(
                    path=self.best_auc_model_path,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    val_auc=val_auc,
                    val_pr_auc=val_pr_auc,
                )

                print(f"Saved best-AUC model with val_auc={val_auc:.6f}")

        end_time = int(round(time.time() * 1000))
        total_time = end_time - start_time

        print("Total Training Time:", total_time, "ms")
        print("Best validation loss:", self.best_val_loss)
        print("Best validation AUC:", self.best_val_auc)

        return self.history