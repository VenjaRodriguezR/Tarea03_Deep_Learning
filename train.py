# Parte 3. Training Loop
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import time
import numpy as np
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(
    train_data,
    val_data,
    model,
    architecture_name,  # Nombre de la arquitectura
    training_params,
    num_classes,
    criterion=nn.CrossEntropyLoss(),
):
    # Inicializa un experimento único en wandb
    wandb.init(
        project="Tarea03_Deep_Learning",
        name=f"{architecture_name}-experiment",  # Nombre del experimento
        tags=[architecture_name, "multiclass", "pretrained"],
        config={
            "architecture": architecture_name,
            "learning_rate": training_params["learning_rate"],
            "batch_size": training_params["batch_size"],
            "num_epochs": training_params["num_epochs"],
            "num_classes": num_classes,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
        },
    )

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params["learning_rate"],
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=training_params["batch_size"],
        shuffle=True,
        num_workers=min(6, torch.get_num_threads()),
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=training_params["batch_size"],
        shuffle=False,
        num_workers=min(6, torch.get_num_threads()),
        pin_memory=True,
        persistent_workers=True,
    )

    # Métricas
    train_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes).to(device)
    val_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes).to(device)

    best_val_loss = float("inf")
    best_model_weights = None

    train_loss = []
    val_loss = []

    for e in range(training_params["num_epochs"]):
        start_time = time.time()

        # Entrenamiento
        model.train()
        train_batch_loss = []
        train_metric.reset()
        for batch in train_dataloader:
            X, y = batch
            X, y = X.to(device), y.long().to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())
            train_metric(y_hat, y)

        train_epoch_loss = np.mean(train_batch_loss)
        train_f1 = train_metric.compute()

        # Validación
        model.eval()
        val_batch_loss = []
        val_metric.reset()
        with torch.no_grad():
            for batch in val_dataloader:
                X, y = batch
                X, y = X.to(device), y.long().to(device)

                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_batch_loss.append(loss.item())
                val_metric(y_hat, y)

        val_epoch_loss = np.mean(val_batch_loss)
        val_f1 = val_metric.compute()

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_weights = model.state_dict()

        wandb.log({
            "epoch": e + 1,
            "train_loss": train_epoch_loss,
            "train_f1": train_f1.item(),
            "val_loss": val_epoch_loss,
            "val_f1": val_f1.item(),
            "elapsed_time": time.time() - start_time,
        })

        print(
            f"Epoch: {e+1}/{training_params['num_epochs']} - "
            f"Time: {time.time() - start_time:.2f}s - "
            f"Train Loss: {train_epoch_loss:.4f}, Train F1: {train_f1:.4f} - "
            f"Validation Loss: {val_epoch_loss:.4f}, Validation F1: {val_f1:.4f}"
        )

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("Loaded best model weights based on validation loss.")

    wandb.finish()
    return model, train_loss, val_loss

