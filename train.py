# Parte 3. Training Loop
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#(a) Training parameters
def train_model(train_data, val_data, model, training_params, num_classes = 10, criterion = nn.CrossEntropyLoss()):
    #(b) Modelo a la GPU
    model.to(device)

    #(c) Optimizador Adam con weight decay.
    optimizer = torch.optim.Adam(model.parameters(), lr = training_params["learning_rate"], weight_decay = training_params["weight_decay"])

    #(d) Dataloaders con batch size determinado
    train_dataloader = DataLoader(train_data, batch_size = training_params["batch_size"], shuffle = True, pin_memory = True, num_workers = 6, drop_last = True)
    test_dataloader = DataLoader(val_data, batch_size = training_params["batch_size"], shuffle = False, pin_memory = True, num_workers = 6)

    #Definimos las métricas a utilizar en el train y validation
    train_metric = torchmetrics.F1Score(task = "multiclass", num_classes = num_classes).to(device)
    test_metric = torchmetrics.F1Score(task = "multiclass", num_classes = num_classes).to(device)

    #Listas para monitorear el loss de train y validation
    train_losses = []
    test_losses = []

    #(e) Training y Validation Loop
    for e in range(training_params["num_epochs"]):

        start_time = time.time()

        train_batch_losses = []
        test_batch_losses = []

        model.train()
        # (e) Training
        for batch in train_dataloader:

            X, y = batch["X"].to(device), batch["y"].to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            tm = train_metric(y_pred, y)
            train_batch_losses.append(loss.item())
    
        #Obtenemos el promedio del loss en el entrenamiento de cada batch        
        tm = train_metric.compute()
        train_epoch_loss = np.mean(train_batch_losses)

        model.eval()
        #(e) Validation
        with torch.no_grad():

            for batch in test_dataloader:

                X, y = batch["X"].to(device), batch["y"].to(device)

                y_pred = model(X)
                loss = criterion(y_pred, y)
                tst_m = test_metric(y_pred, y)
                test_batch_losses.append(loss.item())
        
        #Obtenemos el promedio del loss de cada batch
        tst_m = test_metric.compute()
        test_epoch_loss = np.mean(test_batch_losses)
        end_time = time.time()

        train_losses.append(train_epoch_loss)
        test_losses.append(test_epoch_loss)

        epoch_time = end_time - start_time

        #(f) Reporte final 
        print(f"Epoch: {e+1}- Time: {epoch_time:.2f} - Train Loss: {train_epoch_loss:.4f} - Validation Loss: {test_epoch_loss:.4f}- Train F1-Score: {tm:.4f} - Validation F1-Score: {tst_m:.4f}")


    model.train()

    #(g) Retornamos el modelo, train loss y validation loss
    return model, train_losses, test_losses
