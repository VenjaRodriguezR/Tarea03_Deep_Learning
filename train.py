# Parte 3. Training Loop
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import time
import numpy as np
import wandb
from tqdm import tqdm
import os
import timm
from transforms import transforming
import json
from utils import generate_architecture_name
from torchvision.datasets import ImageFolder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(
    train_data: ImageFolder,
    val_data: ImageFolder,
    model: timm.models,
    architecture_name: str,  # Nombre de la arquitectura
    training_params: dict,
    num_classes: int,
    criterion: torch.nn.modules.loss = nn.CrossEntropyLoss(),
) -> tuple:  
    
    print(f"Using device: {device}")
    # Inicializa un experimento único en wandb
    torch.cuda.empty_cache()

    wandb.init(
        project = "Tarea03_Deep_Learning",
        name = f"{architecture_name}-experiment",  # Nombre del experimento
        tags = [architecture_name, "multiclass", "pretrained"],
        config = {
            "architecture": architecture_name,
            "learning_rate": training_params["learning_rate"],
            "batch_size": training_params["batch_size"],
            "num_epochs": training_params["num_epochs"],
            "num_classes": num_classes,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
        },
        settings = wandb.Settings(init_timeout = 180)
    )

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = training_params["learning_rate"],
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size = training_params["batch_size"],
        shuffle = True,
        num_workers = 6,
        pin_memory = True,
        persistent_workers = True,
        drop_last = True
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size = training_params["batch_size"],
        shuffle = False,
        num_workers =  6,
        pin_memory = True,
        persistent_workers = True,
    )

    # Métricas
    train_metric = torchmetrics.Accuracy(task = "multiclass", num_classes = num_classes).to(device)
    val_metric = torchmetrics.Accuracy(task = "multiclass", num_classes = num_classes).to(device)

    early_stopping_patience = 7
    best_val_loss = float("inf")
    patience_counter = 0

    # Directorio para guardar los checkpoints
    checkpoint_dir = os.path.join('checkpoints', architecture_name)
    os.makedirs(checkpoint_dir, exist_ok = True)

    train_loss = []
    val_loss = []
    val_acc_history = []

    for e in tqdm(range(training_params["num_epochs"])):

        start_time = time.time()
        train_batch_loss = []
        val_batch_loss = []

        train_metric.reset()
        val_metric.reset()

        model.train()
    
        for batch in train_dataloader:
            X, y = batch
            X, y = X.to(device), y.long().to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            tr_acc = train_metric(y_hat, y)
            train_batch_loss.append(loss.item())

        tr_acc = train_metric.compute()
        train_epoch_loss = np.mean(train_batch_loss)

        model.eval()

        with torch.no_grad():
            for batch in val_dataloader:
                X, y = batch
                X, y = X.to(device), y.long().to(device)

                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_acc = val_metric(y_hat, y)
                val_batch_loss.append(loss.item())

        val_acc = val_metric.compute()
        val_epoch_loss = np.mean(val_batch_loss)

        val_acc_history.append(val_acc.item())

        end_time = time.time()
        elapsed_time = end_time - start_time

        wandb.log({
            "epoch": e + 1,
            "train_loss": train_epoch_loss,
            "train_acc": tr_acc.item(),
            "val_loss": val_epoch_loss,
            "val_acc": val_acc.item(),
            "elapsed_time": elapsed_time,
        })

        print(
            f"Epoch: {e+1}/{training_params['num_epochs']} - "
            f"Time: {elapsed_time:.2f}s - "
            f"Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {tr_acc:.4f} - "
            f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
        )

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)

        # Checkpointing
        if val_epoch_loss < best_val_loss:
          best_val_loss = val_epoch_loss
          patience_counter = 0
          # Guardar el modelo
          checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{e+1}.pth')
          torch.save(model.state_dict(), checkpoint_path)
          print(f'Model checkpoint saved at epoch {e+1}')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {e - 5}. Best Validation Loss: {best_val_loss:.4f}')
            break

    wandb.finish()
    return model, train_loss, val_loss, val_acc_history


##########################################################################################################
## We are gonna take a feature extraction approach
class FrozenNet(nn.Module):
    def __init__(self, num_classes: int = 47, frozen: bool = True, type: str = "efficientnet_b5.sw_in12k", pretraining: bool = True ):
        
        super().__init__()
        self.num_classes = num_classes
        # Cargar EfficientNet preentrenado de timm
        self.backbone = timm.create_model(type, pretrained = pretraining, num_classes = 0)

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        for layer in [self.backbone.conv_head, self.backbone.bn2, self.backbone.global_pool]:
            for param in layer.parameters():
                param.requires_grad = True

        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 512),          # Primera capa totalmente conectada
            nn.BatchNorm1d(512),         # Normalización por lotes para la salida de la capa
            nn.ReLU(),                   # Activación no lineal
            nn.Dropout(0.3),             # Dropout para prevenir el sobreajuste

            nn.Linear(512, 256),         # Segunda capa totalmente conectada más pequeña
            nn.BatchNorm1d(256),         # Normalización por lotes
            nn.ReLU(),                   # Activación no lineal
            nn.Dropout(0.3),             # Dropout adicional para regularizar

            nn.Linear(256, self.num_classes)  # Capa final de salida
        )

        # self.fc_layers.apply(initialize_weights_kaiming)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_layers(x)
        return x
    

##############################################################################################################
def testing(model_input: timm.models, data_eva: ImageFolder, training_params: dict):
    # Cambiar el modelo al modo de evaluación
    model_input.eval()

    # Inicializar el cálculo de F1-score
    acc_metric = torchmetrics.Accuracy(task = "multiclass", num_classes = 47).to(device)
    acc_metric.reset()
    data_eva = DataLoader(data_eva, batch_size = training_params["batch_size"], shuffle = False, num_workers = 6, pin_memory = True)
    criterion = nn.CrossEntropyLoss()
    # Desactivar el cálculo de gradientes ya que no necesitamos backpropagation
    with torch.no_grad():
        test_loss = []
        # Iterar sobre el conjunto de datos de test
        for batch in data_eva:
            # Mover los datos al dispositivo adecuado (CPU o GPU)
            features, target = batch[0].to(device), batch[1].to(device)
            # Hacer predicciones con el modelo
            output = model_input(features)
            loss = criterion(output, target.squeeze())  # Asegurarse de que las dimensiones coincidan
            test_loss.append(loss.item())

            # Convertir las predicciones a clases con la probabilidad máxima
            preds = torch.argmax(output, dim = 1)

            # Actualizar el cálculo del la accuracy
            acc_metric.update(preds, target)

    # Calcular la accuracy final
    acc_score = acc_metric.compute().item()

    print(f'Test Accuracy: {acc_score:.5f}')

    return acc_score

####################################################################################33
def save_results_to_json(result_file: str, results: dict) -> None:
    """
    Guarda los resultados en un archivo JSON, asegurando que los datos sean serializables.
    """
    try:
        # Crear directorio si no existe
        directory = os.path.dirname(result_file)
        if directory:  # Solo crea el directorio si no es vacío
            os.makedirs(directory, exist_ok = True)

        # Convertir todos los valores problemáticos a tipos serializables
        results_serializable = {
            key: (int(value) if isinstance(value, (np.integer, np.int64)) else
                  float(value) if isinstance(value, (np.floating, np.float64)) else
                  value)
            for key, value in results.items()
        }

        if os.path.exists(result_file):
            # Cargar los resultados existentes si el archivo ya existe
            with open(result_file, "r") as f:
                existing_results = json.load(f)
        else:
            existing_results = []

        # Agregar nuevos resultados al archivo
        existing_results.append(results_serializable)

        # Escribir los resultados actualizados
        with open(result_file, "w") as f:
            json.dump(existing_results, f, indent = 4)
        print(f"Resultados guardados correctamente en {result_file}")
    except Exception as e:
        print(f"Error al guardar resultados en JSON: {e}")



########################################################################################
def run_experiment(
    selected_transforms: list,
    model_name: str,
    base_name: str = "experiment",
    extra_info: str = None,
    model_type: str = "efficientnetv2_rw_m",
    num_classes: int = 47,
    training_params: dict = None,
    frozen: bool = True,
    result_file :str = "results/efficient_net.json",
    normalize:bool = True,
    use_float:bool = False,
    resize_size: int = 256,
    pretraining = True  # Tamaño de `resize` configurable
):
    """
    Ejecuta un experimento completo: entrenamiento, validación y prueba.

    Args:
        selected_transforms (list): Transformaciones a aplicar.
        model_name (str): Nombre del modelo base.
        base_name (str): Nombre base para nombrar el experimento.
        extra_info (str): Información extra para el nombre del experimento.
        model_type (str): Tipo de modelo preentrenado.
        num_classes (int): Número de clases en la tarea.
        training_params (dict): Parámetros de entrenamiento.
        frozen (bool): Si los pesos del modelo deben ser congelados.
        result_file (str): Archivo para guardar los resultados.
        normalize (bool): Si se normalizan los datos.
        use_float (bool): Si se convierten a flotantes los datos.
        resize_size (int): Tamaño del `resize`.

    Returns:
        None
    """
    if training_params is None:
        training_params = {"learning_rate": 3e-4, "batch_size": 32, "num_epochs": 50}

    # Generar nombre de la arquitectura
    architecture_name = generate_architecture_name(model_name, base_name, extra_info)
    print(f"Running experiment: {architecture_name}")

    # Preparar datos
    train_data, val_data = transforming(
        selected_transforms = selected_transforms,
        resize_size = resize_size,
        normalize = normalize,
        use_float = use_float
    )

    # Crear el modelo
    model = FrozenNet(num_classes = num_classes, frozen = frozen, type = model_type, pretraining = pretraining)

    try:
        # Entrenar el modelo
        trained_model, train_loss, val_loss, val_acc_history = train_model(
            train_data,
            val_data,
            model = model,
            architecture_name = architecture_name,
            training_params = training_params,
            num_classes = num_classes,
        )

        # Evaluación en conjunto de prueba
        # print("\n=== Evaluación en el conjunto de prueba ===")
        # test_accuracy = testing(trained_model, test_data, training_params = training_params)

        # Calcular métricas relevantes
        best_epoch = int(np.argmin(val_loss) + 1) if val_loss else -1
        best_val_loss = float(val_loss[best_epoch - 1]) if val_loss else None
        best_train_loss = float(train_loss[best_epoch - 1]) if train_loss else None
        max_val_acc = max(val_acc_history)
        
        test_accuracy = 0
        
        # Preparar resultados
        results = {
            "architecture": architecture_name,
            "model_type": model_type,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_train_loss": best_train_loss,
            "max_val_acc": max_val_acc,
            "test_accuracy": float(test_accuracy),
            "training_params": training_params,
        }

        # Guardar resultados
        print("Results to save:", results)
        save_results_to_json(result_file, results)

    except Exception as e:
        print(f"Error durante el experimento: {e}")

    print(f"Experimento terminado para {architecture_name}")



################################################################################
def automate_training(model_names: list, selected_transforms: list, num_classes: int, 
                      training_params: dict, frozen: bool = True, result_file: str = "efficient_net.json"):
    """
    Automate training for a list of model names from timm.
    
    Args:
        model_names (list): List of model names as they appear in timm.
        selected_transforms (list): List of augmentations to apply.
        num_classes (int): Number of classes in the classification task.
        training_params (dict): Training parameters (learning rate, batch size, num epochs).
        frozen (bool): Whether to freeze the backbone during training.
        result_file (str): Path to the JSON file for saving results.
    """
    for model_name in model_names:
        # Generate architecture_name dynamically
        architecture_name = generate_architecture_name(model_name)
        
        print(f"Starting training for model: {model_name} with architecture_name: {architecture_name}")
        
        try:
            # Run the experiment
            run_experiment(
                selected_transforms = selected_transforms,
                architecture_name = architecture_name,
                model_type = model_name,
                num_classes = num_classes,
                training_params = training_params,
                frozen = frozen,
                result_file = result_file,  # Pasar el archivo de resultados
            )
        except Exception as e:
            print(f"Error occurred while training model {model_name}: {e}")
        print(f"Finished training for model: {model_name}\n")

###################################################################################################################
import json

def find_best_model(file_path: str, criterion: str = "max_val_acc"):
    """
    Encuentra el mejor modelo según el criterio proporcionado.

    Args:
        file_path (str): Ruta al archivo JSON con los resultados.
        criterion (str): Criterio para filtrar el mejor modelo. Puede ser:
                         - "max_val_acc" para el mayor accuracy de validación.
                         - "test_accuracy" para el mayor accuracy de prueba.
                         - "best_val_loss" para la menor pérdida de validación.

    Returns:
        dict: Datos del mejor modelo según el criterio.
    """
    # Leer el archivo JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    # Verificar que el criterio es válido
    valid_criteria = ["max_val_acc", "test_accuracy", "best_val_loss"]
    if criterion not in valid_criteria:
        raise ValueError(f"Criterio no válido. Usa uno de: {valid_criteria}")

    # Determinar el mejor modelo
    if criterion == "best_val_loss":
        best_model = min(data, key = lambda x: x[criterion])
    else:
        best_model = max(data, key = lambda x: x[criterion])

    return best_model

###################################################################################################3

# Define una función para inicializar pesos usando Kaiming Normal
def initialize_weights_kaiming(module: torch.nn.Module):
    if isinstance(module, nn.Linear):  # Para capas lineales
        nn.init.kaiming_normal_(module.weight, nonlinearity = 'relu')  # Inicialización Kaiming Normal
        if module.bias is not None:  # Inicializa el sesgo si existe
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):  # Para capas convolucionales
        nn.init.kaiming_normal_(module.weight, nonlinearity = 'relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)




