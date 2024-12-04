import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from collections import defaultdict
import os
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from torchvision.datasets import ImageFolder
import timm
from torch.utils.data import DataLoader

SEED = 0


#############################################################################

# SEETS SEED FOR REPRODUCIBILITY
def set_seed(seed: int = SEED) -> None:

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


#############################################################################
# COUNTS INSTANCES OF PLANT IMAGES BY CLASS

def count_instances_per_class(image_folder: ImageFolder) -> defaultdict:

    class_counts = defaultdict(int)

    for _, class_idx in image_folder.samples:
        class_counts[image_folder.classes[class_idx]] += 1
    return class_counts


#############################################################################

# Automatización del nombre de la arquitectura
def generate_architecture_name(model_name: str, base_name: str = "augmented", extra_info: str = None) -> str:
    """
    Genera un nombre de arquitectura dinámico basado en el modelo, un nombre base y opcionalmente información extra.
    Este nombre aparecerá en Wanb y en el nombre de la carpeta en checkpoints

    Args:
        model_name (string): The name you will give your model
        base_name (string): Extension of model_name
        extra_info (string): Extension of model_name

    Returns:
        String: f"{model_name}_{base_name}"
    """
    if extra_info:
        return f"{model_name}_{base_name}_{extra_info}"
    return f"{model_name}_{base_name}"

#############################################################################

def visualize_transformed_images(dataset: ImageFolder, num_images: int = 5, figsize: tuple = (15, 15)) -> None:
    """
    Visualiza imágenes transformadas del dataset.

    Args:
        dataset: Dataset transformado (por ejemplo, de `ImageFolder`).
        num_images (int): Número de imágenes a visualizar.
        figsize (tuple): Tamaño del gráfico.
    """
    # Seleccionar imágenes aleatorias
    indices = random.sample(range(len(dataset)), num_images)

    # Crear figura
    fig, axes = plt.subplots(1, num_images, figsize = figsize)

    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]
        # Convertir el tensor a numpy para mostrarlo con matplotlib
        img = img.permute(1, 2, 0).numpy()
        # Mostrar imagen
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Class: {dataset.classes[label]}")

    plt.tight_layout()
    plt.show()

#############################################################################

def calculate_class_accuracies(model: timm.models, dataloader: DataLoader, class_labels: dict , device: str = 'cuda') -> dict:
    """
    Calcula la precisión por clase para un modelo dado.

    Args:
        model: El modelo cargado.
        dataloader: DataLoader con datos de validación o prueba.
        class_labels: Diccionario que mapea índices de clases a nombres.
        device: Dispositivo para la evaluación ('cpu' o 'cuda').

    Returns:
        dict: Precisión por clase.
    """
    # Inicializar contadores
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    # Cambiar el modelo al modo de evaluación
    model.eval()
    model.to(device)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim = 1)

            # Acumular correctos y totales por clase
            for target, prediction in zip(targets, predictions):
                total_per_class[target.item()] += 1
                if target == prediction:
                    correct_per_class[target.item()] += 1

    # Calcular precisión por clase
    class_accuracies = {
        class_labels[class_idx]: correct_per_class[class_idx] / total_per_class[class_idx]
        for class_idx in class_labels
    }

    return class_accuracies
