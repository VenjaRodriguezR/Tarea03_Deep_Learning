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

# Plots training curves
def plot_training_curves(train_loss: list, validation_loss: list, n_epochs: int, title: str = "") -> None:

    plt.plot(range(1, n_epochs + 1), train_loss, label = "Train Loss")
    plt.plot(range(1, n_epochs + 1), validation_loss, label = "Validation Loss")
    plt.title(title)
    plt.legend()
    plt.show()

#############################################################################

# It lets you see the images from its numeric representation
def plot_images(idx: int, data: ImageFolder) -> None:

    plt.imshow(data[idx][0].permute(1, 2, 0)) # (H, W, C)
    class_label = data[idx][1]
    plt.title(data.classes[class_label])
    plt.axis("off")
    print(data[idx][0].shape)

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
# OBTAINS IMAGES WITH AN SPECIFIC FORMAT

def get_images_by_format(base_dir: str, target_extension = ".gif") -> list:

    images_by_format = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(target_extension):
                img_path = os.path.join(root, file)
                images_by_format.append(img_path)

    return images_by_format

#############################################################################

def plot_images_for_class(class_name: str, base_path: str, n: int = 10, figsize: tuple = (10, 10)) -> None:
    """
    Genera un gráfico con imágenes de una clase específica.

    Parameters:
    - class_name (str): Nombre de la clase que deseas graficar.
    - base_path (str): Ruta base donde están las carpetas de las clases.
    - n (int): Número máximo de imágenes a mostrar.
    - figsize (tuple): Tamaño del gráfico.
    """
    # Ruta completa de la clase
    class_path = os.path.join(base_path, class_name)

    # Verificar que la ruta exista
    if not os.path.exists(class_path):
        print(f"La clase '{class_name}' no existe en la ruta '{base_path}'.")
        return

    # Listar imágenes de la clase
    images = [os.path.join(class_path, img) for img in os.listdir(class_path) 
              if img.lower().endswith(('jpg', 'png', 'jpeg', "gif", "jpe", "webp", "jfif"))]
    
    if len(images) == 0:
        print(f"No se encontraron imágenes para la clase '{class_name}'.")
        return
    
    # Tomar las primeras `n` imágenes
    images = images[ : n]

    # Crear la figura y el grid
    fig = plt.figure(figsize = figsize)

    grid = ImageGrid(fig, 111, nrows_ncols = (1, len(images)), axes_pad = 0.4)

    # Añadir imágenes al grid

    for img_path, ax in zip(images, grid):
        ax.axis('off')  # Quitar los ejes
        img = Image.open(img_path)
        ax.imshow(img)
    

    plt.show()

#############################################################################

# Función para mostrar imágenes de todas las clases
def plot_images_for_all_classes(base_path: str, n: int = 5, figsize: tuple = (10, 10)) -> None:
    """
    Genera gráficos con imágenes de todas las clases en un dataset.

    Parameters:
    - base_path (str): Ruta base donde están las carpetas de las clases.
    - n (int): Número máximo de imágenes a mostrar por clase.
    - figsize (tuple): Tamaño del gráfico para cada clase.
    """
    # Listar carpetas (clases) en la ruta base
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for class_name in classes:
        print(f"Mostrando imágenes para la clase: {class_name}")
        plot_images_for_class(class_name, base_path, n = n, figsize = figsize)

#############################################################################

def count_images_by_size(base_dir = "house_plant_species/train"):

    size_count = defaultdict(int)
    extensions = defaultdict(int)

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        size = img.size  # (width, height)
                        size_count[size] += 1
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
            
            _, ext = os.path.splitext(file)
            if ext:
                extensions[ext.lower()] += 1

    return size_count, extensions

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
            predictions = torch.argmax(outputs, dim=1)

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
