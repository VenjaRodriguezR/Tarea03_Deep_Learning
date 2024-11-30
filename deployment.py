import gradio as gr
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import torch.nn as nn
import numpy as np


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

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_layers(x)
        return x
    


# Obtener etiquetas de clases desde las carpetas del dataset
def get_class_labels(data_path: str) -> dict:

    dataset = datasets.ImageFolder(data_path)

    return {v: k for k, v in dataset.class_to_idx.items()}

# Ruta al dataset
data_path = "house_plant_species/validation"  # Ajusta esta ruta según tu estructura de directorios

# Cargar etiquetas de clases desde las carpetas
class_labels = get_class_labels(data_path)

# Función para cargar el modelo

def load_model(use_checkpoint: bool, model_name: str = "efficientnet_b5.sw_in12k", checkpoint_path: str = None,
                num_classes: str = len(class_labels), in_channels: int = 3) -> timm.models:

    if use_checkpoint:
        model = FrozenNet()

        if checkpoint_path is None:
            raise ValueError("Debe proporcionar un checkpoint_path si use_checkpoint es True")
        
        model.load_state_dict(torch.load(checkpoint_path, map_location = torch.device('cpu')))

        print(f"Modelo cargado desde checkpoint: {checkpoint_path}")
    else:
        model = timm.create_model(model_name, pretrained = True, num_classes = num_classes, in_chans = in_channels)
        print(f"Modelo preentrenado {model_name} cargado desde timm")
    model.eval()
    return model


# Cargar el modelo
model = load_model(
    use_checkpoint = True,
    model_name = "efficientnet_b5.sw_in12k",
    checkpoint_path = "checkpoint_epoch_9.pth",
    num_classes = 47 ,
    in_channels = 3,
)

# Transformaciones de imagen
transform = A.Compose([
    A.Resize(416, 416),
    A.ToFloat(),
    ToTensorV2(),

])

# Función de predicción

def predict_species(image):
    image = image = transform(image = np.array(image))["image"].unsqueeze(0)  # Añadir dimensión de batch

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim = 0)
        top_idx = probabilities.argmax().item()
    return f"🌱 Predicción: {class_labels[top_idx]} (Confianza: {probabilities[top_idx]:.2f})"

# Descripción de la aplicación
description = """
### Plantifier 🌿
Bienvenido a **Plantifier**, tu clasificador de plantas de interior. Sube una imagen de una planta y el modelo intentará identificar su especie.
"""

# Crear la interfaz de Gradio con el tema 'small_and_pretty'

interface = gr.Interface(
    fn = predict_species,
    inputs = gr.Image(type = "pil"),
    outputs = "text",
    title = "Plantifier 🌿",
    description = description,
    theme = 'JohnSmith9982/small_and_pretty'
)

# Ejecutar la aplicación con Gradio
if __name__ == "__main__":
    interface.launch(share = True)
