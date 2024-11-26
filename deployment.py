import gradio as gr
import torch
import timm
from torchvision import transforms, datasets
from PIL import Image

# Obtener etiquetas de clases desde las carpetas del dataset
def get_class_labels(data_path):
    dataset = datasets.ImageFolder(data_path)
    return {v: k for k, v in dataset.class_to_idx.items()}

# Ruta al dataset
data_path = "house_plant_species/train"  # Ajusta esta ruta según tu estructura de directorios

# Cargar etiquetas de clases desde las carpetas
class_labels = get_class_labels(data_path)

# Función para cargar el modelo
def load_model(use_checkpoint, model_name="resnet18", checkpoint_path=None, num_classes=len(class_labels), in_channels=3):
    if use_checkpoint:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=in_channels)
        if checkpoint_path is None:
            raise ValueError("Debe proporcionar un checkpoint_path si use_checkpoint es True")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        print(f"Modelo cargado desde checkpoint: {checkpoint_path}")
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=in_channels)
        print(f"Modelo preentrenado {model_name} cargado desde timm")
    model.eval()
    return model

# Configuración inicial
use_checkpoint = False  # Cambia a True si deseas usar un checkpoint
#checkpoint_path = "resnet18_best_model.pth"  # Ruta al checkpoint (opcional)
model_name = "resnet18"  # Nombre del modelo
num_classes = len(class_labels)  # Número de clases
in_channels = 3  # Número de canales de entrada

# Cargar el modelo
model = load_model(
    use_checkpoint=use_checkpoint,
    model_name=model_name,
    checkpoint_path = None,
    num_classes=num_classes,
    in_channels=in_channels,
)

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Función de predicción
def predict_species(image):
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_idx = probabilities.argmax().item()
    return f"🌱 Predicción: {class_labels[top_idx]} (Confianza: {probabilities[top_idx]:.2f})"

# Descripción de la aplicación
description = """
### Plantifier 🌿
Bienvenido a **Plantifier**, tu clasificador de plantas de interior. Sube una imagen de una planta y el modelo intentará identificar su especie.
"""

# Crear la interfaz de Gradio con el tema 'small_and_pretty'
interface = gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plantifier 🌿",
    description=description,
    theme='JohnSmith9982/small_and_pretty'
)

# Ejecutar la aplicación con Gradio
if __name__ == "__main__":
    interface.launch(share=True)
