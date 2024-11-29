#Preparemos algoritmo generalizado de transfomaciones con Albumentations

#CATALOGO ESTILO DICCIONARIO DE LAS TRANSFORMACIONES
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torch.utils.data import random_split

class Transforms:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image):
        # Imprime el modo de la imagen antes de cualquier transformación
        #print(f"Modo de imagen original: {image.mode}")

        # Forzar conversión a RGB si el modo no es RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            #print("La imagen fue convertida a RGB.")

        # Convierte a numpy array y aplica las transformaciones
        image = np.array(image)
        transformed_image = self.transform(image = image)["image"]

        # Devuelve la imagen transformada
        return transformed_image


# Catálogo de transformaciones disponibles
TRANSFORMATIONS_DIC = {
    "random_rotate": A.RandomRotate90(p = 0.5),
    "horizontal_flip": A.HorizontalFlip(p = 0.5),
    "brightness_contrast": A.RandomBrightnessContrast(p = 0.2),
    "shift_scale_rotate": A.ShiftScaleRotate(shift_limit = 0.05, scale_limit = 0.05, rotate_limit = 15, p = 0.5),
    "blur": A.GaussianBlur(blur_limit = (3, 7), p = 0.2),
    "noise": A.GaussNoise(p = 0.2),
    "perspective": A.Perspective(scale = (0.05, 0.1), p = 0.3),
    "random_crop": A.RandomCrop(224, 224, p = 1.0),
    "color_jitter": A.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1, p = 0.5),
}

basic_transforms = A.Compose([A.Resize(256, 256), A.ToFloat(), ToTensorV2()])

# ES IMPORTANTE NOTAR QUE SE RECOMIENDA PARTIR SIEMPRE CON resize
# Ahora necesitamos una funcion que genere los A.Compose([lista de transfomaciones])

def get_transforms(selected_transforms):

    transforms = [A.Resize(256, 256)]
    transforms += [TRANSFORMATIONS_DIC[name] for name in selected_transforms]
    transforms.append(A.ToFloat())
    transforms.append(ToTensorV2())
    return A.Compose(transforms)

# Preparar datos
def transforming(selected_transforms):
    """
    Crea los conjuntos de datos con augmentations para entrenamiento y validación.
    Aplica solo las transformaciones básicas al conjunto de prueba.
    """
    # Dataset original sin transformaciones para dividir
    dataset = ImageFolder("house_plant_species/validation")
    
    # Dividir en validación y prueba
    test_size = int(0.2 * len(dataset))
    validation_size = len(dataset) - test_size
    validation_data, test_data = random_split(dataset, [validation_size, test_size])
    
    # Crear transformaciones
    train_augmentations = get_transforms(selected_transforms)

    # Aplicar transformaciones
    train_data = ImageFolder(
        "house_plant_species/train",
        transform = Transforms(train_augmentations),
    )

    validation_data.dataset.transform = Transforms(basic_transforms)  # Cambiar transformación en validación
    test_data.dataset.transform = Transforms(basic_transforms)  # Transformación básica para prueba

    return train_data, validation_data, test_data