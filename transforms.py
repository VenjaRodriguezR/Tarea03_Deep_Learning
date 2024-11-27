#Preparemos algoritmo generalizado de transfomaciones con Albumentations

#CATALOGO ESTILO DICCIONARIO DE LAS TRANSFORMACIONES
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


class Transforms:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image):
        image = np.array(image)
        return self.transform(image=image)["image"]
    
# Catálogo de transformaciones disponibles
TRANSFORMATIONS_DIC = {
    "random_rotate": A.RandomRotate90(p=0.5),
    "resize": A.Resize(256, 256),
    "horizontal_flip": A.HorizontalFlip(p=0.5),
    "brightness_contrast": A.RandomBrightnessContrast(p=0.2),
    "shift_scale_rotate": A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    "blur": A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    "noise": A.GaussNoise(p=0.2),
    "perspective": A.Perspective(scale=(0.05, 0.1), p=0.3),
    "normalize": A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "random_crop": A.RandomCrop(224, 224, p=1.0),
    "color_jitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
}
# ES IMPORTANTE NOTAR QUE SE RECOMIENDA PARTIR SIEMPRE CON resize
# Ahora necesitamos una funcion que genere los A.Compose([lista de transfomaciones])

def get_transforms(selected_transforms):

    transforms = [TRANSFORMATIONS_DIC[name] for name in selected_transforms]
    transforms.append(A.ToFloat())
    transforms.append(ToTensorV2())
    return A.Compose(transforms)

def transforming(selected_transforms):
    # Crear transformaciones
    train_augmentations = get_transforms(selected_transforms)
    validation_augmentations = get_transforms(selected_transforms)

    # Datasets
    train_data = ImageFolder(
        "house_plant_species/train",
        transform=Transforms(train_augmentations),
    )
    validation_data = ImageFolder(
        "house_plant_species/validation",
        transform=Transforms(validation_augmentations),
    )
    print(f"Elementos en Entrenamiento: {len(train_data)}")
    print(f"Elementos en Validación: {len(validation_data)}")

    return train_data, validation_data