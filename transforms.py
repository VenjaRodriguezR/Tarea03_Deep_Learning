import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image

class Transforms:
    """
    Takes an image, converts it to RGB type needed and then applies a series of transformations
    (see get_transforms())

    Attributes:
        transform (A.compose): A.compose object with the series of transformations
    
    Example of use:
        train_transforms = A.Compose([A.Resize(256, 256), A.ToFloat(), ToTensorV2()]
        train_data = ImageFolder("path/to/train/folder", transform = Transforms(train_transforms))

    """
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image: Image.Image) -> ToTensorV2:
        # Forzar conversión a RGB si el modo no es RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convierte a numpy array y aplica las transformaciones
        image = np.array(image)
        transformed_image = self.transform(image = image)["image"]

        return transformed_image

#########################################################################################
# Catálogo de transformaciones disponibles
# WARNING, USING NORMALIZE WILL MOST LIKELY WON'T WORK FOR TRAINING, 
# IT SEEMS IT CHANGES THE TYPE OF DATA, USE A.ToFloat() instead

TRANSFORMATIONS_DIC = {
    "random_rotate": A.Rotate(limit = (-15, 15)),
    "horizontal_flip": A.HorizontalFlip(p = 0.5),
    "advanced_blur": A.AdvancedBlur(p = 0.5),
    "perspective": A.Perspective(p = 0.5),
    "clahe": A.CLAHE(p = 0.5),
    "random_border_crop": A.RandomCropFromBorders(p = 0.5),
    "fancy_pca": A.FancyPCA(p = 0.5),
    "brightness_contrast": A.RandomBrightnessContrast(p = 0.2),
    "shift_scale_rotate": A.ShiftScaleRotate(shift_limit = 0.05, scale_limit = 0.05, rotate_limit = 15, p = 0.5),
    "blur": A.GaussianBlur(blur_limit = (3, 7), p = 0.2),
    "noise": A.GaussNoise(p = 0.2),
    "perspective": A.Perspective(scale = (0.05, 0.1), p = 0.3),
    "random_crop": A.RandomCrop(224, 224, p = 1.0),
    "color_jitter": A.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.3, hue = 0.2, p = 0.7),
    "center_crop": A.CenterCrop(384, 384),
    "normalize": A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, p = 1.0),
    "resize": lambda size: A.Resize(size, size),
    "vertical": A.VerticalFlip(p = 0.5)
}

#########################################################################################

def get_transforms(selected_transforms: list, resize_size: int = 256, normalize: bool = False, use_float: bool = True) -> A.Compose:
    """
    Genera transformaciones dinámicas basadas en los parámetros dados.

    Args:
        selected_transforms (list): Lista de transformaciones a aplicar.
        resize_size (int): Tamaño del resize.
        normalize (bool): Si debe incluirse `normalize`.
        use_float (bool): Si debe incluirse `ToFloat`.

    Returns:
        A.Compose: Transformaciones combinadas.
    """
    # Transformaciones básicas iniciales con resize
    transforms = [TRANSFORMATIONS_DIC["resize"](resize_size)]

    # Añadir augmentations seleccionados
    transforms += [TRANSFORMATIONS_DIC[name] for name in selected_transforms]

    # Añadir normalización o conversión a flotante y ToTensor
    if normalize:
        transforms.append(TRANSFORMATIONS_DIC["normalize"])
    if use_float:
        transforms.append(A.ToFloat())

    transforms.append(ToTensorV2())

    return A.Compose(transforms)


#########################################################################################

def transforming(selected_transforms: list, resize_size: int = 256, normalize: bool = False, use_float: bool = True) -> tuple:
    """
    Crea los conjuntos de datos con augmentations para entrenamiento
    Aplica solo las transformaciones básicas al conjunto de validación

    Args:
        selected_transforms (list): Lista de augmentations a aplicar.
        resize_size (int): Tamaño del resize inicial.
        normalize (bool): Si se normalizan los datos.
        use_float (bool): Si se convierten a flotantes los datos.

    Returns:
        Tuple: (train_data, validation_data)
    """
    
    # Crear transformaciones
    train_augmentations = get_transforms(selected_transforms, resize_size, normalize, use_float)

    train_data = ImageFolder(
        "house_plant_species/train",
        transform = Transforms(train_augmentations),
    )

    validation_data = ImageFolder(
        "house_plant_species/validation",
        transform = Transforms(get_transforms([], resize_size, normalize, use_float)) )

    return train_data, validation_data


