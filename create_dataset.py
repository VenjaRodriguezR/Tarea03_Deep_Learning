import shutil
import zipfile
from pathlib import Path
import glob
import argparse

### No modifique este archivo ###


class DataCreator:
    def __init__(self, zip_name):
        self.path = zip_name

    def create_data(self, train_perc):
        print("Descomprimiendo Archivos...")
        self.extract_zip()
        print("Creando datos de Entrenamiento...")
        train_paths = self.create_file_paths(perc=train_perc)
        self.move_to_folder(train_paths)
        print("Creando datos de Validación...")
        val_paths = self.create_file_paths(perc=1)
        self.move_to_folder(val_paths, folder="validation")
        self.remove_empty_folders()
        print(
            f"Datos creados exitosamente: {train_perc*100:.1f}% para Entrenamiento y {(1-train_perc)*100:.1f}% para Validación."
        )

    def extract_zip(self):
        with zipfile.ZipFile(f"{self.path}.zip", "r") as f:
            f.extractall(self.path)

    def create_file_paths(self, perc):
        paths = []
        folders = glob.glob(f"{self.path}/*")
        for p in folders:
            if "train" not in p and "validation" not in p:
                files = sorted(glob.glob(f"{p}/*.*"))
                n_files = int(len(files) * perc)
                paths.extend(files[:n_files])

        return paths

    def move_to_folder(self, paths_to_move, folder="train"):
        new_paths = []
        for path in paths_to_move:
            text = path.split("/")
            text.insert(1, folder)
            new_paths.append("/".join(text))

        for path in new_paths:
            Path(path).parent.mkdir(exist_ok=True, parents=True)

        for ptm, np in zip(paths_to_move, new_paths):
            shutil.move(ptm, np)

    def remove_empty_folders(self):
        for p in glob.glob(f"{self.path}/*"):
            if "train" not in p and "validation" not in p:
                shutil.rmtree(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI para descomprimir los archivos. NO INCLUYA LA EXTENSIÓN .zip EN EL PATH SOLICITADO."
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        type=str,
        help="Path al .zip. No incluya la extensión.",
    )
    parser.add_argument(
        "-tp",
        "--training_perc",
        required=True,
        type=float,
        help="Porcentaje de datos a usar en Entrenamiento",
        default=0.8,
    )
    args = parser.parse_args()
    data = DataCreator(args.path)
    data.create_data(train_perc=args.training_perc)
