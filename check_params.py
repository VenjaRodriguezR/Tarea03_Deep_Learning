from torchinfo import summary
import timm

### No modifique este archivo ###

model_timm = timm.create_model(
    "resnet50",
    pretrained=True,
)


def check_params(model, model_name):
    print(model_name)
    output = summary(model, input_size=(1, 3, 224, 224), verbose=0)
    for line in str(output).split("\n"):
        if "Total params:" in line:
            print(line)
            return line


def convert_to_int(string):
    try:
        return int(string.split(":")[1].strip().replace(",", ""))
    except:
        return 0


def compare_model_params(model):
    my_model_params = check_params(model, model_name="Tu modelo: ")
    print("======================================")
    timm_model_params = check_params(
        model_timm, model_name="Modelo de Timm: "
    )
    if convert_to_int(my_model_params) != convert_to_int(
        timm_model_params
    ):
        raise ValueError(
            "Su implementación es incorrecta: El número de parámetros no calza."
        )
    else:
        print("Felicitaciones: Su implementación es correcta.")
