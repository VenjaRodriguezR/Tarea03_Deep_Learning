import matplotlib.pyplot as plt


def plot_number(idx, data):

    plt.imshow(data[idx]["X"].numpy().reshape(28, 28), cmap = "gray")
    plt.title(f"Etiqueta: {data[idx]['y']}")
    plt.show()


def plot_training_curves(train_loss, validation_loss, n_epochs, title = ""):

    plt.plot(range(1, n_epochs + 1), train_loss, label = "Train Loss")
    plt.plot(range(1, n_epochs + 1), validation_loss, label = "Validation Loss")
    plt.title(title)
    plt.legend()
    plt.show()

## Es importante cambiar el orden de los canales para poder mostrar la imagen...
def plot_images(idx, data):
    plt.imshow(data[idx][0].permute(1, 2, 0))
    class_label = data[idx][1]
    plt.title(data.classes[class_label])
    plt.axis("off")
    print(data[idx][0].shape)