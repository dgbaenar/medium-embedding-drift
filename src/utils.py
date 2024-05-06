import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import cv2
import tensorflow as tf


def get_worst_preds(batch_gen, Y_pred, threshold):
    rotten = []
    batch_gen.reset()
    for i, c in enumerate(batch_gen.classes):
        if Y_pred[i][c] < threshold:
            pred_c = np.argmax(Y_pred[i])
            rotten.append((batch_gen.filepaths[i], c, pred_c, Y_pred[i][pred_c]))
    return rotten


def show_worst_preds(batch_gen, Y_pred, class_names, threshold=0.1):
    """
    This function shows the images for which the model gave a probability of less than the given threshold

    Arguments:
        batch_generator {DirectoryIterator} -- Batch generator created by ImageDataGenerator.flow_from_directory, for example
        Y_pred {numpy.array} -- An array of the outputs of the model
        class_names {[str]} -- a list of strings containing the names of the classes in correct order
        threshold {float} -- Probability threshold
    """

    preds = get_worst_preds(batch_gen, Y_pred, threshold)
    print(f"Got {len(preds)} worst predictions")
    plt.figure()
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()
    for pred_data, ax in zip(preds, axes):
        img = np.array(Image.open(pred_data[0]))
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'Pred: {class_names[pred_data[2]]} ({pred_data[3]:.2f}) - {class_names[pred_data[1]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def crop_resize_image(img, target=(224, 224)):
    """ Crop and resize a PIL Image to a squared target size """
    width, height = img.size
    crop_size = min(width, height)

    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img.resize(target, Image.ANTIALIAS)


def list_jpg_files(directory):
    files = []
    # Recorre el directorio y subdirectorios
    for dirpath, dirnames, filenames in os.walk(directory):
        # Filtra y añade solo los archivos .jpg
        jpg_files = [os.path.join(dirpath, file) for file in filenames if file.lower().endswith('.jpg')]
        files.extend(jpg_files)
    return files


def simulate_covariate_shift(image, adjustment_factor, shift_x=10, shift_y=10):
    # Get the height and width of the image
    height, width = image.shape[:2]

    image[:, :, 2] = image[:, :, 2] * adjustment_factor  # Red channel

    # Create a transformation matrix for the shift
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Apply the translation to the image
    shifted_image = cv2.warpAffine(image, M, (width, height))

    return shifted_image


def apply_covariate_shift_to_images(output_directory, drifted_images_paths):
    # Verificar si el directorio existe y crearlo si no es así
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for index, file in enumerate(drifted_images_paths):
        magnitude = index / 10
        if magnitude > 3:
            magnitude = 3
        image = cv2.imread(file)
        drifted_image = simulate_covariate_shift(image, adjustment_factor=magnitude, shift_x=int(magnitude), shift_y=int(magnitude))
        
        # Construir la nueva ruta de archivo en el directorio de salida
        output_path = os.path.join(output_directory, os.path.basename(file))
        
        # Guardar la imagen modificada en el nuevo directorio
        cv2.imwrite(output_path, drifted_image)


def decode_img(img):
    """Decodifica y procesa una imagen para ajustarla a las necesidades del modelo."""
    img = tf.image.decode_jpeg(img, channels=3)  # Ajusta channels según tus imágenes
    img = tf.image.resize(img, [224, 224])
    return img


def process_path(file_path):
    """Lee y procesa una imagen desde una ruta de archivo."""
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def create_image_dataset(image_paths, batch_size=32):
    """
    Crea un tf.data.Dataset a partir de una lista de rutas de imágenes.

    Args:
    image_paths (list of str): Lista de rutas a las imágenes.
    batch_size (int): Tamaño del lote para el procesamiento del dataset.

    Returns:
    tf.data.Dataset: Dataset listo para ser utilizado para inferencia o entrenamiento.
    """
    list_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    images_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    images_ds = images_ds.batch(batch_size)
    return images_ds