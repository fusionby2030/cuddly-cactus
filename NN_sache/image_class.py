# import the necessary packages

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import io



def prepare_image(image, target):
    """
    Prepare Image Function:
    Accepts Input Image
    Converts the mode to RGB
    Resizes to 224x224 (RESNET)
    Image to array with scaling
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def decode_predictions(predictions):
    results = imagenet_utils.decode_predictions(predictions)
    return results



def open_image(image):
    image = Image.open(io.BytesIO(image))
    return image
