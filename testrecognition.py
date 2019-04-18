import numpy as np
from PIL import Image


STANDART_SIZE = (50, 50)
STANDART_SHAPE = 3 * STANDART_SIZE[0] * STANDART_SIZE[1]


# open image and convert to array
def image_to_array(filename):

    img = Image.open(filename)
    img = img.resize(STANDART_SIZE)
    data = np.asarray(img)
    temp_shape = data.shape[0] * data.shape[1] * data.shape[2]
    data = data.reshape(1, temp_shape)

    return data


