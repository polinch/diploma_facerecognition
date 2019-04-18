import numpy as np
from PIL import Image
from numba import jit, float32, int32


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


# the average face - normalized face matrix
@jit(float32(int32, float32))
def average_face(av_face, face_matrix):

    count_of_face = face_matrix.shape[0]
    for i in range(count_of_face):
        for j in range(STANDART_SHAPE):
            av_face[j] += face_matrix[i][j]
            av_face[j] /= count_of_face

    for i in range(count_of_face):
        for j in range(STANDART_SHAPE):
            face_matrix[i][j] -= av_face[j]

    return face_matrix


