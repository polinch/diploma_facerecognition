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


# create face matrix
@jit(float32(int32))
def face_matrix(count):

    face_matr = np.zeros((count, STANDART_SHAPE))
    for i in range(count):
        face = image_to_array("image" + i + ".jpg")
        for j in range(STANDART_SHAPE):
            face_matr[i][j] = face[j]

    return face_matr


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


# calculate pca
@jit(float32(int32, float32))
def pca(temp_transp, face_matr):

    for i in range(STANDART_SHAPE):
        for j in range(face_matr.shape[0]):
            temp_transp[i][j] = face_matr[j][i]

    cov_matr = np.matmul(face_matr, temp_transp)

    return cov_matr


temp_faces = face_matrix(4)
temp_average = np.zeros(STANDART_SHAPE)
faces = average_face(temp_average, temp_faces)
temp_tr = np.zeros((STANDART_SHAPE, 4))
temp_cov = pca(temp_tr, faces)
print(temp_cov.shape)
