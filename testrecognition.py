import numpy as np
from PIL import Image
from numba import jit, float32, int32


STANDART_SIZE = (100, 100)
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

    face_matr = np.zeros((STANDART_SHAPE, count))
    for i in range(count):
        face = image_to_array("image" + str(i) + ".jpg")
        for j in range(STANDART_SHAPE):
            face_matr[j][i] = face[0][j]

    return face_matr


# the average face - normalized face matrix
@jit(float32(int32, float32))
def average_face(av_face, face_matr):

    count_of_face = face_matr.shape[1]
    for i in range(STANDART_SHAPE):
        for j in range(count_of_face):
            av_face[i] += face_matr[i][j]
            av_face[i] /= count_of_face

    for i in range(STANDART_SHAPE):
        for j in range(count_of_face):
            face_matr[i][j] -= av_face[i]

    return face_matr


# calculate pca
@jit(float32(int32, float32))
def pca(temp_transp, face_matr):

    for i in range(face_matr.shape[1]):
        for j in range(STANDART_SHAPE):
            temp_transp[i][j] = face_matr[j][i]

    cov_matr = np.matmul(temp_transp, face_matr)

    return cov_matr


test_faces = face_matrix(4)
test_avface = np.zeros(STANDART_SHAPE)
change_faces = average_face(test_avface, test_faces)
test_transp = np.zeros((4, STANDART_SHAPE))
test_cov = pca(test_transp, change_faces)
print(test_cov.shape)
