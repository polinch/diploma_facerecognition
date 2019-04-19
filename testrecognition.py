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
@jit()
def face_matrix(count, filename):

    face_matr = np.zeros((STANDART_SHAPE, count))
    for i in range(count):
        face = image_to_array(filename + str(i) + ".jpg")
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
def pca(face_matr):

    temp_transp = np.zeros((face_matr.shape[1], STANDART_SHAPE))
    for i in range(face_matr.shape[1]):
        for j in range(STANDART_SHAPE):
            temp_transp[i][j] = face_matr[j][i]

    cov_matr = np.matmul(temp_transp, face_matr)
    eigvals, eigvecs = np.linalg.eig(cov_matr)

    shape = eigvecs.shape[0]
    new_eigvecs = np.zeros((shape, STANDART_SHAPE))

    for i in range(shape):
        for j in range(shape):
            for k in range(STANDART_SHAPE):
                new_eigvecs[i][k] += eigvecs[i][j] * face_matr[k][j]

    return new_eigvecs

# projected faces into face space - матрица лиц уже нормирована по среднему лицу
@jit()
def transform_new_face(face, face_matr, eigenvecs):

    count = eigenvecs.shape[0]
    weight_vec = np.zeros(count)
    for i in range(count):
        temp = eigenvecs[i, :]
        weight_vec[i] = np.matmul(temp, face)

    return weight_vec


class CalculatePCA(object):

    def __init__(self, count_train, filename_train, filename_test):
        self.count_train_set = count_train
        self.count_test_set = 1
        self.av_face = np.zeros(STANDART_SHAPE)
        self.face_matr_train = face_matrix(self.count_train_set, filename_train)
        self.face_matr_test = face_matrix(self.count_test_set, filename_test)
        self.eigvecs = np.zeros((self.count_train_set, STANDART_SHAPE))

    def calc_pca(self):
        self.face_matr_train = average_face(self.av_face, self.face_matr_train)
        self.eigvecs = pca(self.face_matr_train)

        return self.eigvecs

    def transform_faces(self):
        self.face_matr_test = average_face(self.av_face, self.face_matr_test)

        return transform_new_face(self.face_matr_test, self.face_matr_train, self.eigvecs)


test = CalculatePCA(3, "image", "image0")
test.calc_pca()
temp_test = test.transform_faces()
print(temp_test)





