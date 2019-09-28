import numpy as np
from PIL import Image
from numba import jit, float32, int32


STANDART_SIZE = (180, 200)
STANDART_SHAPE = 2 * STANDART_SIZE[0] * STANDART_SIZE[1]


# open image and convert to array
def image_to_array(filename):

    img = Image.open(filename)
    img = img.resize(STANDART_SIZE)
    data = np.asarray(img)
    temp_shape = 2 * data.shape[0] * data.shape[1]
    data = data.reshape(1, temp_shape)

    return data


# create face matrix
@jit()
def face_matrix(count, filename):

    face_matr = np.zeros((STANDART_SHAPE, count))
    for i in range(count):
        face = image_to_array(filename + str(i) + ".png")
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

    return [face_matr, av_face]


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

    # projected faces from training set into face space
    weight_matr = np.matmul(new_eigvecs, face_matr)

    return [new_eigvecs, weight_matr]

# projected faces into face space - вектор лиц уже нормирован по среднему лицу
# сравнивать два параметра, чтобы точнее идентифицировать лицо - как сравнивать второй параметр?
@jit()
def transform_new_face(face, eigenvecs, weight_matr):

    count = eigenvecs.shape[0]
    weight_vec = np.zeros(count)
    for i in range(count):
        temp = eigenvecs[i, :]
        weight_vec[i] = np.matmul(temp, face)

    # distance from face to face space
    eps_projected = np.zeros(count)
    for i in range(count):
        eps_projected[i] = np.linalg.norm(weight_vec - weight_matr[:, i])

    faces_tilda = np.zeros(STANDART_SHAPE)
    for i in range(count):
        for j in range(STANDART_SHAPE):
            faces_tilda[j] += weight_vec[i] * eigenvecs[i][j]
    faces_tilda.reshape(STANDART_SHAPE)

    temp = np.zeros(STANDART_SHAPE)
    for i in range(STANDART_SHAPE):
        temp[i] = face[i] - faces_tilda[i]
    eps_face = np.linalg.norm(temp)

    # тут надо как-то выбрать значение тета, с которым сравнивать расстояние
    result = np.array([0, eps_projected[0]])
    for i in range(1, count):
        if eps_projected[i] < result[1]:
            result[0] = i
            result[1] = eps_projected[i]

    return result


class CalculatePCA(object):

    def __init__(self, count_train, filename_train):
        self.count_train_set = count_train
        self.count_test_set = 1
        self.av_face = np.zeros(STANDART_SHAPE)
        self.face_matr_train = face_matrix(self.count_train_set, filename_train)
        self.face_matr_test = np.zeros((1, STANDART_SHAPE))
        self.eigvecs = np.zeros((self.count_train_set, STANDART_SHAPE))
        self.weight_matr = np.zeros((self.count_train_set, self.count_train_set))

    def calc_pca(self):
        temp_result = average_face(self.av_face, self.face_matr_train)
        self.av_face = temp_result[1]
        self.face_matr_train = temp_result[0]
        temp = pca(self.face_matr_train)
        self.eigvecs = temp[0]
        self.weight_matr = temp[1]
        # write matrix pca in file
        file = open('matr_pca.txt', 'w')
        for i in range(self.eigvecs.shape[0]):
            for j in range(self.eigvecs.shape[1]):
                file.write(str(self.eigvecs[i][j]) + '\t')
            file.write('\r\n')
        file.close()

        return self.eigvecs

    def pca_without_train(self, count_train, train_filename):
        self.count_train_set = count_train
        self.face_matr_train = face_matrix(count_train, train_filename)
        temp_result = average_face(self.av_face, self.face_matr_train)
        self.av_face = temp_result[1]
        self.face_matr_train = temp_result[0]
        temp_pca = np.genfromtxt('matr_pca.txt', dtype=np.float, delimiter='\t')
        shape_pca = temp_pca.shape
        new_pca = np.zeros((shape_pca[0], shape_pca[1] - 1))
        for i in range(shape_pca[0]):
            # тут -1, потому что записываем в файл пустую строку после последней строки матрицы с собственными векторами
            for j in range(shape_pca[1] - 1):
                self.eigvecs[i][j] = temp_pca[i][j]
        #         new_pca[i][j] = temp_pca[i][j]
        # self.eigvecs = new_pca
        self.weight_matr = np.matmul(self.eigvecs, self.face_matr_train)

        return

    def transform_faces(self, test_filename):
        self.face_matr_test = face_matrix(self.count_test_set, test_filename)
        # normalizing test image
        for i in range(STANDART_SHAPE):
            self.face_matr_test[i] -= self.av_face[i]
        result = transform_new_face(self.face_matr_test, self.eigvecs, self.weight_matr)

        return int(result[0])
