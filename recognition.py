import numpy as np
from PIL import Image
from numba import jit, float32, int32


STANDART_SIZE = (180, 200)
STANDART_SHAPE = 2 * STANDART_SIZE[0] * STANDART_SIZE[1]


# TODO exception?
def convert_image_to_array(filename):

    try:
        image = Image.open(filename).convert('LA')
    except FileNotFoundError:
        print("File not found")
    else:
        photo_array = np.asarray(image)
        photo_array.reshape(STANDART_SHAPE)
        return photo_array


def create_face_matrix(count_of_photo, filename, face_matrix):

    if count_of_photo > 1:
        for i in range(0, count_of_photo):
            photo_array = convert_image_to_array(filename + str(i) + ".png")
            for j in range(0, STANDART_SHAPE):
                face_matrix[j][i] = photo_array[j]

    elif count_of_photo == 1:
        photo_array = convert_image_to_array(filename)
        for i in range(0, STANDART_SHAPE):
            face_matrix[i] = photo_array[i]

    return face_matrix


@jit(float32(int32, float32, float32))
def calculate_average_face(count_of_photo, average_face, face_matrix):

    for i in range(0, count_of_photo):
        for j in range(0, STANDART_SHAPE):
            average_face[i] += face_matrix[i][j]
        average_face[i] /= count_of_photo

    return average_face


@jit(float32(int32, float32, float32))
def normalize_face_matrix(count_of_photo, average_face, face_matrix):

    if count_of_photo > 1:
        for i in range(0, STANDART_SHAPE):
            for j in range(0, count_of_photo):
                face_matrix[i][j] -= average_face[i]

    elif count_of_photo == 1:
        for i in range(0, STANDART_SHAPE):
            face_matrix[i] -= average_face[i]

    return face_matrix


@jit(float32(float32, float32, float32, float32))
def count_pca(face_matrix, eigen_vectors, full_eigen_vectors, weight_matrix):

    count_of_photo = face_matrix.shape[1]
    for i in range(0, count_of_photo):
        for j in range(0, count_of_photo):
            for k in range(0, STANDART_SHAPE):
                full_eigen_vectors[i][k] += eigen_vectors[i][j] * face_matrix[k][j]

    np.matmul(full_eigen_vectors, face_matrix, weight_matrix)

    return [weight_matrix, full_eigen_vectors]


@jit(float32(float32, float32, float32, float32, int32, int32))
def search_by_photo(photo_for_search, weight_vector, weight_matrix, eigen_vectors, distances, result):

    temp_shape = eigen_vectors.shape[0]
    for i in range(0, temp_shape):
        np.matmul(eigen_vectors[i, :], photo_for_search, weight_vector[i])

    for i in range(0, temp_shape):
        distances[i] = np.linalg.norm(weight_vector - weight_matrix[:, i])

    for i in range(0, temp_shape):
        if distances[i] > result[1]:
            result[0] = i
            result[1] = distances[i]

    return int(result[0])


class FaceRecognition(object):

    def __init__(self, count_of_photo, filename_train):

        if count_of_photo < 1:
            print("Value less than 1")
            raise ValueError
        else:
            self.count_of_photo = count_of_photo
            self.face_matrix = np.zeros((STANDART_SHAPE, count_of_photo))
            self.face_matrix = create_face_matrix(count_of_photo, filename_train, self.face_matrix)
            self.average_face = np.zeros(STANDART_SHAPE)
            self.face_for_search = np.zeros(STANDART_SHAPE)
            self.weight_matrix = np.zeros((count_of_photo, count_of_photo))
            self.full_eigen_vectors = np.zeros((self.count_of_photo, STANDART_SHAPE))
    
    def calculate_pca(self):
        self.average_face = calculate_average_face(self.count_of_photo, self.average_face, self.face_matrix)
        self.face_matrix = normalize_face_matrix(self.count_of_photo, self.average_face, self.face_matrix)

        transpose_face_matrix = np.zeros((self.count_of_photo, STANDART_SHAPE))
        # eigen_vectors = np.zeros((self.count_of_photo, self.count_of_photo))
        covariation_matrix = np.zeros((self.count_of_photo, self.count_of_photo))

        for i in range(0, self.count_of_photo):
            for j in range(0, STANDART_SHAPE):
                transpose_face_matrix[i][j] = self.face_matrix[j][i]

        np.matmul(transpose_face_matrix, self.face_matrix, covariation_matrix)
        eigen_vectors = np.linalg.eig(covariation_matrix)[1]

        self.weight_matrix, self.full_eigen_vectors = count_pca(self.count_of_photo, self.face_matrix, eigen_vectors,
                                                                self.full_eigen_vectors, self.weight_matrix)

    def find_face(self, filename_for_search):
        self.face_for_search = create_face_matrix(1, filename_for_search, self.face_for_search)
        self.face_for_search = normalize_face_matrix(1, self.average_face, self.face_for_search)
        weight_vector = np.zeros(self.count_of_photo)
        distances = np.zeros(self.count_of_photo)
        result = np.zeros(2)

        number_of_found_photo = search_by_photo(self.face_for_search, weight_vector, self.weight_matrix,
                                                self.full_eigen_vectors, distances, result)

        return number_of_found_photo

    def count_pca_without_new_train(self, count_of_photo, filename):
        self.count_of_photo = count_of_photo
        self.face_matrix = np.zeros((STANDART_SHAPE, count_of_photo))
        self.face_matrix = create_face_matrix(count_of_photo, filename, self.face_matrix)

        for i in range(0, STANDART_SHAPE):
            self.average_face[i] = 0

        self.average_face = calculate_average_face(self.count_of_photo, self.average_face, self.face_matrix)
        self.face_matrix = normalize_face_matrix(self.count_of_photo, self.average_face, self.face_matrix)

        np.matmul(self.full_eigen_vectors, self.face_matrix, self.weight_matrix)
