import testrecognition as fr
import numpy as np
import matplotlib.pyplot as plt

# for plot
x = np.arange(40, 62, 2)
y = np.zeros(x.shape[0])
t = x[1] - x[0]

test = fr.CalculatePCA(x[0], 'train')
test.calc_pca()
print('PCA - ' + str(x[0]) + ' photos')
count_true = 0
count_test = 3
# for i in range(x[0]):
#     for j in range(count_test):
#         r = test.transform_faces('test' + str(count_test * i + j))
#         if r == i:
#             count_true += 1
# print('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[0]))
# print('Quality = ' + str((count_true / (count_test * x[0]) * 100)) + '% ')
# print("-------------------------------------------------------------------------\n")
# y[0] = (count_true / (x[0] * count_test)) * 100
#
# for k in range(1, x.shape[0]):
#     test.pca_without_train(x[k], 'train')
#     print('PCA without new train - ' + str(x[k]) + ' photos')
#     count_true = 0
#     for i in range(x[k]):
#         for j in range(count_test):
#             r = test.transform_faces('test' + str(count_test * i + j))
#             temp_r = divmod(r, 40)[1]
#             if r == i:
#                 count_true += 1
#     print('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[k]))
#     print('Quality = ' + str((count_true / (count_test * x[k])) * 100) + '%')
#     print("--------------------------------------------------------------------------\n")
#     y[k] = (count_true / (count_test * x[k])) * 100
#
# graphic = plt.plot(x, y)
# plt.grid(True)
# plt.scatter(x, y)
# plt.xlabel('Количество фотографий. Шаг - ' + str(t) + ' фото')
# plt.ylabel('Точность распознавания')
# plt.show(graphic)


# x = np.arange(40, 71, 1)
# y = np.zeros(x.shape[0])
# t = x[1] - x[0]
#
# test = fr.CalculatePCA(x[0], 'train')
# test.calc_pca()
# print('PCA - ' + str(x[0]) + ' photos')
# count_true = 0
# count_test = 3
# for i in range(x[0]):
#     for j in range(count_test):
#         r = test.transform_faces('test' + str(count_test * i + j))
#         if r == i:
#             count_true += 1
#     # print('\n')
# print('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[0]))
# print('Quality = ' + str((count_true / (count_test * x[0]) * 100)) + '% ')
# print("-------------------------------------------------------------------------\n")
# y[0] = (count_true / (x[0] * count_test)) * 100
#
# for k in range(1, x.shape[0]):
#     test.pca_without_train(x[k], 'train')
#     print('PCA without new train - ' + str(x[k]) + ' photos')
#     count_true = 0
#     for i in range(x[k]):
#         for j in range(count_test):
#             r = test.transform_faces('test' + str(count_test * i + j))
#             temp_r = divmod(r, 40)[1]
#             if r == i:
#                 count_true += 1
#     print('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[k]))
#     print('Quality = ' + str((count_true / (count_test * x[k])) * 100) + '%')
#     print("--------------------------------------------------------------------------\n")
#     y[k] = (count_true / (count_test * x[k])) * 100
#
# graphic = plt.plot(x, y)
# plt.grid(True)
# plt.scatter(x, y)
# plt.xlabel('Количество фотографий. Шаг - ' + str(t) + ' фото')
# plt.ylabel('Точность распознавания')
# plt.show(graphic)



