import testrecognition as fr
import numpy as np
import matplotlib.pyplot as plt
import os

# for plot
# x = np.arange(40, 62, 2)
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

path_train = "/home/polina/PycharmProjects/facerecognition/test8 - middle/train/"
path_test = "/home/polina/PycharmProjects/facerecognition/test8 - middle/test/"


# info for plot
x = np.arange(110, 350, 10)
y = np.zeros(x.shape[0])
t = x[1] - x[0]

# loh file
log_file = open("logs/logfile_start" + str(x[0]) + "_step" + str(t) + ".txt", "w+")
x_file = open("x_file.txt", "w+")
y_file = open("y_file.txt", "w+")

x.tofile(x_file, ",")
x_file.close()

os.chdir(path_train)
test = fr.CalculatePCA(x[0], 'train')
test.calc_pca()
print('PCA - ' + str(x[0]) + ' photos')
count_true = 0
count_test = 6
number_of_man = 0
os.chdir(path_test)
count_people = int(x[0] / 4)
for i in range(0, x[0]):
    for j in range(0, count_test):
        r = test.transform_faces('test' + str(count_test * i + j))
        # print(i, r, number_of_man)
        # print("\n")
        # if number_of_man <= r <= number_of_man + 5:
        #     count_true += 1
        if r == i:
            count_true += 1
    # if i % 4 == 0 and i > 0:
    #     number_of_man += 1
    # print('\n')

log_file.write('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[0]))
log_file.write('Quality = ' + str((count_true / (count_test * x[0]) * 100)) + '% ')
log_file.write("-------------------------------------------------------------------------\n")
print('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[0]))
print('Quality = ' + str((count_true / (count_test * x[0]) * 100)) + '% ')
print("-------------------------------------------------------------------------\n")
y[0] = (count_true / (x[0] * count_test)) * 100

for k in range(1, x.shape[0]):
    number_of_man = 0
    os.chdir(path_train)
    # test.pca_without_train(x[k], 'train')
    test = fr.CalculatePCA(x[k], 'train')
    test.calc_pca()
    print('PCA - ' + str(x[k]) + ' photos')
    # print('PCA without new train - ' + str(x[k]) + ' photos')
    count_true = 0
    os.chdir(path_test)
    # count_people = int(x[k] / 4)
    for i in range(0, x[k]):
        for j in range(0, count_test):
            r = test.transform_faces('test' + str(count_test * i + j))
            # temp_r = divmod(r, 40)[1]
            # if number_of_man <= r <= number_of_man + 5:
            #     count_true += 1
            if r == i:
                count_true += 1
        # if i % 4 == 0:
        #     number_of_man += 1
    log_file.write('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[0]))
    log_file.write('Quality = ' + str((count_true / (count_test * x[0]) * 100)) + '% ')
    log_file.write("-------------------------------------------------------------------------\n")
    print('Count true:' + str(count_true) + ' / Count tests:' + str(count_test * x[k]))
    print('Quality = ' + str((count_true / (count_test * x[k])) * 100) + '%')
    print("--------------------------------------------------------------------------\n")
    y[k] = (count_true / (count_test * x[k])) * 100

y.tofile(y_file, ",")
y_file.close()
log_file.close()

graphic = plt.plot(x, y)
plt.grid(True)
plt.scatter(x, y)
plt.xlabel('Количество фотографий. Шаг - ' + str(t) + ' фото')
plt.ylabel('Точность распознавания')
# plt.show()
plt.savefig('plot_start' + str(x[0]) + "_step" + str(t) + ".png")
