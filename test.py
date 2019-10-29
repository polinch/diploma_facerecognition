import recognition as fr
import numpy as np
import matplotlib.pyplot as plt
import os


# TODO writing in a logfile
def logging(count_of_true, count_of_test, count):
    print('Count true:' + str(count_of_true) + ' / Count tests:' + str(count_of_test * count[0]))
    print('Quality = ' + str((count_of_true / (count_of_test * count[0]) * 100)) + '% ')
    print("-------------------------------------------------------------------------\n")


path_train = ""
path_test = ""


count_of_photo = np.arange(110, 350, 10)
quality = np.zeros(count_of_photo.shape[0])
step = count_of_photo[1] - count_of_photo[0]

recognition = fr.FaceRecognition(count_of_photo[0], "train")
recognition.calculate_pca()

count_true = 0
count_test = 4

# TODO refactoring facerecognition/temp.py
for i in range(0, count_of_photo[0]):
    for j in range(0, count_test):
        result = recognition.find_face("test" + str(count_test * i + j))
        if result == i:
            count_true += 1
logging(count_true, count_test, count_of_photo)

print(count_of_photo.shape)






