import os
from PIL import Image

path = "/home/polina/Рабочий стол/facedb3/"
path_to_save = "/home/polina/Рабочий стол/new_faces/"
os.chdir(path)


i = 0
j = 0
k = 0
people_index = 0

for database in os.listdir(path):
    for people_folder in os.listdir(path + database + "/"):
        # os.chdir(path + database + "/" + people_folder + "/")
        for people_photo in os.listdir(path + database + "/" + people_folder + "/"):
            photo = Image.open(path + database + "/" + people_folder + "/" + people_photo).convert('LA')
            photo.resize((180, 200))
            os.chdir(path_to_save)
            if 1 <= i <= 4:
                os.chdir(path_to_save + "train/")
                photo.save("train" + str(j) + ".png") # people_index + 377 * temp) + ".png")
                j += 1
            elif 10 <= i <= 15:
                os.chdir(path_to_save + "test/")
                photo.save("test" + str(k) + "0.png")
                k += 1
            i += 1
        i = 0
        people_index += 1

print(len(os.listdir(path_to_save + "test")))
print(len(os.listdir(path_to_save + "train")))


# os.chdir(path)
#
# for people_folder in os.listdir(path):
#     for people_photo in os.listdir(path + people_folder):
#         photo = Image.open(path + people_folder + "/" + people_photo).convert("LA")
#         if i == 1:
#             os.chdir(path_to_save + "train/")
#             photo.save("train" + str(j) + ".png")
#                        # str(people_index + 152 * temp) + ".png")
#             j += 1
#         elif 10 < i < 15:
#             os.chdir(path_to_save + "test/")
#             photo.save("test" + str(k) + "0.png")
#             k += 1
#         i += 1
#     i = 0
#     people_index += 1

print(len(os.listdir(path_to_save + "test")))
print(len(os.listdir(path_to_save + "train")))
