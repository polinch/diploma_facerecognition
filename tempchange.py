import os
from PIL import Image


os.chdir("/home/polina/Рабочий стол/tempface")
counter = 69

# for i in range(3):
#     os.rename("mhwill." + str(12 + i) + ".jpg", "test" + str(3 * counter + i) + "0.jpg")

for i in range(70):
    img = Image.open("train" + str(i) + ".jpg").convert('LA')
    img.save("train" + str(i) + ".png", "PNG")
    for j in range(3):
        img = Image.open("test" + str(3 * i + j) + "0.jpg").convert('LA')
        img.save("test" + str(3 * i + j) + "0.png", "PNG")