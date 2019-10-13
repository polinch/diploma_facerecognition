import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(60, 150, 3)
# t = 3
# y = np.zeros(x.shape)
t = 10
# file = open("y.txt")
x = np.genfromtxt('x_file.txt', dtype=np.int_, delimiter=',')
y = np.genfromtxt("y_file.txt", dtype=np.float, delimiter=',')

graphic = plt.plot(x, y)
plt.grid(True)
plt.scatter(x, y)
plt.xlabel('Количество фотографий. Шаг - ' + str(t) + ' фото')
plt.ylabel('Точность распознавания')
plt.show()
