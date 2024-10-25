from MandelbrotPybind import *
import matplotlib.pyplot as plt
import numpy as np

width = 1920
height = 1080
iterations = 8000

m = Mandelbrot(width, height, iterations)
m.pixelCalculation()
m.copyBack()
data = m.getData()

data = np.array(data).reshape(height, width)

plt.imshow(data, cmap='jet')
plt.show()