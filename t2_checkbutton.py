import numpy as np
import matplotlib.pyplot as plt
from util.filter import *
from util.image_manipulation import *
from util.plotter import *

# image = plt.imread("./images/frame01284.png")
image = plt.imread("./images/bright.png")

analyzed_image = Interactive2DFFT(image)
analyzed_image.reset_figure()
analyzed_image.apply_filter(high_pass_filter, radius=30)

plt.imsave("filtered_image.png", np.abs(analyzed_image.ifft_img.image), cmap="gray")
plt_maximize()
plt.show()
