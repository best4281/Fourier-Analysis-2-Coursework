from matplotlib import pyplot as plt
from PIL import Image
from util.interactive_filter import InteractiveFilterFigure
from util.plotter import plt_maximize

image = Image.open("images/frame01284.png")
# image = Image.open("images/bright.png")
plt_fig = InteractiveFilterFigure(image)
plt_maximize()

plt.show()
