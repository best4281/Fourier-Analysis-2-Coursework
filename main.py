from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from util.interactive_filter import InteractiveFilterFigure, OneCutoffFilter
from util.plotter import plt_maximize

image = Image.open("images/frame01284.png")
# image = Image.open("images/bright.png")
plt_fig = InteractiveFilterFigure(image)
plt_maximize()
plt_fig.add_filter(
    OneCutoffFilter(
        "Low Pass Test",
        *plt_fig.fft_shape,
        filter_generator=np.zeros,
        adjustable=0,
    ),
    **{
        "label": "Cutoff Frequency",
        "valmin": 0,
        "valmax": 1000,
        "valstep": 1,
    },
)
plt.show()
