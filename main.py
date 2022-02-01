from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from util.interactive_filter import InteractiveFilterFigure, OneCutoffFilter
from util.plotter import plt_maximize


def ideal_low_pass_array(rows, cols, cutoff_frequency):
    """
    A function to generate a 2D array of ideal low pass filter.
    :param rows: number of rows in the array (matrix), i.e. height of the image
    :param cols: number of columns in the array (matrix), i.e. width of the image
    :param cutoff_frequency: cutoff frequency of the filter, i.e. radius of the filter circle
    :return: 2D array of ideal low pass filter
    """

    # This should return a 2D array of the same size as the input image.
    # Members of this array should only be 0 or 1.
    # 0 means that the pixel should be removed, 1 means that the pixel should be kept. (matrix multiplication with original image)
    # The center of the filter circle should be at the center of the array.

    # Example:
    #                ⌈ 0 0 0 0 ⌉
    # filter_array = | 0 1 1 0 | # This is the ideal low pass filter matrix with rows=4, cols=4, and cutoff_frequency=1.
    #                | 0 1 1 0 | # Notice that 1 is the radius, not the diameter.
    #                ⌊ 0 0 0 0 ⌋

    # Hint:
    # 1. You can use np.zeros() to create an array of zeros -> https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # 2. Beware that the circle must originate at the center of the array.
    # 3. Use the circle equation as a condition to determine the value of the members of the array.
    # 4. Be careful with the return value. It should be a 2D array.

    filter_array: np.ndarray

    # Your code here

    return filter_array


if __name__ == "__main__":
    # You can change the font size here if the text is too big.
    font_settings = {"font.size": 10}
    plt.rcParams.update(font_settings)

    image = Image.open("images/frame01284.png")
    # image = Image.open("images/bright.png")

    # You can also open your own image using the following code: image = Image.open("path/to/your/image.png")
    # Have fun experimenting with different images!

    plt_fig = InteractiveFilterFigure(image)

    # Comment out the following line to add your implemented low pass filter above, by removing the """ """
    """
    plt_fig.add_filter(
        OneCutoffFilter(
            "Low Pass Test",
            plt_fig.fft_shape[0],
            plt_fig.fft_shape[1],
            filter_generator=ideal_low_pass_array,
            adjustable=1,
        ),
        label="Cutoff Frequency",
        valmin=0,
        valmax=int(plt_fig.fft_shape[0] / 4),
        valstep=1,
    )
    """

    plt_maximize()
    plt.show()
