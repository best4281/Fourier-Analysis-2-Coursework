from math import inf
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PIL import Image

from util.widget_handling import InteractionWidgets, WidgetType
from util.filter import (
    OneCutoffFilter,
    TwoCutoffFilter,
    ManyParamsFilter,
    FilterBase,
    bpf_generator,
    hpf_generator,
    low_gaussian_generator,
    high_gaussian_generator,
)


class ImageSubplot:
    def __init__(
        self,
        img: np.ndarray,
        title: str,
        visible: bool,
        cmap=None,
        conversion=lambda im: im,
        ax=None,
        **imshow_kwargs,
    ):
        """
        Class for the subplot of the matplotlib figure.
        :param img: image to be displayed
        :param title: title of the subplot
        :param visible: if the subplot is visible
        :param cmap: colormap to be used for the image
        :param conversion: function to convert the image to the proper format that can be displayed (must return a numpy array)
        :param ax: matplotlib axes object
        :param imshow_kwargs: additional arguments for the imshow function
        """
        self.image = img
        self.title = title
        self.visible = visible
        self.conversion = conversion
        self.cmap = cmap
        self.ax = ax
        self._imshow_kwargs = imshow_kwargs

    def __call__(self):
        return self.conversion(self.image)

    def update_ax(self, img: np.ndarray = None):
        """
        Updates the image of the subplot.
        :param img: new image
        """
        if img is not None:
            if img.shape != self.image.shape:
                raise ValueError("Image shape does not match")
            self.image = img
            self.ax.cla()
            self.ax.set_title(self.title)
            self.ax.imshow(self.conversion(self.image), cmap=self.cmap, **self._imshow_kwargs)

    def show_image(self):
        self.ax.cla()
        self.ax.set_title(self.title)
        self.ax.imshow(self.conversion(self.image), cmap=self.cmap, **self._imshow_kwargs)


class InteractiveFilterFigure:

    layout_condition = [
        (lambda x: slice(None, None), lambda x: slice(None, None)),
        (lambda x: slice(None, None), lambda x: slice(x * 6, x * 6 + 6)),
        (lambda x: slice(None, None), lambda x: slice(x * 4, x * 4 + 4)),
        (lambda x: int(x / 2), lambda x: slice((x % 2) * 6, (x % 2) * 6 + 6)),
        (lambda x: int(x / 3), lambda x: slice(x * 4, x * 4 + 4) if x <= 2 else slice((x - 3) * 6, (x - 3) * 6 + 6)),
        (lambda x: int(x / 3), lambda x: slice((x % 3) * 4, (x % 3) * 4 + 4)),
        (lambda x: int(x / 4), lambda x: slice(x * 3, x * 3 + 3) if x <= 3 else slice((x - 4) * 4, (x - 4) * 4 + 4)),
        (lambda x: int(x / 4), lambda x: slice((x % 4) * 3, (x % 4) * 3 + 3)),
    ]

    def __init__(
        self,
        original_image,
        starter_filter: str = "Gaussian Highpass Filter",
        labels=[
            "Original Image",
            "Grayscale",
            "not shifted 2D FFT (log)",
            "shifted 2D FFT (log)",
            "phase of 2D FFT",
            "Filter array",
            "Filtered 2D FFT (log)",
            "Inverse 2D FFT",
        ],
        initial_visibility: np.array = np.array([True, False, False, True, True, True, True, True]),
        gridspec: dict = {"nrows": 2, "ncols": 12, "left": 0.2, "right": 0.95, "bottom": 0.15},
        **kwargs,
    ):
        """
        Class for the interactive filter figure.
        :param original_image: original image to be filtered, must be a numpy array/ PIL image/ file path (str)
        :param labels: labels of each subplots, must be list or tuple
        :param initial_visibility: visibility of each subplot when the figure is shown, must be list or tuple
        :param gridspec: dict of gridspec parameters for the figure
        :param kwargs: additional arguments for the figure constructor
        """
        print("Initializing interactive filter figure...")
        if isinstance(original_image, np.ndarray):
            self._original_image = original_image
            if original_image.ndim == 2:
                self._grayscale_image = original_image.copy()
            else:
                # See discussion: https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale
                self._grayscale_image = np.dot(original_image[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
        elif isinstance(original_image, Image.Image):
            self._original_image = np.asarray(original_image)
            if self._original_image.ndim == 2:
                self._grayscale_image = self._original_image.copy()
            else:
                self._grayscale_image = np.asarray(original_image.convert("L"))
        elif isinstance(original_image, str):
            im = Image.open(original_image)
            self._original_image = np.asarray(im)
            self._grayscale_image = np.asarray(self._original_image.convert("L"))
        else:
            raise ValueError("original_image must be a numpy array or a string")

        if len(labels) != len(initial_visibility):
            raise ValueError(
                f"Labels and initial visibility length must be the same: {len(labels)} != {len(initial_visibility)}"
            )

        if self._original_image.size > 2123366400:
            # Fun fact: The number of pixels in a 32k at 16:9 ratio image is 530841600 pixels (30720x17280). The number above is 4 times of this number.
            # This warning should be ignore by default python installation.
            # Nothing should go wrong if you try to open an oversized image and does a 2D Fourier Transform, right?
            warnings.warn(
                "Image size is relatively large, the time to process it may be long or you may run out of system memory.",
                ResourceWarning,
            )
        print("Performing 2D Fourier Transform. Depending on the image size, this may take a while...")
        self._non_shifted_fft = np.fft.fft2(self._grayscale_image)
        self._fft = np.fft.fftshift(self._non_shifted_fft)
        self._max_fft_dim = np.max(self._fft.shape)
        self.filter_array = np.ones(self._fft.shape)
        self.filtered_fft = self._fft.copy()
        self.inverse_fft = np.fft.ifft2(np.fft.ifftshift(self.filtered_fft))
        self._original_transform = [
            self.filter_array.copy(),
            self.filtered_fft.copy(),
            self.inverse_fft.copy(),
        ]
        print("2D Fourier Transform completed. Initializing matplotlib figure...")
        self._labels = labels
        self._initial_visibility = initial_visibility
        self._starter_filter = starter_filter
        self.visible_axes = np.sum(self._initial_visibility)
        self.fig = plt.figure(**kwargs)
        self.gridspec = self.fig.add_gridspec(**gridspec)
        self.subplots = []
        self.subplots.append(
            ImageSubplot(
                self._original_image,
                labels[0],
                initial_visibility[0],
                "gray" if self._original_image.ndim == 2 else None,
            )
        )
        self.subplots.append(
            ImageSubplot(self._grayscale_image, labels[1], initial_visibility[1], "gray", vmin=0, vmax=255)
        )
        self.subplots.append(
            ImageSubplot(self._non_shifted_fft, labels[2], initial_visibility[2], "gray", lambda im: np.log(np.abs(im)))
        )
        self.subplots.append(
            ImageSubplot(self._fft, labels[3], initial_visibility[3], "gray", lambda im: np.log(np.abs(im)))
        )
        self.subplots.append(ImageSubplot(self._fft, labels[4], initial_visibility[4], "gray", lambda im: np.angle(im)))
        self.subplots.append(ImageSubplot(self.filter_array, labels[5], initial_visibility[5], "gray", vmin=0, vmax=1))
        self.subplots.append(
            ImageSubplot(
                self.filtered_fft,
                labels[6],
                initial_visibility[6],
                "gray",
                lambda im: np.log(np.abs(im)),
            )
        )
        self.subplots.append(
            ImageSubplot(
                self.inverse_fft,
                labels[7],
                initial_visibility[7],
                "gray",
                lambda im: np.abs(np.real(im)),
                vmin=0,
                vmax=255,
            )
        )

        self.filters = {
            "Gaussian Highpass Filter": (
                OneCutoffFilter(
                    "Gaussian Highpass Filter",
                    self._fft.shape[0],
                    self._fft.shape[1],
                    filter_generator=high_gaussian_generator,
                    initial_args=[1],
                ),
                {
                    "label": "Sigma",
                    "valmin": 0,
                    "valmax": 25,
                    "valinit": 1,
                    "initcolor": "red",
                },
            ),
            "Gaussian Lowpass Filter": (
                OneCutoffFilter(
                    "Gaussian Lowpass Filter",
                    self._fft.shape[0],
                    self._fft.shape[1],
                    filter_generator=low_gaussian_generator,
                    initial_args=[1],
                ),
                {
                    "label": "Sigma",
                    "valmin": 0,
                    "valmax": 25,
                    "valinit": 1,
                    "initcolor": "red",
                },
            ),
            "Ideal Bandpass Filter": (
                TwoCutoffFilter(
                    "Ideal Bandpass Filter",
                    self._fft.shape[0],
                    self._fft.shape[1],
                    filter_generator=bpf_generator,
                    initial_args=[int(self._max_fft_dim) / 100, (self._max_fft_dim / 10)],
                ),
                {
                    "label": "Cutoff Frequency Range",
                    "valmin": 0,
                    "valmax": int(self._max_fft_dim / 4),
                    "valstep": 1,
                    "valinit": (int(self._max_fft_dim) / 100, (self._max_fft_dim / 10)),
                },
            ),
            "Ideal Highpass Filter": (
                OneCutoffFilter(
                    "Ideal Highpass Filter",
                    self._fft.shape[0],
                    self._fft.shape[1],
                    filter_generator=hpf_generator,
                    initial_args=[int(self._max_fft_dim / 100)],
                ),
                {
                    "valmin": 0,
                    "valmax": int(self._max_fft_dim / 4),
                    "valstep": 1,
                    "valinit": int(self._max_fft_dim / 100),
                    "initcolor": "red",
                },
            ),
        }
        for fil in self.filters.values():
            fil[0].make_sliders(**fil[1])

        self.widgets = {
            "filter_radiobuttons": InteractionWidgets(
                0.02,
                0.15,
                0.16,
                0.25,
                WidgetType.RADIOBUTTONS,
                self.set_filter,
                labels=list(self.filters.keys()),
                active=[*self.filters.keys()].index(starter_filter),
            ),
            "axes_checkbuttons": InteractionWidgets(
                0.02,
                0.63,
                0.16,
                0.25,
                WidgetType.CHECKBUTTONS,
                self.arrange_subplots,
                labels=self._labels,
                actives=self._initial_visibility,
            ),
            "filter_reset_button": InteractionWidgets(
                0.05, 0.42, 0.1, 0.05, WidgetType.BUTTON, self.reset_filter, label="Show no filter"
            ),
            "layout_reset_button": InteractionWidgets(
                0.05, 0.49, 0.1, 0.05, WidgetType.BUTTON, self.set_figure_layout, label="Reset subplots layout"
            ),
            "view_reset_button": InteractionWidgets(
                0.05, 0.56, 0.1, 0.05, WidgetType.BUTTON, self.reset_view, label="Reset view in all subplot"
            ),
        }

        self.arrange_subplots()
        self.filters[starter_filter][0].set_active(True, self)

    @property
    def fft_shape(self):
        return self._fft.shape

    def add_filter(self, new_filter: FilterBase, **sliderkwargs):
        self.filters[new_filter.name] = (new_filter, sliderkwargs)
        self.fig.delaxes(self.widgets["filter_radiobuttons"].ax)
        self.widgets["filter_radiobuttons"] = (
            InteractionWidgets(
                0.02,
                0.15,
                0.16,
                0.25,
                WidgetType.RADIOBUTTONS,
                self.set_filter,
                labels=list(self.filters.keys()),
                active=[*self.filters.keys()].index(self._starter_filter),
            ),
        )
        self.filters[new_filter.name][0].make_sliders(**sliderkwargs)

    def set_filter(self, filter_name: str):
        self.filters[filter_name][0].set_active(True, self)
        self.fig.canvas.draw()

    def calc_filter(self, new_filter_array: np.ndarray):
        if self.filter_array.shape != new_filter_array.shape:
            raise ValueError(
                f"Filter array shape must be the same as the original image shape: {self.filter_array.shape} != {new_filter_array.shape}"
            )
        self.filter_array = new_filter_array
        self.subplots[5].update_ax(self.filter_array)
        self.filtered_fft = np.multiply(self._fft, self.filter_array)
        self.subplots[6].update_ax(self.filtered_fft)
        self.inverse_fft = np.fft.ifft2(np.fft.ifftshift(self.filtered_fft))
        self.subplots[7].update_ax(self.inverse_fft)
        self.fig.canvas.draw()

    def reset_filter(self, event=None):
        if event is not None:
            if event.button is not MouseButton.LEFT:
                return
        for n in (5, 6, 7):
            self.subplots[n].update_ax(self._original_transform[n - 5])

    def set_figure_layout(self, event=None, **kwargs):
        if event is not None:
            if event.button is not MouseButton.LEFT:
                return
        if "override" in kwargs:
            if len(kwargs["override"]) != len(self.subplots):
                print("Override length is not equal to subplots length. Override is ignored.")
                visibility = self._initial_visibility
            else:
                visibility = kwargs["override"]
        else:
            visibility = self._initial_visibility
        self.visible_axes = np.sum(visibility)
        eql = 0
        self.widgets["axes_checkbuttons"].disconnect_callback()
        for i, subplot in enumerate(self.subplots):
            if subplot.visible != visibility[i]:
                subplot.visible = visibility[i]
                self.widgets["axes_checkbuttons"].widget.set_active(i)
            else:
                eql += 1
        self.widgets["axes_checkbuttons"].connect_callback()
        if eql == len(self.subplots):
            return
        self.arrange_subplots()

    def reset_view(self, event=None):
        if event is not None:
            if event.button is not MouseButton.LEFT:
                return
        plt.setp(
            [subplot.ax for subplot in self.subplots],
            xlim=(0, self._fft.shape[1]),
            ylim=(self._fft.shape[0], 0),
        )

    def arrange_subplots(self, label=None):
        # assert len(self.subplots) == 8
        if label is not None:
            index = self._labels.index(label)
            self.subplots[index].visible = not self.subplots[index].visible
            if self.subplots[index].visible:
                self.visible_axes += 1
            else:
                self.visible_axes -= 1
        if self.visible_axes > 8:
            raise ValueError(
                """A number of visible axes is not supported. Since you encounter this, You can try to modify my code to work with more axes."""
            )
        cnt = 0
        for i, subplot in enumerate(self.subplots):
            if subplot.ax is None:
                subplot.ax = self.fig.add_subplot(
                    self.gridspec[
                        self.layout_condition[self.visible_axes - 1][0](cnt),
                        self.layout_condition[self.visible_axes - 1][1](cnt),
                    ],
                    title=subplot.title,
                    facecolor="black",
                )
                subplot.ax.get_xaxis().set_visible(False)
                subplot.ax.get_yaxis().set_visible(False)
                subplot.show_image()
                if subplot.visible:
                    cnt += 1
            else:
                if subplot.visible:
                    self.subplots[i].ax.set_subplotspec(
                        self.gridspec[
                            self.layout_condition[self.visible_axes - 1][0](cnt),
                            self.layout_condition[self.visible_axes - 1][1](cnt),
                        ]
                    )
                    cnt += 1
                    # self.subplots[i].update_ax(redraw=True if label is None else False)
            subplot.ax.set_visible(subplot.visible)
            subplot.ax.set_navigate(subplot.visible)
        self.fig.canvas.draw()
