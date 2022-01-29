import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons, Button
from matplotlib.axes import SubplotBase


class ImageSubplot:
    def __init__(
        self,
        img: np.ndarray,
        title: str,
        visibility: bool,
        conversion=lambda im: im,
        ax=None,
    ):
        self.image = img
        self.title = title
        self.visibility = visibility
        self.conversion = conversion
        self.ax = ax

    def __call__(self):
        return self.conversion(self.image)

    def __bool__(self):
        return self.visibility


class Interactive2DFFT:
    def __init__(
        self,
        image: np.ndarray,
        isolated_channel: int = 0,
        labels: list = ["Original", "FFT", "Filtered FFT", "Inverse FFT"],
        initial_visibility: np.array = np.array([True, True, True, True]),
        create_plot: bool = True,
        gridspec: dict = {"nrows": 2, "ncols": 6, "left": 0.2, "right": 0.95},
        **kwargs,
    ):
        if image.size > 2123366400:
            # Fun fact: The number of pixels in a 32k at 16:9 ratio image is 530841600 pixels (30720x17280). The number above is 4 times of this number.
            # This warning should be ignore by default python installation.
            # Nothing should go wrong if you try to open an oversized image and does a 2D Fourier Transform, right?
            warnings.warn(
                "Image size is relatively large, the time to process it may be long or you may run out of system memory.",
                ResourceWarning,
            )

        self.original_image = ImageSubplot(image, labels[0], initial_visibility[0])
        self.fft = ImageSubplot(
            np.fft.fftshift(np.fft.fft2(isolate_channel(image, isolated_channel))),
            labels[1],
            initial_visibility[1],
            lambda im: np.log(np.abs(im)),
        )
        self.filtered_fft = ImageSubplot(
            self.fft.image,
            labels[2],
            initial_visibility[2],
            lambda im: np.log(np.abs(im)),
        )
        self.ifft_img = ImageSubplot(
            np.fft.ifft2(np.fft.ifftshift(self.filtered_fft.image)),
            labels[3],
            initial_visibility[3],
            lambda im: np.abs(im),
        )
        self.labels = labels
        self.display_images = [
            self.original_image,
            self.fft,
            self.filtered_fft,
            self.ifft_img,
        ]
        self.visible_axes = np.sum([int(x) for x in initial_visibility])
        if create_plot:
            self.create_plot(labels, initial_visibility, **kwargs)
            self.gs = self.fig.add_gridspec(**gridspec)
            self.brush = None

    def __setaxis(self, image_index, state: bool):
        self.display_images[image_index].ax.set_visible(state)
        self.display_images[image_index].ax.set_navigate(state)

    def create_plot(self, labels, visibility, **kwargs):
        self.fig = plt.figure(**kwargs)
        self.check = CheckButtons(plt.axes([0.05, 0.4, 0.1, 0.15]), labels, visibility)
        self.check.on_clicked(self.arrange_pictures)
        self.reset_button = Button(
            plt.axes([0.05, 0.325, 0.1, 0.05]), "Reset figure to initial state"
        )

        self.reset_button.on_clicked(self.reset_figure)
        self.toggle_brush = Button(plt.axes([0.05, 0.25, 0.1, 0.05]), "Enable brush")
        self.toggle_brush.on_clicked(self.__enable_brush)

    def reset_figure(self, event=None):
        assert len(self.display_images) == 4
        for i in range(4):
            if self.display_images[i].ax is not None:
                self.fig.delaxes(self.display_images[i].ax)
            if not self.display_images[i].visibility:
                self.check.set_active(i)
                self.display_images[i].visibility = True
            self.display_images[i].ax = self.fig.add_subplot(
                self.gs[int(i / 2), (i % 2) * 3 : (i % 2) * 3 + 3],
                title=self.display_images[i].title,
                visible=self.display_images[i].visibility,
            )
            self.display_images[i].ax.imshow(
                self.display_images[i](), cmap="gray" if i != 0 else None
            )
        plt.draw()

    def arrange_pictures(self, label=None):
        # assert len(self.display_images) == 4
        index = self.labels.index(label)
        self.display_images[index].visibility = not self.display_images[index].visibility
        self.visible_axes = np.sum([x.visibility for x in self.display_images])
        condition1 = lambda x: slice(None, None)
        if self.visible_axes == 1:
            condition2 = condition1
        elif self.visible_axes == 2:
            condition2 = lambda x: slice(x * 3, x * 3 + 3)
        elif self.visible_axes == 3:
            condition2 = lambda x: slice(x * 2, x * 2 + 2)
        elif self.visible_axes == 4:
            condition1 = lambda x: int(x / 2)
            condition2 = lambda x: slice((x % 2) * 3, (x % 2) * 3 + 3)
        elif self.visible_axes >= 5:
            raise ValueError(
                """A number of visible axes is not supported. Since you encounter this, You can try to modify my code to work with more axes."""
            )
        cnt = 0
        for i, img in enumerate(self.display_images):
            if img.visibility:
                self.display_images[i].ax.set_subplotspec(self.gs[condition1(cnt), condition2(cnt)])
                cnt += 1
            self.__setaxis(i, img.visibility)
        plt.draw()

    def __enable_brush(self, event=None):
        plt.ion()
        self.brush = BrushHandler(self)
        self.toggle_brush.label.set_text("Disable brush")
        self.toggle_brush.on_clicked(self.__disable_brush)

    def __disable_brush(self, event=None):
        if self.brush is not None:
            self.brush = None
        plt.ioff()
        self.toggle_brush.label.set_text("Enable brush")
        self.toggle_brush.on_clicked(self.__enable_brush)

    def apply_filter(self, filter_func, **kwargs):
        self.filtered_fft.image = filter_func(self.fft.image, **kwargs)
        self.ifft_img.image = np.fft.ifft2(np.fft.ifftshift(self.filtered_fft.image))
        self.reset_figure(None)


class BrushHandler:
    def __init__(self, session: Interactive2DFFT, brush_size=100, **kwargs):
        self.session = session
        self.fig = session.fig
        self.brush_size = brush_size
        self.brush_circle = patches.Circle((0, 0), brush_size)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.current_brush = None
        self.brush_exist = False
        self.x0, self.y0 = 0, 0
        self.pressevent = None

    def on_press(self, event):
        if event.inaxes is not None and event.button == 1:
            if not isinstance(event.inaxes, SubplotBase):
                return
            else:
                if self.brush_exist:
                    self.current_brush.remove()
                    self.x0, self.y0 = event.xdata, event.ydata
                    self.current_brush = event.inaxes.add_patch(
                        patches.Circle((event.xdata, event.ydata), self.brush_size)
                    )
                else:
                    self.x0, self.y0 = event.xdata, event.ydata
                    self.current_brush = event.inaxes.add_patch(
                        patches.Circle((event.xdata, event.ydata), self.brush_size)
                    )
                    self.brush_exist = True
        else:
            if self.brush_exist:
                self.x0, self.y0 = 0, 0
                if isinstance(self.current_brush, patches.Circle) and self.brush_exist:
                    self.current_brush.remove()
                self.brush_exist = False
            return

        self.pressevent = event

    def on_release(self, event):
        self.pressevent = None
        self.x0, self.y0 = 0, 0
        if isinstance(self.current_brush, patches.Circle) and self.brush_exist:
            self.current_brush.remove()
        self.brush_exist = False

    def on_move(self, event):
        if (
            self.pressevent is None
            or event.inaxes != self.pressevent.inaxes
            or event.inaxes is None
            or event.button != 1
        ):
            return

        dx = event.xdata - self.pressevent.xdata
        dy = event.ydata - self.pressevent.ydata
        self.current_brush.center = self.x0 + dx, self.y0 + dy


def extract_channel(img, channel: int):
    """Extract the specified channel of the input image. and set other channels to 0."""
    if channel not in range(3):
        raise ValueError("channel must be 0, 1, or 2.")
    channel_img = np.copy(img)
    for i in range(3):
        if i != channel:
            channel_img[:, :, i] = 0
    return channel_img


def isolate_channel(img, channel: int):
    """Isolate the specified channel of the input image."""
    if channel not in range(3):
        raise ValueError("channel must be 0, 1, or 2.")
    channel_img = np.copy(img)
    return channel_img[:, :, channel]
