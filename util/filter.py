from abc import abstractmethod
import traceback
import numpy as np
from scipy.ndimage import fourier_gaussian
from util.widget_handling import InteractionWidgets, WidgetType


class FilterBase:
    def __init__(self, name, rows, cols, filter_generator=None, adjustable=1, initial_args=[], initial_kwargs={}):
        """
        Base class for filters.
        :param rows: number of rows in the filter array
        :param cols: number of columns in the filter array
        :param filter_generator: function that generates the filter array
        :param adjustable: number of sliders that can be adjusted
        :param initial_args: initial values for the arguments of the filter function.
        these args should be representing the adjustable parameters and sliders should be implemented to adjust them.
        :param initial_kwargs: initial values for the keyword arguments of the filter function
        :param kwargs: additional arguments for the filter function
        """
        self.name = name
        self.rows = rows
        self.cols = cols
        self._filter_generator = filter_generator
        if initial_args:
            self.args_values = initial_args
        elif isinstance(adjustable, int):
            self.args_values = [1] * adjustable
        else:
            raise ValueError("initial_args and/or must be specified")
        self.kwargs_values = initial_kwargs
        if filter_generator is None:
            self.filter_array = np.ones((rows, cols))
        else:
            self.set_filter(filter_generator)
        self.apply_button = None
        self.sliders = []
        self.binded_fig = None

    def __call__(self, spectrum):
        try:
            return spectrum * self.filter_array
        except Exception:
            # most of the error is caused by the filter array not being the same size as the spectrum
            traceback.print_exc()
            print("Cannot multiply spectrum and filter array, returning original spectrum")
            return spectrum

    @abstractmethod
    def update_array(self, event=None):
        pass

    def set_filter(self, func):
        try:
            tmp = func(self.rows, self.cols, *self.args_values, **self.kwargs_values)
            if tmp.shape == (self.rows, self.cols):
                self.filter_array = tmp
                self._filter_generator = func

            else:
                raise ValueError(f"Filter array shape does not match {self.filter_array.shape} and {tmp.shape}")
        except:
            try:
                tmp = func((self.rows, self.cols), *self.args_values, **self.kwargs_values)
                if tmp.shape == (self.rows, self.cols):
                    self.filter_array = tmp
                    self._filter_generator = func
                else:
                    raise ValueError(f"Filter array shape does not match {self.filter_array.shape} and {tmp.shape}")
            except Exception:
                traceback.print_exc()
                print("Filter array cannot be generated, using default (no filtering)")
                self.filter_array = np.ones((self.rows, self.cols))

    def set_active(self, active, binded_fig=None):
        self.apply_button.ax.set_visible(active)
        for slider in self.sliders:
            slider.ax.set_visible(active)
        if active:
            self.apply_button.connect_callback()
            if binded_fig is None:
                print("No figure is binded to the filter, the changes will not be reflected in the figure")
                self.binded_fig = None
                return
            self.binded_fig = binded_fig
            self.binded_fig.calc_filter(self.filter_array)
            for fil in self.binded_fig.filters.values():
                if fil[0] is not self:
                    fil[0].set_active(False)
        else:
            self.apply_button.disconnect_callback()
            self.binded_fig = None

    @abstractmethod
    def make_sliders(self):
        """
        Method that creates sliders for adjustable parameters. (MUST BE IMPLEMENTED)
        """


class OneCutoffFilter(FilterBase):
    def make_sliders(self, label="Cutoff Frequency", **kwargs):
        if self.apply_button is not None:
            return
        self.sliders.append(InteractionWidgets(0.2, 0.06, 0.6, 0.03, WidgetType.SLIDER, label=label, **kwargs))
        self.sliders[0].ax.set_visible(False)
        self.apply_button = InteractionWidgets(
            0.85, 0.05, 0.1, 0.05, WidgetType.BUTTON, self.update_array, label="Apply parameter(s)"
        )
        self.apply_button.disconnect_callback()
        self.apply_button.ax.set_visible(False)

    def update_array(self, event=None):
        self.args_values[0] = self.sliders[0].widget.val
        try:
            self.filter_array = self._filter_generator(self.rows, self.cols, *self.args_values, *self.kwargs_values)
        except:
            try:
                self.filter_array = self._filter_generator(self.rows, self.cols)
            except Exception:
                traceback.print_exc()
                print("Cannot generate new filter array, filter array is not updated")
                return
        if self.binded_fig is not None:
            self.binded_fig.calc_filter(self.filter_array)


class TwoCutoffFilter(FilterBase):
    def make_sliders(self, label="Frequency range", **kwargs):
        if self.apply_button is not None:
            return
        self.sliders.append(
            InteractionWidgets(
                0.2,
                0.06,
                0.6,
                0.03,
                WidgetType.RANGESLIDER,
                label=label,
                **kwargs,
            )
        )
        self.sliders[0].ax.set_visible(False)
        self.apply_button = InteractionWidgets(
            0.85, 0.05, 0.1, 0.05, WidgetType.BUTTON, self.update_array, label="Apply parameter(s)"
        )
        self.apply_button.ax.set_visible(False)

    def update_array(self, event=None):
        self.args_values[0], self.args_values[1] = self.sliders[0].widget.val
        try:
            self.filter_array = self._filter_generator(self.rows, self.cols, *self.args_values, *self.kwargs_values)
        except:
            try:
                self.filter_array = self._filter_generator(self.rows, self.cols)
            except Exception:
                traceback.print_exc()
                print("Cannot generate new filter array, filter array is not updated")
                return
        if self.binded_fig is not None:
            self.binded_fig.calc_filter(self.filter_array)


# Not fully implemented, but should work
# Also not used in this version.
class ManyParamsFilter(FilterBase):
    def make_sliders(self, **kwargs):
        if self.apply_button is not None:
            return
        i = 0
        for param, widgetkwargs in kwargs.items():
            self.sliders.append(
                InteractionWidgets(0.2, 0.01 + i * 0.04, 0.6, 0.03, WidgetType.SLIDER, label=param, **widgetkwargs)
            )
            self.sliders[i].ax.set_visible(False)
            i += 1
        self.apply_button = InteractionWidgets(
            0.85, 0.05, 0.1, 0.05, WidgetType.BUTTON, self.update_array, label="Apply parameter(s)"
        )
        self.apply_button.ax.set_visible(False)

    def update_array(self, event=None):
        for i, slider in enumerate(self.sliders):
            self.args_values[i] = slider.widget.val
        try:
            self.filter_array = self._filter_generator(
                self.rows, self.cols, *np.ravel(self.args_values), *self.kwargs_values
            )
        except:
            try:
                self.filter_array = self._filter_generator(self.rows, self.cols)
            except Exception:
                traceback.print_exc()
                print("Cannot generate new filter array, filter array is not updated")
                return
        if self.binded_fig is not None:
            self.binded_fig.calc_filter(self.filter_array)


def high_pass_filter(spectrum: np.ndarray, radius: int = None) -> np.ndarray:
    """
    Perform an 'ideal' high pass filter on the given image spectrum

    Parameters
    ----------
    spectrum : numpy.ndarray
        2-dimensional numpy array that is the image spectrum after performing 2D FFT
    radius : int
        The radius of the ideal filter, i.e. the cutoff frequency of the filter.
        If 'None', the radius is set to 1/3 of the minimum of the image's height and width

    Returns
    ----------
    filtered_spectrum : numpy.ndarray
        2-dimensional numpy array of the image spectrum after applying the ideal high pass filter with the given radius
    """
    if len(spectrum.shape) != 2:
        raise ValueError("spectrum must be a 2D array")
    hpf = hpf_generator(spectrum.shape[0], spectrum.shape[1], radius)
    return np.multiply(spectrum, hpf)


def low_pass_filter(spectrum: np.ndarray, radius: int = None) -> np.ndarray:
    """
    Perform an 'ideal' low pass filter on the given image spectrum

    Parameters
    ----------
    spectrum : numpy.ndarray
        2-dimensional numpy array that is the image spectrum after performing 2D FFT
    radius : int
        The radius of the ideal filter, i.e. the cutoff frequency of the filter.
        If 'None', the radius is set to 1/3 of the minimum of the image's height and width

    Returns
    ----------
    filtered_spectrum : numpy.ndarray
        2-dimensional numpy array of the image spectrum after applying the ideal low pass filter with the given radius
    """
    if len(spectrum.shape) != 2:
        raise ValueError("spectrum must be a 2D array")
    rows, cols = spectrum.shape
    mid_row = int(rows / 2)
    mid_col = int(cols / 2)
    if radius is None:
        radius = np.minimum(mid_row, mid_col) / 3
    lpf = np.zeros((rows, cols), np.uint8)  # create a matrix of zeros (discard all frequencies)
    for i in range(rows):
        for j in range(cols):
            if (i - mid_row) ** 2 + (j - mid_col) ** 2 <= radius ** 2:
                lpf[i][j] = 1  # if inside the circle, keep the frequency
    return np.multiply(spectrum, lpf)


def band_pass_filter(spectrum: np.ndarray, inner_radius: int = None, outer_radius: int = None) -> np.ndarray:
    """
    Perform an 'ideal' band pass filter on the given image spectrum

    Parameters
    ----------
    spectrum : numpy.ndarray
        2-dimensional numpy array that is the image spectrum after performing FFT
    inner_radius : int
        The inner radius of the ideal band pass filter, i.e. the 'lower' cutoff frequency of the filter.
        If 'None' and no `outer_radius` provided, `inner_radius` will be set to 1/5 of the minimum of the image's height and width.
        If 'None' with `outer_radius` provided, the radius is set to 1/3 of the `outer_radius`.
    outer_radius : int
        The outer radius of the ideal band pass filter, , i.e. the 'upper' cutoff frequency of the filter.
        If 'None' and no `inner_radius` provided, `outer_radius` will be set to 3/5 of the minimum of the image's height and width.
        If 'None' with `inner_radius` provided, the radius is set to 3 times of the `inner_radius`.

    Returns
    ----------
    filtered_spectrum : numpy.ndarray
        2-dimensional numpy array of the image spectrum after applying the ideal band pass filter with the given specifications
    """
    if len(spectrum.shape) != 2:
        raise ValueError("spectrum must be a 2D array")
    bpf = bpf_generator(spectrum.shape[0], spectrum.shape[1], inner_radius, outer_radius)
    return np.multiply(spectrum, bpf)


def hpf_generator(rows, cols, radius=None):
    mid_row = int(rows / 2)
    mid_col = int(cols / 2)
    if radius is None:
        radius = np.minimum(mid_row, mid_col) / 3
    hpf = np.zeros((rows, cols), np.uint8)  # create a matrix of zeros (discard all frequencies)
    for i in range(rows):
        for j in range(cols):
            if (i - mid_row) ** 2 + (j - mid_col) ** 2 > radius ** 2:
                hpf[i][j] = 1  # if outside the circle, keep the frequency
    return hpf


def bpf_generator(rows, cols, cutoff_1=None, cutoff_2=None):
    mid_row = int(rows / 2)
    mid_col = int(cols / 2)
    if cutoff_1 is None and cutoff_2 is None:
        cutoff_1 = np.minimum(mid_row, mid_col) / 5
        cutoff_2 = cutoff_1 * 3
    elif cutoff_1 is None:
        cutoff_1 = cutoff_2 / 3
    elif cutoff_2 is None:
        cutoff_2 = cutoff_1 * 3
    if cutoff_1 > cutoff_2:
        # swap values
        cutoff_1, cutoff_2 = cutoff_2, cutoff_1
    bpf = np.zeros((rows, cols), np.uint8)  # create a matrix of zeros (discard all frequencies)
    for i in range(rows):
        for j in range(cols):
            if (i - mid_row) ** 2 + (j - mid_col) ** 2 >= cutoff_1 ** 2 and (i - mid_row) ** 2 + (
                j - mid_col
            ) ** 2 <= cutoff_2 ** 2:
                bpf[i][j] = 1  # if inside the annulus, keep the frequency
    return bpf


def low_gaussian_generator(rows, cols, radius=None):
    if radius is None:
        radius = 1
    return np.fft.fftshift(fourier_gaussian(np.ones((rows, cols)), radius))


def high_gaussian_generator(rows, cols, radius=None):
    if radius is None:
        radius = 1
    ones_arr = np.ones((rows, cols))
    return ones_arr - np.fft.fftshift(fourier_gaussian(ones_arr, radius))
