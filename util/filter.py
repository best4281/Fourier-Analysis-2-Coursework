import numpy as np


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
    rows, cols = spectrum.shape
    mid_row = int(rows / 2)
    mid_col = int(cols / 2)
    if radius is None:
        radius = np.minimum(mid_row, mid_col) / 3
    hpf = np.zeros((rows, cols), np.uint8) # create a matrix of zeros (discard all frequencies)
    for i in range(rows):
        for j in range(cols):
            if (i - mid_row) ** 2 + (j - mid_col) ** 2 > radius ** 2:
                hpf[i][j] = 1 # if outside the circle, keep the frequency
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
    lpf = np.zeros((rows, cols), np.uint8) # create a matrix of zeros (discard all frequencies)
    for i in range(rows):
        for j in range(cols):
            if (i - mid_row) ** 2 + (j - mid_col) ** 2 <= radius ** 2:
                lpf[i][j] = 1 # if inside the circle, keep the frequency
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
    rows, cols = spectrum.shape
    mid_row = int(rows / 2)
    mid_col = int(cols / 2)
    if inner_radius is None and outer_radius is None:
        inner_radius = np.minimum(mid_row, mid_col) / 5
        outer_radius = inner_radius * 3
    elif inner_radius is None:
        inner_radius = outer_radius / 3
    elif outer_radius is None:
        outer_radius = inner_radius * 3
    bpf = np.zeros((rows, cols), np.uint8) # create a matrix of zeros (discard all frequencies)
    for i in range(rows):
        for j in range(cols):
            if (i - mid_row) ** 2 + (j - mid_col) ** 2 >= inner_radius ** 2 and \
                (i - mid_row) ** 2 + (j - mid_col) ** 2 <= outer_radius ** 2:
                bpf[i][j] = 1 # if inside the annulus, keep the frequency
    return np.multiply(spectrum, bpf)
