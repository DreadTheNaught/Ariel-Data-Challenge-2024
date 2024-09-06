'''
Contains functions used for preprocessing the data
'''
import numpy as np
import itertools 
from astropy.stats import sigma_clip

def adc_convert(signal:np.ndarray, gain:float, offset:float)->np.ndarray:
    """
    Converts an analog signal to a digital signal using gain and offset.

    This function adjusts the input signal by dividing it by a specified gain and then 
    adding an offset to it. The result is a transformed signal that simulates the 
    behavior of an analog-to-digital converter (ADC).

    Parameters:
    ----------
    signal : np.ndarray
        The input signal array to be converted. It is expected to be a NumPy array.
    
    gain : float
        The gain value to adjust the signal. The signal is divided by this value.
    
    offset : float
        The offset value to be added to the adjusted signal.

    Returns:
    -------
    np.ndarray
        The converted signal as a NumPy array with dtype `np.float64`.

    Example:
    --------
    >>> import numpy as np
    >>> signal = np.array([1.0, 2.0, 3.0])
    >>> gain = 2.0
    >>> offset = 0.5
    >>> ADC_convert(signal, gain, offset)
    array([1.0, 1.5, 2.0])
    """
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    """
    Masks hot and dead pixels in a signal based on dark and dead pixel data.

    This function identifies hot pixels in the dark frame using sigma clipping and masks 
    them in the signal. It also applies the provided dead pixel mask to the signal, 
    resulting in a masked array where both hot and dead pixels are excluded.

    Parameters:
    ----------
    signal : np.ndarray
        The input signal array, typically a 3D array (e.g., multiple 2D frames).
    
    dead : np.ndarray
        A 2D boolean array indicating the location of dead pixels (True for dead pixels).
    
    dark : np.ndarray
        A 2D array representing a dark frame, used to identify hot pixels.
    
    Returns:
    -------
    np.ma.MaskedArray
        The signal array with hot and dead pixels masked.

    Example:
    --------
    >>> import numpy as np
    >>> from astropy.stats import sigma_clip
    >>> signal = np.random.random((10, 256, 256))
    >>> dead = np.zeros((256, 256), dtype=bool)
    >>> dark = np.random.random((256, 256))
    >>> dead[100:110, 100:110] = True  # Example dead pixels
    >>> masked_signal = mask_hot_dead(signal, dead, dark)
    """
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def apply_linear_corr(linear_corr:np.ndarray, clean_signal:np.ndarray)->np.ndarray:
    """
    Applies a linearity correction to a signal based on inverse polynomial coefficients.

    The function corrects the non-linearity in the pixel response of the signal, which 
    is caused by capacitive leakage on the readout electronics during integration. The 
    correction is applied using calibration data provided as inverse polynomial 
    coefficients, which describe the non-linear behavior of the pixel response.

    Parameters:
    ----------
    linear_corr : np.ndarray
        A 3D array containing the coefficients of the inverse polynomial function for 
        each pixel. The shape is expected to be (degree+1, height, width).
    
    clean_signal : np.ndarray
        The input signal to be corrected, typically a 3D array where the first dimension 
        is the time or frame index, and the second and third dimensions correspond to 
        the pixel grid.

    Returns:
    -------
    np.ndarray
        The signal with linearity correction applied.

    Notes:
    -----
    The non-linearity of the pixel response is due to capacitive leakage on the readout 
    electronics of each pixel during the integration time. The response of each pixel 
    is not perfectly linear with the number of electrons collected, but this effect can 
    be modeled and corrected using a polynomial function.

    Example:
    --------
    >>> import numpy as np
    >>> import itertools
    >>> linear_corr = np.random.random((3, 256, 256))  # Example coefficients
    >>> clean_signal = np.random.random((10, 256, 256))  # Example signal data
    >>> corrected_signal = apply_linear_corr(linear_corr, clean_signal)
    """
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark(signal:np.ndarray, dead:np.ndarray, dark:np.ndarray, dt:np.ndarray)->np.ndarray:
    """
    Corrects the signal for dark current by subtracting the calibrated dark frame.

    This function pre-processes the input signal by correcting for dark current, which 
    is a constant signal that accumulates in each pixel during the integration time. 
    The dark current map is first adjusted for dead pixels, then subtracted from the 
    signal after being scaled by the exposure time.

    Parameters:
    ----------
    signal : np.ndarray
        The input signal to be corrected, typically a 3D array where the first dimension 
        represents the time or frame index, and the second and third dimensions 
        correspond to the pixel grid.
    
    dead : np.ndarray
        A 2D boolean array indicating the location of dead pixels (True for dead pixels).
    
    dark : np.ndarray
        A 2D array representing the dark frame, which maps the detector's response to 
        dark current for a very short exposure time.
    
    dt : np.ndarray
        A 1D array representing the exposure time for each frame in the signal.

    Returns:
    -------
    np.ndarray
        The signal array with dark current correction applied.

    Notes:
    -----
    Dark current is an inherent signal in each pixel that builds up during integration 
    time, independent of incoming light. The correction is performed by subtracting the 
    product of the dark frame and the exposure time from the signal, accounting for dead 
    pixels.

    Example:
    --------
    >>> import numpy as np
    >>> signal = np.random.random((10, 256, 256))  # Example signal data
    >>> dead = np.zeros((256, 256), dtype=bool)
    >>> dark = np.random.random((256, 256))  # Example dark frame
    >>> dt = np.ones(10) * 0.5  # Example exposure times
    >>> corrected_signal = clean_dark(signal, dead, dark, dt)
    """
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal

def get_cds(signal:np.ndarray)->np.ndarray:
    """
    Computes the Correlated Double Sampling (CDS) from the input signal.

    The function calculates the difference between the signal at the end of exposure 
    and the signal at the start of exposure for each frame. This process, known as 
    Correlated Double Sampling (CDS), is used to reduce noise by subtracting the 
    initial readout from the final readout in a sequence of alternating exposures.

    Parameters:
    ----------
    signal : np.ndarray
        The input signal array, typically a 4D array where the first dimension is the 
        time or frame index, the second dimension alternates between the start and end 
        of the exposure, and the last two dimensions correspond to the pixel grid.

    Returns:
    -------
    np.ndarray
        The CDS-corrected signal, representing the difference between the end and start 
        of exposure for each frame.

    Notes:
    -----
    The science frames are read twice for each exposure, once at the start and once at 
    the end. The final CDS is computed as the difference between the signal at the end 
    of the exposure and the signal at the start, effectively reducing readout noise.

    Example:
    --------
    >>> import numpy as np
    >>> signal = np.random.random((10, 4, 256, 256))  # Example signal data
    >>> cds_signal = get_cds(signal)
    """
    cds = signal[1::2, :, :] - signal[::2, :, :]
    return cds

def correct_flat_field(flat, dead, signal):
    """
    Applies flat field correction to the input signal, accounting for pixel-to-pixel variations.

    The function corrects for variations in the detector’s response, such as differences 
    in quantum efficiency across pixels. This is achieved by dividing the signal by a 
    flat field, which is a map of the detector's response to uniform illumination. 
    Dead pixels are masked out before applying the correction.

    Parameters:
    ----------
    flat : np.ndarray
        A 2D array representing the flat field, which maps the detector’s response to 
        uniform illumination.
    
    dead : np.ndarray
        A 2D boolean array indicating the location of dead pixels (True for dead pixels).
    
    signal : np.ndarray
        The input signal to be corrected, typically a 3D array where the first dimension 
        represents the time or frame index, and the second and third dimensions 
        correspond to the pixel grid.

    Returns:
    -------
    np.ndarray
        The signal array after flat field correction.

    Notes:
    -----
    Flat field correction is used to address pixel-to-pixel variations in the detector, 
    such as differences in quantum efficiency. The correction is performed by dividing 
    the signal by the flat field, which has been adjusted for dead pixels.

    Example:
    --------
    >>> import numpy as np
    >>> flat = np.random.random((256, 256))  # Example flat field
    >>> dead = np.zeros((256, 256), dtype=bool)
    >>> signal = np.random.random((10, 256, 256))  # Example signal data
    >>> corrected_signal = correct_flat_field(flat, dead, signal)
    """
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal

def bin_obs(cds_signal, binning):
    """
    Performs time binning on Correlated Double Sampling (CDS) signals to reduce data size.

    This function bins time series observations of CDS signals together at a specified 
    frequency to save space. The time dimension is reduced by summing over specified 
    intervals, effectively averaging the signal over time bins.

    Parameters:
    ----------
    cds_signal : np.ndarray
        The input CDS signal array, typically a 4D array where the first dimension is 
        the time or frame index, the second dimension is the pixel index, and the last 
        two dimensions correspond to the pixel grid.
    
    binning : int
        The binning factor, representing the number of consecutive time frames to sum 
        together into a single bin.

    Returns:
    -------
    np.ndarray
        The binned CDS signal array with the time dimension reduced according to the 
        specified binning factor.

    Notes:
    -----
    Time binning is often used to reduce data size by aggregating observations over 
    specified intervals. This can be useful for saving space and reducing noise in 
    time series data.

    Example:
    --------
    >>> import numpy as np
    >>> cds_signal = np.random.random((100, 2, 256, 256))  # Example CDS signal data
    >>> binning = 5
    >>> binned_signal = bin_obs(cds_signal, binning)
    """
    cds_transposed = cds_signal.transpose(0,2,1)
    cds_binned = np.zeros((cds_transposed.shape[0]//binning, cds_transposed.shape[1], cds_transposed.shape[2]))
    for i in range(cds_transposed.shape[0]//binning):
        cds_binned[i, :, :] = np.sum(cds_transposed[i*binning:(i+1)*binning, :, :], axis=0)
    return cds_binned