"""
This module contains scripts for generating a grid of frequencies.

Functions
---------
createFrequency(numin, numax, nchan)
    Creates a linearly spaced grid of frequencies over the
    indicated range.

ASKAP12():
    Returns the set of frequencies associated with the
    ASKAP 12 coverage (700 - 1300, 1500 - 1800 MHz)
    
ASKAP36():
    Returns the set of frequencies asociated with the
    ASKAP 36 coverage (1130 - 1430 MHz)
"""
from .units import MHz, c
import numpy as np

def createFrequency(
    numin:float,
    numax:float,
    nchan:int,
) -> np.ndarray:
    """
    Creates a linearly spaced grid of frequencies over the
    indicating range.

    Parameters
    ----------
    numin : float
        The minimum frequency coverage in MHz

    numax : float
        The maximum frequency coverage in MHz

    nchan : int
        The number of frequency channels

    Returns
    -------
    freq : numpy.ndarray
        Returns a numpy array of the frequencies in Hz
        linearly spaced between the minimum and maximum
        frequencies
    """

    # ======================================
    # Convert MHz to Hz
    # ======================================
    numax = numax * MHz
    numin = numin * MHz

    # ======================================
    # Generate evenly spaced grid of freq.
    # over the freq. range and return
    # ======================================
    return np.linspace(numin, numax, nchan)

def ASKAP12() -> np.ndarray:
    """
    Returns the set of frequencies associated with the
    ASKAP 12 coverage (700 - 1300, 1500 - 1800 MHz)
    """
	# ======================================
    # Get the two frequency windows
	# ======================================
    band1 = createFrequency(numin=700., numax=1300., nchan=600)
    band2 = createFrequency(numin=1500., numax=1800., nchan=300)

	# ======================================
    # Combine frequencies and return
	# ======================================
    return np.concatenate((band1, band2))

def ASKAP36() -> np.ndarray:
    """
    Returns the set of frequencies asociated with the
    ASKAP 36 coverage (1130 - 1430 MHz)
    """
    return createFrequency(numin=1130., numax=1430., nchan=300)