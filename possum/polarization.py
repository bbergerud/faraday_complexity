"""
Methods for generating polarization spectra.

Functions
---------
addPolarizationNoise(polarization, sigma, seed)
    Adds random noise to the Polarization spectrum. Assumes noise is
    independent of the frequency.

createPolarization(nu, chi, depth, amplitude, apply_shift)
    Creates a complex polarization spectrum based on the provided parameters.
"""

import numpy as np
from typing import Iterable, Optional, Union
from .units import c

def addPolarizationNoise(
    polarization:np.ndarray,
    sigma:float, 
    seed:Optional[int]=None
) -> np.ndarray:
    """
    Adds random noise to the Polarization spectrum. Assumes noise is
    independent of the frequency.

    Parameters
    ----------
    polarization : np.ndarray
        The complex polarization array

    sigma : float
        The standard deviation of the Gaussian noise. The noise 
        in the real and imaginary components is Gaussian with a
        standard deviations sigma/sqrt(2). 

    seed : int, optional
        A random seed for reproducibility.

    Returns
    -------
    polarization : np.ndarray
        The updated polarization array.

    Examples
    --------
    see example usage in createPolarization
    """
    if seed is not None:
        np.random.seed(seed)

    noiseReal = np.random.normal(loc=0, scale=sigma/np.sqrt(2), size=polarization.size)
    noiseImag = np.random.normal(loc=0, scale=sigma/np.sqrt(2), size=polarization.size)

    return polarization + noiseReal + 1j*noiseImag

def createPolarization(
    nu:np.ndarray,
    chi:Union[float,Iterable],
    depth:Union[float,Iterable],
    amplitude:Union[float,Iterable]=1,
    apply_shift:bool=False,
) -> np.ndarray:
    """
    Creates a complex polarization spectrum based on the provided parameters.

    Parameters
    ----------
    nu : np.ndarray
        The range of frequencies in Hz

    chi : float, Iterable
        The intrinsic polarization angle for each source [rad].
        Can be a float for a single source or an iterable containing
        the value for each component for a complex source.

    depth : float, Iterable
        The faraday depth for each source [rad/m^2]. Can be a float
        for a single source or an iterable containing the value for
        each component for a complex source.

    amplitude : float, Iterable
        The amplitude of the signal for each each. Can be a float
        for a single source or an iterable containing the value for
        each component for a complex source.

    apply_shift : bool
        Boolean indicating whether to shift the squared wavelengths
        by subtracting the mean (True) or not (False).

    Returns
    -------
    polarization : np.ndarray
        The complex polarization

    Examples
    --------
    import matplotlib.pyplot as plt
    import numpy as np
    from possum.coverage import ASKAP12
    from possum.polarization import createPolarization, addPolarizationNoise

    np.random.seed(0)
    nu = ASKAP12()
    p = createPolarization(nu=nu, chi=0, depth=20, amplitude=1)
    p = addPolarizationNoise(polarization=p, sigma=0.2)

    fig, ax = plt.subplots()
    ax.scatter(nu/1e6, p.real, label='$Q$', s=5)
    ax.scatter(nu/1e6, p.imag, label='$U$', s=5)
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlabel(r'$\nu$ (MHz)')
    ax.set_ylabel(r'$P_{\nu}$ (Jy/beam)')
    fig.show()
    """
    # ===========================================
    # Convert parameters to matrices
    # ===========================================
    chi = np.asmatrix(chi).T
    depth = np.asmatrix(depth).T
    amplitude = np.asmatrix(amplitude).T
    lamSq = np.asmatrix((c/nu)**2)

    # ===========================================
    # Compute the polarization
    # ===========================================
    polarization = amplitude.T * np.exp(2j * (chi + depth*lamSq))
    return np.ravel(polarization)
