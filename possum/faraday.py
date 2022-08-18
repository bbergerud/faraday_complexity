"""
Routines for working with Faraday spectra

Functions
---------
createFaraday(nu, polarization, phi)
    Converts a polarization spectrum into a Faraday spectrum.
"""

import numpy as np
from .units import c

def createFaraday(
    nu:np.ndarray,
    polarization:np.ndarray,
    phi:np.ndarray,
) -> np.ndarray:
    """
    Converts a polarization spectrum into a normalized Faraday spectrum.

    Parameters
    ----------
    nu : np.ndarray
        The frequency coverage

    polarization : np.ndarray
        The complex polarization array

    phi : np.ndarray
        The range of Faraday depths

    Returns
    -------
    faraday : np.ndarray
        The Faraday spectrum normalized to have a peak amplitude of 1. 
        Aligned with the provided phi values.

    Examples
    --------   
    import matplotlib.pyplot as plt
    import numpy as np
    from possum.coverage import ASKAP12
    from possum.polarization import createPolarization, addPolarizationNoise
    from possum.faraday import createFaraday

    nu = ASKAP12()
    p = createPolarization(nu=nu, chi=0, depth=20, amplitude=1)
    p = addPolarizationNoise(polarization=p, sigma=0.1)

    phi = np.linspace(-50,50,100)
    f = createFaraday(nu=nu, polarization=p, phi=phi)

    fig, ax = plt.subplots()
    ax.plot(phi, abs(f), label='abs')
    ax.plot(phi, f.real, label='real')
    ax.plot(phi, f.imag, label='imag')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlabel(r'$\phi$ (rad m$^{2}$)')
    ax.set_ylabel(r'$P_{\nu}$ (Jy/beam)')
    fig.show()
    """

    # ===========================================
    # Variables
    # ===========================================
    faraday = np.zeros(len(phi)).astype('complex')

    # ===========================================
    # Create Faraday spectrum
    # ===========================================
    lamSq = (c/nu)**2
    delta = lamSq - np.mean(lamSq)

    for i, p in enumerate(phi):
        far = np.exp(-2j * p * delta)
        far = np.sum(polarization * far)
        faraday[i] = far

    # ===========================================
    # Normalize spectrum and return
    # ===========================================
    return faraday / np.abs(faraday).max()