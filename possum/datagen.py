"""
Routines for creating training data for Tensorflow

Functions
---------
createDataArray(params, nu, phi, noise_function)
    Creates an array of Faraday spectra based on the provided parameters.
    Combines createPolarizationArray and createFaradayArray into a single
    call.

createFaradayArray(nu, phi, polarization)
    Converts a polarization array into a normalized Faraday array.

createPolarizationArray(params, nu, noise_function)
    Generates an array of polarizations associated with the parameters
    and frequencies.

datagen(nu, phi, batch, seed, transform, yield_params, **kwargs)
    Generator that yields a Faraday spectrum and class label

generateParams(size, amplitude_generator, chi_generator, depth_generator, noise_generator, p_complex, seed)
    Generates a set of parameters of the given size
"""

import numpy as np
from typing import Dict, Optional
from .faraday import createFaraday
from .polarization import addPolarizationNoise, createPolarization

def createDataArray(
    params:dict,
    nu:np.ndarray,
    phi:np.ndarray,
    noise_function:Optional[callable] = addPolarizationNoise,
    noise_kwargs:dict={},
) -> np.ndarray:
    """
    Creates an array of Faraday spectra based on the provided parameters.
    Combines createPolarizationArray and createFaradayArray into a single
    call.

    Parameters
    ----------
    params : dict
        A set of parameter values returned by generateParams

    nu : np.ndarray
        A set of wavelengths over which to calculate the polarization

    phi : np.ndarray
        The range of Faraday depths

    noise_function : callable, optional
        A function for adding noise to the Polarization spectrum.
        It should take as input the polarization and any arguments
        contained in params['noise'] or noise_kwargs. If None,
        then no noise is added.

    noise_kwargs : dict
        A dictionary containing additional keyword arguments to pass
        into the noise function that are not contained in params.

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
    from possum.datagen import generateParams, createDataArray

    n = 5
    nu = ASKAP12()
    phi = np.linspace(-50, 50, 101)
    params = generateParams(n)
    data = createDataArray(params=params, nu=nu, phi=phi)

    def plot(i):
        fig, ax = plt.subplots()
        ax.plot(phi, abs(data[i]), label='abs')
        ax.plot(phi, data[i].real, label='real')
        ax.plot(phi, data[i].imag, label='imag')
        ax.set_xlabel(r'$\phi$ (rad m$^{2}$)')
        ax.set_ylabel(r'$P_{\nu}$ (Jy/beam)')
        fig.legend()
        fig.show()

    for i in range(n):
        plot(i)
    """
    polarization = createPolarizationArray(params=params, nu=nu, noise_function=noise_function, noise_kwargs=noise_kwargs)
    return createFaradayArray(nu=nu, phi=phi, polarization=polarization)

def createFaradayArray(
    nu:np.ndarray,
    phi:np.ndarray,
    polarization:np.ndarray
) -> np.ndarray:
    """
    Converts a polarization array into a normalized Faraday array.

    Parameters
    ----------
    nu : np.ndarray
        The frequency coverage

    phi : np.ndarray
        The range of Faraday depths

    polarization : np.ndarray
        The complex polarization array

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
    from possum.datagen import generateParams, createPolarizationArray, createFaradayArray

    n = 5
    nu = ASKAP12()
    phi = np.linspace(-50, 50, 101)
    params = generateParams(n)
    polarization = createPolarizationArray(params=params, nu=nu)
    data = createFaradayArray(nu=nu, phi=phi, polarization=polarization)

    def plot(i):
        fig, ax = plt.subplots()
        ax.plot(phi, abs(data[i]), label='abs')
        ax.plot(phi, data[i].real, label='real')
        ax.plot(phi, data[i].imag, label='imag')
        ax.set_xlabel(r'$\phi$ (rad m$^{2}$)')
        ax.set_ylabel(r'$P_{\nu}$ (Jy/beam)')
        fig.legend()
        fig.show()

    for i in range(n):
        plot(i)
    """
    size = len(polarization)

    faraday = np.empty((size, len(phi)), dtype='cfloat')
    for i in range(size):
        faraday[i] = createFaraday(
            nu=nu,
            phi=phi,
            polarization=polarization[i]
        )

    return faraday

def createPolarizationArray(
    params:dict,
    nu:np.ndarray,
    noise_function:Optional[callable] = addPolarizationNoise,
    noise_kwargs:dict={},
) -> np.ndarray:
    """
    Generates an array of polarizations associated with the parameters
    and frequencies.

    Parameters
    ----------
    params : dict
        A set of parameter values returned by generateParams

    nu : np.ndarray
        A set of wavelengths over which to calculate the polarization

    noise_function : callable, optional
        A function for adding noise to the Polarization spectrum.
        It should take as input the polarization and any arguments
        contained in params['noise'] or noise_kwargs. If None,
        then no noise is added.

    noise_kwargs : dict
        A dictionary containing additional keyword arguments to pass
        into the noise function that are not contained in params.

    Returns
    -------
    polarization : np.ndarray
        The complex polarization for each of the sources

    Examples
    --------
    import matplotlib.pyplot as plt
    import numpy as np
    from possum.coverage import ASKAP12
    from possum.datagen import generateParams, createPolarizationArray

    n = 5
    nu = ASKAP12(); MHz = nu/1e6
    phi = np.linspace(-50, 50, 101)
    params = generateParams(n)
    data = createPolarizationArray(params=params, nu=nu)

    def plot(i):
        fig, ax = plt.subplots()
        ax.scatter(MHz, data[i].real, label='Q (real)', s=5)
        ax.scatter(MHz, data[i].imag, label='U (imag)', s=5)
        ax.set_xlabel(r'$\nu$ (MHz)')
        ax.set_ylabel(r'$P_{\nu}$ (Jy/beam)')
        fig.legend()
        fig.show()

    for i in range(n):
        plot(i)
    """
    size = len(params['label'])

    polarization = np.empty((size, len(nu)), dtype='cfloat')
    for i in range(size):
        p = createPolarization(
            nu=nu,
            chi=params['chi'][i],
            depth=params['depth'][i],
            amplitude=params['amplitude'][i],
        )

        if noise_function is not None:
            polarization[i] = noise_function(
                polarization=p,
                **{k:v[i] for k,v in params['noise'].items()},
                **noise_kwargs
            )

    return polarization

def datagen(
    nu:np.ndarray,
    phi:np.ndarray,
    batch:int=32,
    seed:bool=None,
    noise_function:Optional[callable]=addPolarizationNoise,
    noise_kwargs:dict={},
    transform:Optional[callable]=None,
    yield_params:bool=False,
    **kwargs
):
    """
    Generator that yields a Faraday spectrum and class label

    Parameters
    ----------
    nu : np.ndarray
        A set of wavelengths over which to calculate the polarization

    phi : np.ndarray
        The range of Faraday depths

    batch : int
        The number of object instance to create in each mini-batch

    seed : int, optional
        A random number seed for reproducibility. Optional.

    noise_function : callable, optional
        A function for adding noise to the Polarization spectrum.
        It should take as input the polarization and any arguments
        contained in params['noise'] or noise_kwargs. If None,
        then no noise is added.

    noise_kwargs : dict
        A dictionary containing additional keyword arguments to pass
        into the noise function that are not contained in params.

    transform : callable, optional
        A function for transforming the Faraday spectrum. Useful for
        removing the complex format into something like the magnitude
        or splitting into two channels.

    yield_params : bool
        Boolean indicating whether to also yield the parameter dictionary
        (True) or just the label (False). Default is False to align with
        Tensorflow's expectation for training.

    **kwargs
        Additional parameters to pass into the generateParams function

    Returns
    -------
    faraday : np.ndarray
        The Faraday spectrum normalized to have a peak amplitude of 1. 
        Aligned with the provided phi values.

    label : np.ndarray, optional
        The class labels associated with the Faraday spectrum, with 0
        corresponding to a simple source and 1 a complex source. The
        label is returned if yield_params=False.
    
    params : dict
        The parameters values generated during the call. Useful for
        accessing the data used to generate the Faraday spectra. This
        is returned if yield_params=True. 

    Examples
    --------
    import numpy as np
    from possum.coverage import ASKAP12
    from possum.datagen import datagen

    nu = ASKAP12(); MHz = nu/1e6
    phi = np.arange(-100,101)
    dgen = datagen(nu=nu, phi=phi,batch=32)

    X, y = next(iter(dgen))
    print(X.shape, y.shape)
    """
    # ===========================================
    # Set the random seed (if applicable)
    # ===========================================
    if seed != None:
        np.random.seed(seed)

    while True:
        params = generateParams(size=batch, **kwargs)
        X = createDataArray(params=params, nu=nu, phi=phi, noise_function=noise_function, noise_kwargs=noise_kwargs)
        yield X if transform is None else transform(X), params if yield_params else params['label']

def generateParams(
    size:int,
    amplitude_generator:callable = lambda size: np.random.uniform(0.01, 1.0, size),
    chi_generator:callable = lambda size: np.random.uniform(-np.pi, np.pi, size),
    depth_generator:callable = lambda size: np.random.uniform(-50, 50, size),
    noise_generator:Dict[str,callable] = {'sigma': lambda size: np.random.rand(size)},
    p_complex:float = 0.5,
    seed:Optional[int] = None,
) -> Dict[str,np.ndarray]:
    """
    Generates a set of parameters of the given size

    Parameters
    ----------
    size : int
        The number of Faraday sources to generate parameters for

    amplitude_generator : callable
        Function that takes as input a size parameter and returns
        a numpy array of random amplitude values of the specified
        size. Note that this function is only applied to the second
        component, thus it should return a value between 0 and 1 to
        ensure the first component is the primary component.

    chi_generator : callable
        Function that takes as input a size parameter and returns a
        numpy array of random polarization angles of the specified size.

    depth_generator : callable
        Function that takes as input a size parameter and returns a
        numpy array of random Faraday depths of the specified size.

    noise_generator : callable
        A dictionary containing a set of (key,value) pairs where the key
        represent the argument to the noise function and whose value is
        a function that takes as input a size parameter and returns a
        a sequence of random parameter values of the specified size.        

    p_complex : float
        The probability that a source is complex.

    seed : int, optional
        A random number seed for reproducibility. Optional.

    Returns
    -------
    params : dict
        A dictionary containing the values associated with the
        amplitude, chi, depth, label, and noise parameters, where
        label=0 indicates a simple source and label=1 a complex source.

    Examples
    --------
    from possum.datagen import generateParams

    params = generateParams(5)
    print(params)
    """
    if seed != None:
        np.random.seed(seed)

    # Generate the labels (0=simple, 1=complex)
    label = np.random.binomial(1, p_complex, size)

    # Generate parameter values
    amplitude = np.ones(size).astype('object')
    chi = chi_generator(size).astype('object')
    depth = depth_generator(size).astype('object')
    noise = {k:v(size) for k,v in noise_generator.items()}

    # Add secondary components
    loc = np.where(label == 1)[0]
    complex_size = len(loc)

    if complex_size > 0:
        params = [amplitude, chi, depth]
        funcs = [amplitude_generator, chi_generator, depth_generator]
        for x,f in zip(params, funcs):
            x[loc] = list(zip(x[loc], f(complex_size)))

    return {
        'amplitude': amplitude,
        'chi': chi,
        'depth': depth,
        'label': label,
        'noise': noise,
    }

