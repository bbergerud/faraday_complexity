# Faraday Complexity
Faraday rotation can reveal important properties about the medium between us and a radio source and has been used extensively to probe the physical properties of astronomical objects such as the [solar corona](https://astronomy.swin.edu.au/cosmos/C/Corona), [H <span style="font-variant:small-caps">ii</span> (star-forming) regions](https://astronomy.swin.edu.au/cosmos/h/HII+Region), along with the [interstellar](https://astronomy.swin.edu.au/cosmos/I/Interstellar+Gas+Cloud), [intergalactic](https://astronomy.swin.edu.au/cosmos/I/Intergalactic+Medium), and [intracluster](https://astronomy.swin.edu.au/cosmos/I/Intra-cluster+Medium) mediums. However, if the signal is comprised of more than one radio source, then standard analysis using a single component can produce results unrelated to the underlining signals.

The goal of this project was to classify Faraday sources as being simple or "complex" for the [POSSUM (Polarisation Sky Survey of the Universe's Magnetism) survey](https://possum-survey.org/), which is part of the [Australian SKA Pathfinder (ASKAP)](https://www.atnf.csiro.au/projects/askap/index.html). [Shea Brown](https://github.com/sheabrown) came up with the idea for the project, while [Jacob Isbell](https://github.com/jwisbell), [Daniel LaRocca](https://github.com/DanielLaRocca), and I did most of the programming and analysis. This repository is meant to provide an introduction with a streamlined code implementation.

## Faraday Rotation
The polarization that astronomers measure is often represented as a complex number, with $Q$ and $U$ denoting the real and imaginary components respectively. The polarization at a given wavelength $\lambda$ that we observe in the presence of an intervening medium between us an the radio source can be expressed as

$$
P(\lambda^2) = P_0 e^{2i(\chi_0 + \phi \lambda^2)} = Q(\lambda^2) + i\cdot U(\lambda^2)
$$

where $\chi_0$ is the intrinsic polarization angle of the source and $\phi$ is the "Faraday depth". The Faraday depth depends on the amount of material between us and the source as well as the magnetic orientation and is given by

$$
\phi(\boldsymbol{r}) \propto \int_{\boldsymbol{r}}^{0} n_e \boldsymbol{B} \cdot d\boldsymbol{\ell}
$$

where $n_e$ is the electron density, $\boldsymbol{B}$ the magnetic field vector, $d\boldsymbol{\ell}$ an infinitesmal distance along the line of sight, and  $\boldsymbol{r}$ the location of the synchrotron radio source.


Faraday rotation is often parameterized by the rotation measure (RM), which measures the relationship between the wavelength and polarization angle $\chi$ via

$$
\chi(\lambda^2) = \chi_0 + RM \cdot \lambda^2
$$

This linearity, however, is only applicable for simple cases and if there is more than once source the linear relationship will often break down. Two additional problems are the "$n\pi$" ambiguity (multiple "solutions" in $\lambda^2$ space) and bandpass depolarization. One common means of reducing these issues is to apply RM synthesis, which inverts a complex polarization spectrum into a Faraday spectrum:

$$
F(\phi) \propto \int_{-\infty}^{\infty} P(\lambda^2) \exp\left[-2i\phi(\lambda^2 - \lambda_0^2)\right] \; d\lambda^2
$$

Complex sources can sometimes create issues, however, where the RM derived from RM synthesis can be well fit by a simple model that doesn't characterize the individual components nor their mean while also underestimating the uncertainty [(Farnsworth, Rudnick, and Brown, 2011)](https://ui.adsabs.harvard.edu/abs/2011AJ....141..191F/abstract). Thus separating simple and complex sources in large automated surveys can be helpful for improving the accuracy of scientific studies.


## Synthetic Data

For this project we constructed synthetic data designed to mimic observations taken by the POSSUM survey. We considered two cases, spectra consisting of a single Faraday source and those consisting of two Faraday sources. For the complex (two-component) case, we generated a polarization spectrum for each source and then added them together:

$$
P(\lambda^2) = P_1 e^{2i(\chi_1 + \phi_1 \lambda^2)} + P_2 e^{2i(\chi_x + \phi_2 \lambda^2)}
$$

For each Polarization spectrum we then added random noise to the real and imaginary components, assuming the noise is indepenent of frequency.

As an example, the following code snippet shows how to generate a noisy polarization spectrum for a single source usnig the ASKAP 12 coverage, which we show in the Figure below:

```python
import matplotlib.pyplot as plt
import numpy as np
from possum.coverage import ASKAP12
from possum.polarization import createPolarization, addPolarizationNoise

np.random.seed(0)
nu = ASKAP12(); MHz=nu/1e6
p = createPolarization(nu=nu, chi=0, depth=20, amplitude=1)
p = addPolarizationNoise(polarization=p, sigma=0.2)
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(MHz, p.real, label='$Q$ (real)', s=5)
ax.scatter(MHz, p.imag, label='$R$ (imag)', s=5)
ax.legend(loc='lower right', frameon=False)
ax.set_xlabel(r'$\nu$ (MHz)')
ax.set_ylabel(r'$P_{\nu}$ (Jy/beam)')
fig.tight_layout()
fig.show()
```

![Polarization Spectrum](figures/polarization_spectrum.png)

While the polarization coverage has a wavelength gap, we can cast it to a Faraday spectrum using the standard inversion

$$
F(\phi) \propto \sum_{k=1}^{K} P_k e^{2i\phi(\lambda_k^2 - \lambda_0^2)}
$$

where $K$ is the number of channels, $\lambda_0^2 = \langle \lambda^2 \rangle$, and $P_k$ the complex polarization in channel $k$.

The following code snippet shows how we can cast the polarization spectrum into a normalized Faraday one. Note that the amplitude of the signal peaks at the Faraday depth $(\phi=20)$.

```python
from possum.faraday import createFaraday

phi = np.linspace(-50,50,100)
f = createFaraday(nu=nu, polarization=p, phi=phi)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(phi, abs(f), label='abs')
ax.plot(phi, f.real, label='real')
ax.plot(phi, f.imag, label='imag')
ax.legend(loc='lower right', frameon=False)
ax.set_xlabel(r'$\phi$ (rad m$^{2}$)')
ax.set_ylabel(r'$P_{\nu}$ (Jy/beam)')
fig.tight_layout()
fig.show()
```
![Faraday Spectrum](figures/faraday_spectrum.png)

## Publication
[Shea Brown, Brandon Bergerud, Allison Costa, B M Gaensler, Jacob Isbell, Daniel LaRocca, Ray Norris, Cormac Purcell, Lawrence Rudnick, Xiaohui Sun, Classifying complex Faraday spectra with convolutional neural networks, Monthly Notices of the Royal Astronomical Society, Volume 483, Issue 1, February 2019, Pages 964–970.](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483..964B)