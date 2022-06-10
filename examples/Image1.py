"""
Example file for the GOLUM run for the
first image.
This corresponds to a Bilby run with an adapted
waveform model. 
"""

import bilby 
import matplotlib.pyplot as plt
import numpy as np
from golum.PE import LensingLikelihoods as Lenslikeli
from golum.PE import prior
from golum.Tools import utils, waveform_models
import pymultinest 

# setup usual things for the run 
outdir = 'Outdir_Event1'
label = 'event1'
sampling_frequency = 2048.
duration = 4.

# setup the injection parameters
injection_parameters = dict(mass_1 = 35.4, mass_2 = 26.7, a_1=0.4, a_2=0.3, 
							tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3, 
						    luminosity_distance = 1000., dec = -0.6, ra = 0.9,
						    theta_jn = 0.4, psi = 2.659, phase = 0.8, 
						    geocent_time = 1126259642.413, n_phase = 0.5)

np.random.seed(12)

# setup the waveform arguments and lensed models
waveform_arguments = dict(waveform_approximant = 'IMRPhenomPv2', 
						  reference_frequency = 50.,
						  minimum_frequency = 20.)
waveform_generator = bilby.gw.WaveformGenerator(duration = duration, 
												sampling_frequency= sampling_frequency,
												frequency_domain_source_model = waveform_models.LensedBBHmodel,
												waveform_arguments = waveform_arguments)

# setup the interferometers
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency,
												  duration = duration,
												  start_time = injection_parameters['geocent_time'] - 3.)
ifos.inject_signal(waveform_generator = waveform_generator,
				   parameters = injection_parameters)

# setup the priors
priors = bilby.gw.prior.BBHPriorDict()
priors.pop('mass_1') 
priors.pop('mass_2')

priors['chirp_mass'] = bilby.core.prior.Uniform(name = 'chirp_mass', latex_label = '$M_{c}$',
												minimum = 10., maximum = 100.,
												unit = '$M_{\\odot}$')

priors['mass_ratio'] = bilby.core.prior.Uniform(name = 'chirp_mass', latex_label = '$q$',
											    minimum = 0.1, maximum = 1.)
priors['mass_1'] = bilby.core.prior.Constraint(name = 'mass_1', latex_label = '$M_1$',
											  minimum = 5., maximum = 100.)
priors['mass_2'] = bilby.core.prior.Constraint(name = 'mass_2', latex_label = '$M_1$',
											   minimum = 5., maximum = 100.)
priors['geocent_time'] = bilby.core.prior.Uniform(minimum = injection_parameters['geocent_time'] - 0.1, 
												 maximum = injection_parameters['geocent_time'] + 0.1,
												 name = 'geocent_time', latex_label = '$t_c$', unit = '$s$')
priors['n_phase'] = prior.MorseFactorPrior(name = 'n_phase', latex_label = '$n_{1}$')

# setup rge likelihood
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers = ifos, 
															waveform_generator = waveform_generator,
															priors = priors)
result = bilby.run_sampler(likelihood = likelihood, priors = priors, 
						  verbose = True, sampler = 'pymultinest',
						  npoints = 4096, outdir = outdir, label = label,
						  injection_parameters = injection_parameters)

# convert m1 and m2 to chirp mass and mass ratio to plot the 
# injected values in the corner plot
injection_parameters['mass_ratio'] = bilby.gw.conversion.component_masses_to_mass_ratio(injection_parameters['mass_1'],
																						injection_parameters['mass_2'])
injection_parameters['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'],
																						injection_parameters['mass_2'])

result.plot_corner(truth = injection_parameters)
