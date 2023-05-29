"""
Example file for the injection of a lensed event and an unlensed
recovery (needed for the Bayes factor)
"""

import bilby
import numpy as np
import matplotlib.pyplot as plt
from golum.tools import waveformmodels, utils 
import pymultinest 


# setup some example quantities for the run 
outdir = 'Outdir_LensedInjection_UnlensedRecovery'
label = 'LensedUnlensedRecovery'
sampling_frequency = 2048.
duration = 4.

event1_parameters = {'mass_1': 36.0, 'mass_2': 29.0, 'a_1': 0.4, 
                     'a_2': 0.3, 'tilt_1': 0.5, 'tilt_2': 1.0, 
                     'phi_12': 1.7, 'phi_jl': 0.3, 
                     'luminosity_distance': 1500.0, 
                     'theta_jn': 0.4, 'psi': 2.659, 'phase': 1.3,
                     'ra': 1.375, 'dec': -1.2108, 
                     'geocent_time': 1126259642.413, 'n_phase' : 0.5}
lensing_parameters = dict(relative_magnification = 2., delta_t = 14*3600, 
                         delta_n = 0.5)

# lens the first event to have a the parameters that would be injected
injection_parameters = utils.make_bbh_parameters_from_first_image_parameters_and_lensing_parameters(event1_parameters,
                                                                                                    lensing_parameters)

# setup the waveform arguments and waveform generators
waveform_arguments = dict(waveform_approximant = 'IMRPhenomPv2',
                          reference_frequency = 50.,
                          minimum_frequency = 20.) 
# WF for injection 
injection_wf_gen = bilby.gw.WaveformGenerator(duration = duration,
                                              sampling_frequency = sampling_frequency,
                                              frequency_domain_source_model = waveformmodels.lensed_bbh_model, 
                                              waveform_arguments = waveform_arguments)
# WF for recovery 
waveform_generator = bilby.gw.WaveformGenerator(duration = duration, 
                                                sampling_frequency = sampling_frequency,
                                                frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole, 
                                                waveform_arguments = waveform_arguments)

# do the lensed injection 
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency, 
                                                   duration = duration, 
                                                   start_time = injection_parameters['geocent_time'] - 3.)
ifos.inject_signal(waveform_generator = injection_wf_gen,
                   parameters = injection_parameters)

# setup the unlensed priors for the recovery 
priors = bilby.gw.prior.BBHPriorDict()
priors['chirp_mass'] = bilby.prior.Uniform(
    name='chirp_mass', latex_label='$M$', minimum=10.0, maximum=100.0,
    unit='$M_{\\odot}$')

priors['mass_ratio'] = bilby.prior.Uniform(
    name='mass_ratio', latex_label='$q$', minimum=0.5, maximum=1.0)

priors['mass_1'] = bilby.core.prior.Constraint(minimum = 5., maximum = 100.,
                                             name = 'mass_1', latex_label = '$M_{1}$')
priors['mass_2'] = bilby.core.prior.Constraint(minimum = 5., maximum = 100.,
                                             name = 'mass_2', latex_label = '$M_{2}$')
priors['geocent_time'] = bilby.core.prior.Uniform(minimum = injection_parameters['geocent_time'] - 0.1, 
                                                  maximum = injection_parameters['geocent_time'] + 0.1, 
                                                  name = 'geocent_time',
                                                  latex_label = '$t_{c}$')
Likelihood = bilby.gw.GravitationalWaveTransient(interferometers = ifos, 
                                                 waveform_generator = waveform_generator,
                                                 priors = priors)
injection_parameters.pop('n_phase')
results = bilby.run_sampler(likelihood = Likelihood, priors = priors, 
                            sampler = 'pymultinest', npoints = 4096, 
                            injection_parameters = injection_parameters,
                            outdir = outdir, label = label)
results.plot_corner() 