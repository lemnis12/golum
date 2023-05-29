"""
Example file of how to run the 
GOLUM analysis for the second image 
"""

import bilby 
import matplotlib.pyplot as plt
import numpy as np
from golum.pe import likelihood as LensLikeli
from golum.pe import prior
from golum.tools import utils, waveformmodels
import corner

# setup the usual things for the run 
outdir = 'Outdir_Event2'
label = 'Event2'
sampling_frequency = 2048.
duration = 4.

np.random.seed(12)

# read the first event output file to get the posteriors and 
# the injection_parameters
event1_file = 'event1_result.json'
event1_parameters, posteriors = utils.read_image1_file(file = event1_file )

# define the lensed parameters
n_2 = 1.
n_samp = 1000
delta_n = ((n_2 - event1_parameters['n_phase'])+2)%2 #trick to stay in domain
lensing_parameters = dict(relative_magnification = 2, delta_t = 14*3600,
                         delta_n = delta_n)
# transform the parameters of the first event to make the lensed injection
injection_lensed = utils.make_bbh_parameters_from_first_image_parameters_and_lensing_parameters(event1_parameters,
                                                                                                lensing_parameters)

# setup the analysis model
waveform_arguments = dict(waveform_approximant = 'IMRPhenomXPHM',
                          reference_frequency = 50.,
                          minimum_frequency = 20.)
waveform_generator = bilby.gw.WaveformGenerator(duration = duration,
                                                sampling_frequency = sampling_frequency, 
                                                frequency_domain_source_model = waveformmodels.lensed_bbh_model,
                                                waveform_arguments = waveform_arguments)

# setup the interferometers 
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency,
                                                   duration = duration, 
                                                   start_time = injection_lensed['geocent_time'] - 3.)
ifos.inject_signal(waveform_generator = waveform_generator,
                   parameters = injection_lensed)

# setup the prior dictionary 
PriorDict = dict(delta_n = prior.MorseFactorDifference(name = 'delta_n', latex_label = '$\Delta$n'),
                 relative_magnification = bilby.core.prior.Uniform(name = 'relative_magnification', 
                                                                    minimum = 0.01, maximum = 10.,
                                                                    latex_label = '$\mu_{rel}$'),
                 delta_t = bilby.core.prior.Uniform(name = 'delta_t', minimum = lensing_parameters['delta_t'] - 0.1,
                                                     maximum = lensing_parameters['delta_t'] + 0.1, latex_label = '$\Delta$t'))
lensing_priors = bilby.core.prior.PriorDict(dictionary = PriorDict)

# setup the likelihood 
lensing_likelihood = LensLikeli.GravitationalWaveTransientImage2(ifos, 
                                                                 waveform_generator = waveform_generator,
                                                                 posteriors = posteriors, n_samp = n_samp, 
                                                                 seed = None, priors = lensing_priors)
# NOTE: if you want to run without lookup
# change 'GravitationalWaveTransientImage2' 
# by 'GravitationalWaveTransientImage2_noLookup'

lensed_result = bilby.run_sampler(likelihood = lensing_likelihood, priors = lensing_priors,
                                 sampler = 'pymultinest', injection_parameters = lensing_parameters,
                                 npoints = 2048, outdir = outdir, label = label)
lensed_result.plot_corner()

# Reweigh the samples from the first event to account for the lensing 
# effect (here we do not load the golum samples and other info)

ref_img_posteriors, golum_img_posteriors = utils.weights_event_1(lensed_result, lensing_likelihood,
                                           ifos, waveform_generator,
                                           outdir = outdir, label = label,
                                           im1_posteriors = event1_file)
truths = [1.375, -1.2108]
fig = corner.corner(np.transpose([np.array(ref_img_posteriors['ra']), np.array(ref_img_posteriors['dec'])]), 
                    color = 'blue', bins = 30, truths = truths, plt_density = False,
                    levels = (0.95,))
plt.show()
