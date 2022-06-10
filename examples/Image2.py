"""
Example file of how to run the 
GOLUM analysis for the second image 
"""

import bilby 
import matplotlib.pyplot as plt
import numpy as np
from golum.PE import LensingLikelihoods as LensLikeli
from golum.PE import prior
from golum.Tools import utils, waveform_models
import pymultinest
import corner

# setup the usual things for the run 
outdir = 'Outdir_Event2'
label = 'Event2'
sampling_frequency = 2048.
duration = 4.

np.random.seed(12)

# read the first event output file to get the posteriors and 
# the injection_parameters
Event1_file = 'event1_result.json'
event1_parameters, posteriors = utils.read_image1_file(file = Event1_file )

# define the lensed parameters
n_2 = 1.
n_samp = 1000
delta_n = ((n_2 - event1_parameters['n_phase'])+2)%2 #trick to stay in domain
LensingParameters = dict(relative_magnification = 2, delta_t = 14*3600,
						 delta_n = delta_n)
# transform the parameters of the first event to make the lensed injection
injection_lensed = utils.MakeLensedParamsFromFirstEvent(event1_parameters,
														LensingParameters)

# setup the analysis model
waveform_arguments = dict(waveform_approximant = 'IMRPhenomPv2',
						  reference_frequency = 50.,
						  minimum_frequency = 20.)
waveform_generator = bilby.gw.WaveformGenerator(duration = duration,
											    sampling_frequency = sampling_frequency, 
											    frequency_domain_source_model = waveform_models.LensedBBHmodel,
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
				 delta_t = bilby.core.prior.Uniform(name = 'delta_t', minimum = LensingParameters['delta_t'] - 0.1,
				 									maximum = LensingParameters['delta_t'] + 0.1, latex_label = '$\Delta$t'))
LensingPriors = bilby.core.prior.PriorDict(dictionary = PriorDict)

# setup the likelihood 
LensingLikelihood = LensLikeli.GravitationalWaveTransientImage2(ifos, 
 																waveform_generator = waveform_generator,
 																posteriors = posteriors, n_samp = n_samp, 
 																seed = None, priors = LensingPriors)
# NOTE: if you want to run without lookup
# change 'GravitationalWaveTransientImage2' 
# by 'GravitationalWaveTransientImage2_noLookup'

LensedResult = bilby.run_sampler(likelihood = LensingLikelihood, priors = LensingPriors,
								 sampler = 'pymultinest', injection_parameters = LensingParameters,
								 npoints = 2048, outdir = outdir, label = label)
LensedResult.plot_corner()

# Reweigh the samples from the first event to account for the lensing 
# effect (here we do not load the golum samples and other info)

ref_img_posteriors, golum_img_posteriors = utils.WeightsEvent1(LensedResult, LensingLikelihood,
								           ifos, waveform_generator,
								           outdir = outdir, label = label)
truths = [1.375, -1.2108]
fig = corner.corner(np.transpose([np.array(ref_img_posteriors['ra']), np.array(ref_img_posteriors['dec'])]), 
					color = 'blue', bins = 30, truths = truths, plt_density = False,
					levels = (0.95,))
plt.show()
