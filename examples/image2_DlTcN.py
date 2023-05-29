"""
Script to test the likelihood in the case where we keep 
tc, n_2 and Dl2 as free parameters. 
"""

import bilby
import matplotlib.pyplot as plt
import numpy as np
from golum.pe import likelihood as LensLikeli
from golum.pe import prior
from golum.tools import utils, waveformmodels
import corner

# usual things for the run 
outdir = 'Outdir_Event2'
label = 'Test_Ev2_FreeDl2'
sampling_frequency = 2048.
duration = 4.

np.random.seed(12)

# read the result from the first event
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

search_parameters = dict(luminosity_distance = injection_lensed['luminosity_distance'],
                         n_phase = injection_lensed['n_phase'],
                         geocent_time = injection_lensed['geocent_time'])
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

# setup the prior dictoinary adapted to the new parameters
PriorDict = dict(n_phase = prior.MorseFactorPrior(name = 'n_phase', latex_label = '$n_{2}$'),
                 luminosity_distance = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', 
                                                                         minimum=1e2, maximum=5e4),
                 geocent_time = bilby.core.prior.Uniform(name = 'geocent_time', latex_label = "$t_{c, 2}$",
                                                         minimum = injection_lensed['geocent_time'] - 0.1,
                                                         maximum = injection_lensed['geocent_time'] + 0.1))
lensing_prior = bilby.core.prior.PriorDict(dictionary =  PriorDict)

# setup the likelihood 
# change to test without lookup table
lensing_likelihood = LensLikeli.GravitationalWaveTransientImage2(interferometers = ifos,
                                                                            waveform_generator = waveform_generator,
                                                                            posteriors = posteriors, n_samp = n_samp,
                                                                            seed = None, priors = lensing_prior, use_effective_parameters=True)
# run sampler and plot results
lensed_results = bilby.run_sampler(likelihood = lensing_likelihood, priors = lensing_prior,
                                  verbose = True, sampler = 'pymultinest',
                                  injection_parameters = search_parameters,
                                  npoints = 2048, outdir = outdir, label = label)
lensed_results.plot_corner()

# do the reweighing assuming signals are the same
# except for the Dl, tc, n
ref_img_posteriors, golum_img_posteriors = utils.weights_event_1(lensed_results, lensing_likelihood,
                                           ifos, waveform_generator,
                                           outdir = outdir, label = label, dl2_free = True,
                                           im1_posteriors = event1_file)
truths = [1.375, -1.2108]
fig = corner.corner(np.transpose([np.array(ref_img_posteriors['ra']), np.array(ref_img_posteriors['dec'])]), 
                    color = 'blue', bins = 30, truths = truths, plt_density = False,
                    levels = (0.95,))
plt.show()
