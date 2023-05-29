"""
Script to do a symmetric run for GOLUM. This alleviates some
of the previous issues and should give more stability.
"""

import bilby 
import corner
import numpy as np
import pymultinest 
import matplotlib.pyplot as plt
from golum.pe import likelihood, prior
from golum.tools import utils, waveformmodels

# give the useful information for the run 
event_1_lensed = # path to lensed run for the first event
event_1_unlensed = # path to unlensed run for the first event
event_2_lensed = # path to lensed run for second event
event_2_unlensed = # path to unlensed run for the second event

outdir = 'Outdir_SymmRunBoostedSamples'
label = 'SymmetricRun'

sampling_frequency = 2048.
duration = 4.
n_samp = 5000

# setup the injection parameters (should be same as for the first event)
image1_parameters = dict(mass_1 = 36.0, mass_2 = 29.2, a_1=0.4, a_2=0.3, 
                            tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3, 
                            luminosity_distance = 1500., dec = -1.2108, ra = 1.375,
                            theta_jn = 0.4, psi = 2.659, phase = 0.8, 
                            geocent_time = 1126259642.413, n_phase = 0.)
LensingParameters = dict(relative_magnification = 2, delta_t = 14*3600,
                         delta_n = 0.5)

# transform the parameters here to be the lensed ones for the second
# image 
image2_parameters = utils.make_bbh_parameters_from_first_image_parameters_and_lensing_parameters(image1_parameters,
                                                                                                 LensingParameters)

# setup the waveform arguments 
waveform_arguments = dict(waveform_approximant = 'IMRPhenomXPHM',
                          reference_frequency = 50.,
                          minimum_frequency = 20.)
waveform_generator = bilby.gw.WaveformGenerator(duration = duration, 
                                                sampling_frequency= sampling_frequency,
                                                frequency_domain_source_model = waveformmodels.lensed_bbh_model,
                                                waveform_arguments = waveform_arguments)

# interfermeters for the first image
np.random.seed(97)
ifos_img1 = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos_img1.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency,
                                                        duration = duration,
                                                        start_time = image1_parameters['geocent_time'] - 3.)
ifos_img1.inject_signal(waveform_generator = waveform_generator,
                        parameters = image1_parameters)

# interferometers for the second image
np.random.seed(12)
ifos_img2 = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos_img2.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency,
                                                        duration = duration,
                                                        start_time = image2_parameters['geocent_time'] - 3.)
ifos_img2.inject_signal(waveform_generator = waveform_generator,
                        parameters = image2_parameters)

# setup the priors for the run 
PriorDict = dict(delta_n_1 = prior.MorseFactorDifference(name = 'delta_n', latex_label = '$\Delta \, n_{12}$'),
                 relative_magnification_1 = bilby.core.prior.Uniform(name = 'relative_magnification', 
                                                                   minimum = 0.01, maximum = 10.,
                                                                   latex_label = '$\mu_{rel}^{21}$'),
                 delta_t_1 = bilby.core.prior.Uniform(name = 'delta_t', minimum = (-1)*LensingParameters['delta_t'] - 0.1,
                                                    maximum = (-1)*LensingParameters['delta_t'] + 0.1, latex_label = '$\Delta$ \, t_{21}$'),
                 delta_n_2 = prior.MorseFactorDifference(name = 'delta_n', latex_label = '$\Delta \, n_2$'),
                 relative_magnification_2 = bilby.core.prior.Uniform(name = 'relative_magnification', 
                                                                   minimum = 0.01, maximum = 10.,
                                                                   latex_label = '$\mu_{rel}^{12}$'),
                 delta_t_2 = bilby.core.prior.Uniform(name = 'delta_t', minimum = LensingParameters['delta_t'] - 0.1,
                                                    maximum = LensingParameters['delta_t'] + 0.1, latex_label = '$\Delta\,t_{12}$'))

priors = bilby.core.prior.PriorDict(dictionary = PriorDict)

# setup the likelihood and run the two GOLUM runs
SymmRes = likelihood.SymmetricGravitationalWaveTransient(ifos_img1, event_1_lensed, waveform_generator, ifos_img2, event_2_lensed,
                                                        n_samples_golum = n_samp, 
                                                        priors = priors,
                                                        n_samples_reweight = int(1e5), # FIXME: fast test to see if runs ok.
                                                        sampler = 'pymultinest', npoints = 2048, 
                                                        npool = 1,
                                                        outdir = outdir, label = label)

# we can compare these final results with the unlensed case
res_unl_img1 = bilby.result.read_in_result(filename = event_1_unlensed)
res_unl_img2 = bilby.result.read_in_result(filename = event_2_unlensed)

logZ_unl = res_unl_img1.log_evidence + res_unl_img2.log_evidence

Clu = SymmRes.log_evidence - logZ_unl

print("Clu = %.3f"%Clu)
final_posts = SymmRes.joint_posteriors.copy()


# plot the sky location 
fig = corner.corner(np.transpose([np.array(final_posts['ra']), np.array(final_posts['dec'])]),
                   color = 'black', bins = 30, plt_density = False,
                   levels = (0.95,), smooth = 0.5)
plt.show()