"""
Example file for the analysis of four images.

If the duration of the signal is changed, do not forget to 
change the start time for the different ifos
"""

import bilby 
import numpy as np
import matplotlib.pyplot as plt
from golum.PE import LensingLikelihoods as LensLikeli
from golum.PE import prior
from golum.Tools import utils, waveform_models
import corner
import pymultinest

# load the file from the first event 
Event1File = 'Event1_model_result.json'
event1_parameters, posteriors = utils.read_image1_file(file = Event1File)

# setup some usual parameters for the run 
outdir = 'Outdir_4images'
label = 'hierarchical_4images'
sampling_frequency = 2048.
duration = 4.

n_samp = 2000 # number of samples to be used by GOLUM
n_points  = int(2e5) # number of samples for the reweighing

n_2 = 1.
n_3 = 0.5
n_4 = 0.

delta_n2 = ((n_2 - event1_parameters['n_phase'])+2)%2 
delta_n3 = ((n_3 - event1_parameters['n_phase'])+2)%2
delta_n4 = ((n_4 - event1_parameters['n_phase'])+2)%2

# make  lensing parameters dictionary  
LensingParameters2 = dict(relative_magnification = 2., delta_t = 3600*14,
						  delta_n = delta_n2)
LensingParameters3 = dict(relative_magnification = 4., delta_t = 3600*16, 
						  delta_n = delta_n3)
LensingParameters4 = dict(relative_magnification = 8., delta_t = 3600*21, 
						  delta_n = delta_n4)

# setup waveform generator for the full run
waveform_arguments = dict(waveform_approximant = 'IMRPhenomPv2', 
						  reference_frequency = 50., minimum_frequency = 20.)
waveform_generator = bilby.gw.WaveformGenerator(duration = duration, 
												sampling_frequency = sampling_frequency,
												frequency_domain_source_model = waveform_models.LensedBBHmodel,
												waveform_arguments = waveform_arguments)

print(" ")
print("##########################################################")
print(" ")
print("Analyzing the first image pair")
print(" ")
print("##########################################################")
print(" ")

injection_lensed_2 = utils.MakeLensedParamsFromFirstEvent(event1_parameters, 
														  LensingParameters2)
# setup the interferometers for the run 
ifos_img2 = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos_img2.set_strain_data_from_power_spectral_densities(sampling_frequency= sampling_frequency,
														duration = duration, 
														start_time = injection_lensed_2['geocent_time'] - 3.)
ifos_img2.inject_signal(waveform_generator = waveform_generator, 
						parameters = injection_lensed_2)

# setup the prior dictionary for the run 
priorDict2 = dict(relative_magnification = bilby.core.prior.Uniform(name = 'relative_magnification',
																    minimum = 0.01, maximum = 10.,
																    latex_label = '$\mu_{21}$'),
				  delta_t = bilby.core.prior.Uniform(name = 'delta_t', minimum = LensingParameters2['delta_t'] - 0.1,
				  									 maximum = LensingParameters2['delta_t'] + 0.1, 
				  									 latex_label = '$\Delta t_{21}$'),
				   delta_n = prior.MorseFactorDifference(name = 'delta_n', latex_label = '$\Delta n_{21}$'))
priors_2 = bilby.core.prior.PriorDict(dictionary = priorDict2)

Likelihood2 = LensLikeli.GravitationalWaveTransientImage2(ifos_img2, 
														  waveform_generator = waveform_generator,
														  posteriors = posteriors, n_samp = n_samp,
														  seed = None, priors = priors_2)
# run the sampler
Results_21 = bilby.run_sampler(likelihood = Likelihood2, priors = priors_2, 
							   sampler = 'pymultinest', injection_parameters = LensingParameters2,
							   npoints = 2048, outdir = outdir, label = '%s_img2'%(label))

Results_21.plot_corner()

# do the reweighing 
Img2Posteriors, golum_posts = utils.WeightsEvent1(Results_21, Likelihood2,
												 ifos_img2, waveform_generator,
												 outdir = outdir, label = 'label', 
												 n_points = n_points)

print(" ")
print("#########################################################################")
print(" ")
print("Analyzing the third image accounting for the 2 first ")
print(" ")
print("#########################################################################")
print( )

injection_lensed_3 = utils.MakeLensedParamsFromFirstEvent(event1_parameters,
														  LensingParameters3)

# setup the ifos for the run 
ifos_img3 = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos_img3.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency,
													    duration = duration,
													    start_time = injection_lensed_3['geocent_time'] - 3.)
ifos_img3.inject_signal(waveform_generator = waveform_generator, 
						parameters = injection_lensed_3)

# setup the prior dictionary for the run 
priorDict3 = dict(relative_magnification = bilby.core.prior.Uniform(name = 'relative_magnification',
																    minimum = 0.01, maximum = 10.,
																    latex_label = '$\mu_{31}$'),
				  delta_t = bilby.core.prior.Uniform(name = 'delta_t', minimum = LensingParameters3['delta_t'] - 0.1, 
				  									 maximum = LensingParameters3['delta_t'] + 0.1,
				  									 latex_label = '$\Delta t_{31}$'),
				  delta_n = prior.MorseFactorDifference(name = 'delta_n', latex_label = '$\Delta n_{31}$'))
priors_3 = bilby.core.prior.PriorDict(dictionary = priorDict3)

# setup the likelihood that takes in the reweighed posteriors of the previous run 
Likelihood3 = LensLikeli.GravitationalWaveTransientImage2(ifos_img3,
														  waveform_generator = waveform_generator, 
														  posteriors = Img2Posteriors, n_samp = n_samp,
														  seed = None, priors = priors_3)
# run the sampler
Results_321 = bilby.run_sampler(likelihood = Likelihood3, priors = priors_3, 
								sampler = 'pymultinest', injection_parameters = LensingParameters3,
								npoints = 2048, outdir = outdir, label = '%s_img3'%(label))
Results_321.plot_corner()

# do the reweighing 
Img3Posteriors, golum_posts = utils.WeightsEvent1(Results_321, Likelihood3,
												 ifos_img3, waveform_generator, 
												 outdir = outdir, label = '%s'%(label),
												 n_points = n_points, n_img = 3)

print(" ")
print("####################################################################################")
print(" ")
print("Analyzing the fourth image accounting for the information from the three first")
print(" ")
print("####################################################################################")
print(" ")

injection_lensed_4 = utils.MakeLensedParamsFromFirstEvent(event1_parameters,
														  LensingParameters4)
# setup the interferometers for this run 
ifos_img4 = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos_img4.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency,
														duration = duration, 
														start_time = injection_lensed_4['geocent_time'] - 3.)
ifos_img4.inject_signal(waveform_generator = waveform_generator,
						parameters = injection_lensed_4)

# setup the prior dictionary 
priorDict4 = dict(relative_magnification = bilby.core.prior.Uniform(name = 'relative_magnification',
																    minimum = 0.01, maximum = 10.,
																    latex_label = '$\mu_{41}$'),
				  delta_t = bilby.core.prior.Uniform(name = 'delta_t', minimum = LensingParameters4['delta_t'] - 0.1,
				  									 maximum = LensingParameters4['delta_t'] + 0.1, 
				  									 latex_label = '$\Delta  t_{41}$'),
				   delta_n = prior.MorseFactorDifference(name = 'delta_n', latex_label = '$\Delta n_{41}$'))
priors_4 = bilby.core.prior.PriorDict(dictionary = priorDict4)

# setup the likelihood 
Likelihood4 = LensLikeli.GravitationalWaveTransientImage2(ifos_img4,
														  waveform_generator = waveform_generator, 
														  posteriors = Img3Posteriors, n_samp = n_samp,
														  seed = None, priors = priors_4)
# run the sampler
Results = bilby.run_sampler(likelihood = Likelihood4, priors = priors_4, 
	   					    sampler = 'pymultinest', injection_parameters = LensingParameters4,
	   					    npoints = 2048, outdir = outdir, label = '%s_img4'%(label))
Results.plot_corner()

Img4Posteriors, _ = utils.WeightsEvent1(Results_321, Likelihood3,
										ifos_img4, waveform_generator, 
										outdir = outdir, label = label,
										n_points = n_points, n_img = 4)

print(" ")
print("DONE")
print(" ")
