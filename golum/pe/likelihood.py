import bilby
from ..lookup.lookup import morse_factor_lookup, indepedendent_morse_factor_lookup, find_nearest
import numpy as np
import attr
import scipy.special
import json

from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList
from bilby.gw.utils import noise_weighted_inner_product
from collections import namedtuple

class GravitationalWaveTransientImage2NoLookup(Likelihood):
    """
    Class replacing the 'GravitationalWaveTransient' function from Bilby to 
    compute the lensing evindence using the GOLUM approximation, but not
    the lookup table
    """

    _calculated_snrs = namedtuple('calculated_snrs',
                             ['d_inner_h',
                              'optimal_snr_squared'])

    def __init__(self, interferometers, waveform_generator, posteriors, n_samp, seed = None, priors = None, use_effective_parameters = False):
        
        """
        Initialization function 

        ARGS:
        -----
        - interferometers: list of the interferometers involved in the observation
        - waveform_generator: Bilby waveform generator object used to analyse the data
                              The frequency_domain_source_model should be LensedBBHmodel
                              to account correctly for the Morse phase
        - posteriors: dictionary of the posterior samples coming from the first GOLUM run
        - n_samp: int, number of samples to be used for the  GOLUM approximation
        - seed: (int, default is None): passed to np.random.seed() for reproducability 
                of the run by taking the same samples in the GOLUM approximation
        - priors: Bilbyprior dict, default is None: Bilby prior dictionary containing 
                 the priors used for the lensing parameters.
        - use_effective_parameters: Boolean, default is False: Use effective parameters instead of image parameters
        """
        self.waveform_generator = waveform_generator
        super(GravitationalWaveTransientImage2NoLookup, self).__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.posteriors = posteriors
        self.n_samp = n_samp
        self.seed = seed
        self.priors = priors

        if self.seed is not None:
            np.random.seed(self.seed)

        # make dictionary for the random samples
        self.samples = dict()
        self.indices = np.random.randint(0, len(self.posteriors['geocent_time']), self.n_samp)
        for key in self.posteriors:
            self.samples[key] = [self.posteriors[key][idx] for idx in self.indices]

        # dictionary to be filled by the sampler
        self.parameters = dict()
        # Wheter to use effective parameters or not
        self.use_effective_parameters = use_effective_parameters


    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={})'.format(self.interferometers, self.waveform_generator)

    def calculate_snrs(self, waveform_polarizations, interferometer, params):
        """
        Function computing the SNR for a given detector and oolarizations
        """
        signal = interferometer.get_detector_response(waveform_polarizations, params)
        d_inner_h = interferometer.inner_product(signal = signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal = signal)
        
        return self._calculated_snrs(d_inner_h = d_inner_h, optimal_snr_squared = optimal_snr_squared)

    @property
    def priors(self):
        return self._prior
    @priors.setter
    def priors(self, priors):
        if priors is not None:
            self._prior = priors.copy()
        else:
            self._prior = None

    def noise_log_likelihood(self):
        """
        Function computing the noise log lieklihod. 
        """
        log_l = 0.

        for ifo in self.interferometers:
            mask = ifo.frequency_mask 
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        return float(np.real(log_l))    

    def log_likelihood_ratio_computation(self, params):
        """
        Fuction computing the log likelihood ratio for a give set 
        of samples. This assumes that the waveform generator is used with the 
        lensed BBH model
        """
        # WF polarizations can directly been computed 
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(params)

        d_inner_h = 0.
        optimal_snr_squared = 0.

        for ifo in self.interferometers:
            per_detector_snr = self.calculate_snrs(waveform_polarizations = waveform_polarizations,
                                                   interferometer = ifo, params = params)
            d_inner_h += per_detector_snr.d_inner_h
            optimal_snr_squared += np.real(per_detector_snr.optimal_snr_squared)
            
        # compute the log likelihood ratio for the set of sample under consideration 
        LogL = np.real(d_inner_h) - optimal_snr_squared /2.

        return LogL

    def log_likelihood_ratio(self):
        """
        Function computing the log likelihood ratio using the Golum 
        approximation 
        """

        log_likelihood_ratios_array = np.zeros(self.n_samp)

        for i in range(self.n_samp):
            params = dict()
            for key in self.samples:
                params[key] = self.samples[key][i]

            # adapt all the parameters to account for lensing
            # here also the Morse factor 
            if self.use_effective_parameters == False:
                params['luminosity_distance'] = np.sqrt(self.parameters['relative_magnification'])*params['luminosity_distance']
                params['geocent_time'] = self.parameters['delta_t'] + params['geocent_time']
                params['n_phase'] = self.parameters['delta_n'] + params['n_phase']
            else:
                params['luminosity_distance'] = self.parameters['luminosity_distance']
                params['geocent_time'] = self.parameters['geocent_time']
                params['n_phase'] = self.parameters['n_phase']
            log_likelihood_ratios_array[i] = self.log_likelihood_ratio_computation(params)

        # take the average over all samples to compute the Golum likelihood
        golum_log_likelihood_ratio = scipy.special.logsumexp(log_likelihood_ratios_array - np.log(len(log_likelihood_ratios_array)))

        return golum_log_likelihood_ratio

    def log_likelihood(self):
        """
        Function computing the log likelihood ratio using all the functions
        defined above 
        """
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

class GravitationalWaveTransientImage2(GravitationalWaveTransientImage2NoLookup):
    """
    Likelihood class for the second image in the Bilby runs.
    This uses the lookup table to compute the values
    """

    def __init__(self, interferometers, waveform_generator, posteriors, n_samp, seed = None, priors = None, use_effective_parameters=False):
        """
        Initilization of the class

        ARGS:
        -----
        - interferometers: list of the interferometers involved in the observation
        - waveform_generator: Bilby waveform generator object used to analyse the data
                              The frequency_domain_source_model should be LensedBBHmodel
                              to account correctly for the Morse phase
        - posteriors: dictionary of the posterior samples coming from the first GOLUM run
        - n_samp: int, number of samples to be used for the  GOLUM approximation
        - seed: (int, default is None): passed to np.random.seed() for reproducability 
                of the run by taking the same samples in the GOLUM approximation
        - priors: Bilbyprior dict, default is None: Bilby prior dictionary containing 
                 the priors used for the lensing parameters.
        - use_effective_parameters: Boolean, default is False: Use effective parameters instead of image parameters
        """

        super(GravitationalWaveTransientImage2, self).__init__(  interferometers, waveform_generator, posteriors, n_samp, seed, priors, use_effective_parameters) 
        self.snr_tables = []
        snr_tab = []
        if self.use_effective_parameters == False:
            for dn in [0, 0.5, 1, 1.5]:
                snr_tab, self.time_array = morse_factor_lookup(self.interferometers, self.waveform_generator, self.samples, dn, self.priors)
                self.snr_tables.append(snr_tab)
        else:
            for n in [0, 0.5, 1]:
                snr_tab, self.time_array = indepedendent_morse_factor_lookup(self.interferometers, self.waveform_generator, self.samples, n, self.priors)
                self.snr_tables.append(snr_tab)
        # compute the noise log likelihood once and for all 
        self.noise_log_l = self.noise_log_likelihood()
        self.use_effective_parameters=use_effective_parameters # Whether or not to use the effective paraemters

    
    def log_likelihood_ratio_lookup(self, geocent_time, mu_rel, snr_dict):
        """
        Function computing the log likelihood ratio for a given 
        set of samples based on the lookup table
        """
        idx = find_nearest(self.time_array, geocent_time)
        d_h = np.real(snr_dict['d_inner_h'][idx])/np.sqrt(mu_rel)
        h_h = np.real(snr_dict['h_inner_h'])/mu_rel
        return (d_h - h_h/2.)

    
    def log_likelihood_ratio_lookup_effective(self, geocent_time, dl, snr_dict):
        """
        Function computing the log likelihood ratio
        for a given sample
        """
        idx = find_nearest(self.time_array, geocent_time)
        d_h = np.real(snr_dict['d_inner_h'][idx])/dl
        h_h = np.real(snr_dict['h_inner_h'])/(dl**2)
        return (d_h - h_h/2.)

    def log_likelihood_ratio(self):
        """
        Function computing the log likelihood ratio 
        based on the GOLUM approximation and the 
        lookup table
        """
        if self.use_effective_parameters == False:
            snr_tab = self.snr_tables[int(self.parameters['delta_n']*2)]
            self.log_likelihood_ratios_array = [self.log_likelihood_ratio_lookup((self.parameters['delta_t']+self.samples['geocent_time'][i]), self.parameters['relative_magnification'], snr_tab[i]) for i in range(self.n_samp)]
        else:
            snr_tab = self.snr_tables[int(self.parameters['n_phase']*2)]
            self.log_likelihood_ratios_array = [self.log_likelihood_ratio_lookup_effective((self.parameters['geocent_time']), self.parameters['luminosity_distance'], snr_tab[i]) for i in range(self.n_samp)]
        return scipy.special.logsumexp(self.log_likelihood_ratios_array - np.log(len(self.log_likelihood_ratios_array)))

    def log_likelihood(self):
        """
        Function computing the log likelihood for golum using the
        functions above
        """
        return self.log_likelihood_ratio() + self.noise_log_l

class JointGravitationalWaveTransient(Likelihood):
    """
    Class implementing the joint likelihood compuation 
    for a pair of lensed events
    """

    @attr.s
    class _CalculatedSNRs:
        d_inner_h = attr.ib()
        optimal_snr_squared = attr.ib()
        complex_matched_filter_snr = attr.ib()

    def __init__(self, interferometers_image_1, interferometers_image_2, waveform_generator, priors = None, use_effective_parameters = False):
        """
        Initialization of the class

        ARGS:
        -----
        - interferometers_image_1: the bilby interferometer list used
                                    to analyze the first image
        - interferometers_image_2: the bilby interferometer list used
                                   to analyze the second image
        - waveform_generator: the bilby waveform generator to be used 
                              for the analysis
        - priors: the priors for the run. Default is None
        - use_effective_parameters: whether we want to use effective parameters
                                    or not. If True, then we are sampling
                                    on dl2, dl1, tc1, tc2, n1, n2.
        """

        self.waveform_generator = waveform_generator
        

        super(JointGravitationalWaveTransient, self).__init__()
        self.ifos_image_1 = interferometers_image_1
        self.ifos_image_2 = interferometers_image_2
        self.priors = priors

        # to be filled by the sampler
        self.parameters = dict()

        # compute the noise log likelihood for the two events
        self._noise_log_likelihood = self.noise_log_likelihood()

        # check in which case we work
        if use_effective_parameters == False:
            self.log_likelihood_ratio = self.log_likelihood_ratio_relative
        else:
            self.log_likelihood_ratio = self.log_likelihood_ratio_effective

    def __repr__(self):
        return self.__class__.__name + '(interferometers={},\n\twaveform_generator={})'.format(self.interferometers, self.waveform_generator)

    @property
    def priors(self):
        return self._prior
    @priors.setter
    def priors(self, priors):
        if priors is not None:
            self._prior = priors.copy()
        else:
            self._prior = None

    def noise_log_likelihood(self):
        """
        Function computing the noise log likelihood for the two 
        images at the same time
        """
        log_l = 0.

        # for the first image 
        for ifo in self.ifos_image_1:
            mask = ifo.frequency_mask
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        # for the second image 
        for ifo in self.ifos_image_2:
            mask = ifo.frequency_mask
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        return float(np.real(log_l))

    def calcualte_snrs(self, waveform_polarizations, interferometer, parameters):
        """
        Function computing the SNR for a given image
        """
        signal = interferometer.get_detector_response(waveform_polarizations, 
                                                      parameters)
        d_inner_h = interferometer.inner_product(signal = signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal = signal)
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        return self._CalculatedSNRs(d_inner_h = d_inner_h, optimal_snr_squared = optimal_snr_squared,
                                    complex_matched_filter_snr = complex_matched_filter_snr)

    def log_likelihood_ratio_relative(self):
        """
        Function computing the log likelihood ratio 
        for the two lensed images when using the relative lensing 
        parameters. 
        """
        params = dict()
        for key in self.parameters.keys():
            if key not in ['relative_magnification', 'delta_t', 'delta_n'] and 'recalib_' not in key:
                params[key] = self.parameters[key]
            elif 'recalib_' in key and 'img1' in key:
                new_key = key.replace('_img1', '')
                params[new_key] = self.parameters[key]

        params_2 = params.copy()
        params_2['luminosity_distance'] = np.sqrt(self.parameters['relative_magnification'])\
                                          *params['luminosity_distance']
        params_2['n_phase'] = params['n_phase'] + self.parameters['delta_n']
        params_2['geocent_time'] = params['geocent_time'] + self.parameters['delta_t']
        for key in self.parameters.keys():
            if 'recalib_' in key and 'img2' in key:
                new_key = key.replace('_img2', '')
                params[new_key] = self.parameters[key]
        

        waveform_polarizations_img1 = self.waveform_generator.frequency_domain_strain(params)
        
        waveform_polarizations_img2 = dict()
        for key in waveform_polarizations_img1:
            waveform_polarizations_img2[key] = (np.sqrt(self.parameters['relative_magnification'])**(-1))*waveform_polarizations_img1[key]*\
                                               np.exp(-1j*np.pi*self.parameters['delta_n'])
        
        d_inner_h_img1 = 0.
        optimal_snr_squared_img1 = 0.
        complex_matched_filter_snr_img1 = 0.

        for ifo in self.ifos_image_1:
            per_detector_snr = self.calcualte_snrs(waveform_polarizations = waveform_polarizations_img1,
                                                  interferometer = ifo, parameters = params)
            d_inner_h_img1 += per_detector_snr.d_inner_h
            optimal_snr_squared_img1 += np.real(per_detector_snr.optimal_snr_squared)
            

        Log_L_img1 = np.real(d_inner_h_img1) - optimal_snr_squared_img1 /2.

        # do the same but for the second image
        d_inner_h_img2 = 0.
        optimal_snr_squared_img2 = 0.
        complex_matched_filter_snr_img2 = 0.

        for ifo in self.ifos_image_2:
            per_detector_snr = self.calcualte_snrs(waveform_polarizations = waveform_polarizations_img2,
                                                   interferometer = ifo, parameters = params_2)
            d_inner_h_img2 += per_detector_snr.d_inner_h
            optimal_snr_squared_img2 += np.real(per_detector_snr.optimal_snr_squared)
            
        Log_L_img2 = (np.real(d_inner_h_img2) - optimal_snr_squared_img2 /2.)


        return float(Log_L_img1+Log_L_img2)

    def log_likelihood_ratio_effective(self):
        """
        Function computing the log likelihood ratio 
        for the two lensed images when using the effective parameters. 
        """
        params = dict()
        params_2 = dict()
        for key in self.parameters.keys():
            if key not in ['luminosity_distance_1', 'luminosity_distance_2', 'geocent_time_1', 'geocent_time_2', 'n_phase_1', 'n_phase_2'] and 'recalib_' not in key:
                params[key] = self.parameters[key]
                params_2[key] = self.parameters[key]
            elif '_1' in key:
                param =  "_".join(key.split("_")[:-1])
                params[param] = self.parameters[key]
            elif '_2' in key:
                param = "_".join(key.split("_")[:-1])
                params_2[param] = self.parameters[key]
            elif 'recalib_' in key and 'img1' in key:
                new_key = key.replace('_img1', '')
                params[new_key] = self.parameters[key]        

        waveform_polarizations_img1 = self.waveform_generator.frequency_domain_strain(params)
        
        waveform_polarizations_img2 = dict()
        for key in waveform_polarizations_img1:
            waveform_polarizations_img2[key] = (params['luminosity_distance']/params_2['luminosity_distance'])*waveform_polarizations_img1[key]*\
                                               np.exp(-1j*np.pi*(params_2['n_phase']-params['n_phase']))
        
        d_inner_h_img1 = 0.
        optimal_snr_squared_img1 = 0.
        complex_matched_filter_snr_img1 = 0.

        for ifo in self.ifos_image_1:
            per_detector_snr = self.calcualte_snrs(waveform_polarizations = waveform_polarizations_img1,
                                                  interferometer = ifo, parameters = params)
            d_inner_h_img1 += per_detector_snr.d_inner_h
            optimal_snr_squared_img1 += np.real(per_detector_snr.optimal_snr_squared)
            

        Log_L_img1 = np.real(d_inner_h_img1) - optimal_snr_squared_img1 /2.

        # do the same but for the second image
        d_inner_h_img2 = 0.
        optimal_snr_squared_img2 = 0.
        complex_matched_filter_snr_img2 = 0.

        for ifo in self.ifos_image_2:
            per_detector_snr = self.calcualte_snrs(waveform_polarizations = waveform_polarizations_img2,
                                                   interferometer = ifo, parameters = params_2)
            d_inner_h_img2 += per_detector_snr.d_inner_h
            optimal_snr_squared_img2 += np.real(per_detector_snr.optimal_snr_squared)
            
        Log_L_img2 = (np.real(d_inner_h_img2) - optimal_snr_squared_img2 /2.)


        return float(Log_L_img1+Log_L_img2)

    def log_likelihood(self):
        """
        Function computing the log likelihood for the event 
        pair using all the different info defined above
        """

        return self.log_likelihood_ratio() + self._noise_log_likelihood
