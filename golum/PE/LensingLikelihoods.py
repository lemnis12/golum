import bilby
from ..Lookup.LensingLookup import MorseFactorLookup, IndependantMorseFactorLookup
from ..Tools import utils
import numpy as np
import attr
import scipy.special

from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList
from bilby.gw.utils import noise_weighted_inner_product
from collections import namedtuple

class GravitationalWaveTransientImage2(Likelihood):
    """
    Likelihood class for the second image in the Bilby runs.
    This uses the lookup table to compute the values
    """

    def __init__(self, interferometers, waveform_generator, posteriors, n_samp, seed = None, priors = None):
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
        """

        # read in the parameters
        self.waveform_generator = waveform_generator
        super(GravitationalWaveTransientImage2, self).__init__(dict())
        self.posteriors = posteriors
        self.interferometers = interferometers
        self.n_samp = n_samp
        self.seed = seed
        self.priors = priors

        if self.seed is not None:
            np.random.seed(self.seed)

        # make a dictionary for the random samples
        self.samples = dict()
        self.indices = np.random.randint(0, len(self.posteriors['geocent_time']), self.n_samp)
        for key in self.posteriors:
            self.samples[key] = np.array(self.posteriors[key])[self.indices]

        # dictionary to be filled by the sampler
        self.parameters = dict()

        self.snr_tables = []
        for dn in [0, 0.5, 1, 1.5]:
            snr_tab, self.time_array = MorseFactorLookup(self.interferometers, self.waveform_generator,
                                                         self.samples, dn, self.priors)
            self.snr_tables.append(snr_tab)

        # compute the noise log likelihood once and for all 
        self.noise_log_l = self.noise_log_likelihood()

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
        Function coputing the noise log likelihood
        """

        log_l = 0.

        for ifo in self.interferometers:
            mask = ifo.frequency_mask 
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        return float(np.real(log_l))

    def find_nearest(self, a, a0):
        """
        Function used to find the closest element in the lookup tables
        """
        idx = np.abs(a-a0).argmin()
        return idx

    def log_likelihood_ratio_lookup(self, geocent_time, mu_rel, snr_dict):
        """
        Function computing the log likelihood ratio for a given 
        set of samples based on the lookup table
        """

        idx = self.find_nearest(self.time_array, geocent_time)
        d_h = np.real(snr_dict['d_inner_h'][idx])/np.sqrt(mu_rel)
        h_h = np.real(snr_dict['h_inner_h'])/mu_rel

        return (d_h - h_h/2.)

    def log_likelihood_ratio(self):
        """
        Function computing the log likelihood ratio 
        based on the GOLUM approximation and the 
        lookup table
        """

        SNR_tab = self.snr_tables[int(self.parameters['delta_n']*2)]
        self.LogLikelisRatio = [self.log_likelihood_ratio_lookup((self.parameters['delta_t']+self.samples['geocent_time'][i]),
                                self.parameters['relative_magnification'], SNR_tab[i]) for i in range(self.n_samp)]
        
        return scipy.special.logsumexp(self.LogLikelisRatio - np.log(len(self.LogLikelisRatio)))

    def log_likelihood(self):
        """
        Function computing the log likelihood for golum using the
        functions above
        """
        return self.log_likelihood_ratio() + self.noise_log_l

class GravitationalWaveTransientImage2_noLookup(Likelihood):
	"""
	Class replacing the 'GravitationalWaveTransient' function from Bilby to 
	compute the lensing evindence using the GOLUM approximation, but not
	the lookup table
	"""

	_CalculatedSNRs = namedtuple('CalculatedSNRs',
                             ['d_inner_h',
                              'optimal_snr_squared'])

	def __init__(self, interferometers, waveform_generator, posteriors, n_samp, seed = None, priors = None):
		
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
		"""
		self.waveform_generator = waveform_generator
		super(GravitationalWaveTransientImage2_noLookup, self).__init__(dict())
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

	def __repr__(self):
		return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={})'.format(self.interferometers, self.waveform_generator)

	def calculate_snrs(self, waveform_polarizations, interferometer, params):
		"""
		Function computing the SNR for a given detector and oolarizations
		"""
		signal = interferometer.get_detector_response(waveform_polarizations, params)
		d_inner_h = interferometer.inner_product(signal = signal)
		optimal_snr_squared = interferometer.optimal_snr_squared(signal = signal)
		
		return self._CalculatedSNRs(d_inner_h = d_inner_h, optimal_snr_squared = optimal_snr_squared)

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

		LogLikelisRatio = [0 for i in range(self.n_samp)]

		for i in range(self.n_samp):
			params = dict()
			for key in self.samples:
				params[key] = self.samples[key][i]

			# adapt all the parameters to account for lensing
			# here also the Morse factor 
			params['luminosity_distance'] = np.sqrt(self.parameters['relative_magnification'])*params['luminosity_distance']
			params['geocent_time'] = self.parameters['delta_t'] + params['geocent_time']
			params['n_phase'] = self.parameters['delta_n'] + params['n_phase']

			LogLikelisRatio[i] = self.log_likelihood_ratio_computation(params)

		# take the average over all samples to compute the Golum likelihood
		GolumLogLikelihoodRatio = scipy.special.logsumexp(LogLikelisRatio - np.log(len(LogLikelisRatio)))

		return GolumLogLikelihoodRatio

	def log_likelihood(self):
		"""
		Function computing the log likelihood ratio using all the functions
		defined above 
		"""
		return self.log_likelihood_ratio() + self.noise_log_likelihood()

class GravitationalWaveTransientImage2_independant(Likelihood):
    """
    Class computing the likelihood for the second image when 
    we use the (Dl2, tc2, n_2) parametrization, with the use
    of a lookup table
    """

    def __init__(self, interferometers, waveform_generator, posteriors, n_samp, seed = None, priors = None):
        """
        Initilization function for the class

        ARGS:
        -----
        - interferometers: list of interferometers involved in the observation
        - waveform_generator: Bilby waveform generator object used to analyze the
                              data. The frequency domain source model should
                              be the LensedBBHmodel to account for the Morse phase
        - posteriors: dictionary of the posterior samples coming from the first GOLUM run
        - n_samp: int, number of samples to be used for the  GOLUM approximation
        - seed: (int, default is None): passed to np.random.seed() for reproducability 
                of the run by taking the same samples in the GOLUM approximation
        - priors: Bilbyprior dict, default is None: Bilby prior dictionary containing 
                 the priors used for the lensing parameters.
        """

        self.waveform_generator = waveform_generator
        super(GravitationalWaveTransientImage2_independant, self).__init__(dict())
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
            self.samples[key] = np.array([self.posteriors[key][idx] for idx in self.indices])

        # define dict to be filled by the sampler
        self.parameters = dict()

        # make the lookup tables
        self.snr_tables = []
        for n2 in [0, 0.5, 1]:
            snr_tab, self.time_array = IndependantMorseFactorLookup(self.interferometers, self.waveform_generator,
                                                                    self.samples, n2, priors = self.priors)
            self.snr_tables.append(snr_tab)

        # compute the noise log likelihood once and for all
        self.noise_log_l = self.noise_log_likelihood()

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
        Function coputing the noise log likelihood
        """

        log_l = 0.

        for ifo in self.interferometers:
            mask = ifo.frequency_mask 
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        return float(np.real(log_l))

    def find_nearest(self, a, a0):
        """
        Function used to find the closest element in the lookup tables
        """
        idx = np.abs(a-a0).argmin()
        return idx

    def log_likelihood_ratio_lookup(self, geocent_time, dl, snr_dict):
        """
        Function computing the log likelihood ratio
        for a given sample
        """
        idx = self.find_nearest(self.time_array, geocent_time)
        d_h = np.real(snr_dict['d_inner_h'][idx])/dl
        h_h = np.real(snr_dict['h_inner_h'])/(dl**2)

        return (d_h - h_h/2.)

    def log_likelihood_ratio(self):
        """
        Function comuting the log likelihood ratio 
        based on the GOLUM method and the lookup table
        """
        SNR_tab = self.snr_tables[int(self.parameters['n_phase']*2)]
        LogLikelisRatio = [self.log_likelihood_ratio_lookup(self.parameters['geocent_time'],
                           self.parameters['luminosity_distance'], SNR_tab[i]) for i in range(self.n_samp)]
        
        return scipy.special.logsumexp(LogLikelisRatio - np.log(len(LogLikelisRatio)))

    def log_likelihood(self):
        """
        Function computing the log likelihood
        for GOLUM using the functions above
        """
        return self.log_likelihood_ratio() + self.noise_log_l


class GravitationalWaveTransientImage2_independant_noLookup(Likelihood):
    """
    Class to compute the likelihood without the lookup table 
    in the case where we want to sample upon Dl2, tc2, n_2
    """
    _CalculatedSNRs = namedtuple('CalculatedSNRs',
                             ['d_inner_h',
                              'optimal_snr_squared'])

    def __init__(self, interferometers, waveform_generator, posteriors, n_samp, seed = None, priors = None):
        
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
        """
        self.waveform_generator = waveform_generator
        super(GravitationalWaveTransientImage2_indpendant_noLookup, self).__init__(dict())
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

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={})'.format(self.interferometers, self.waveform_generator)

    def calculate_snrs(self, waveform_polarizations, interferometer, params):
        """
        Function computing the SNR for a given detector and oolarizations
        """
        signal = interferometer.get_detector_response(waveform_polarizations, params)
        d_inner_h = interferometer.inner_product(signal = signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal = signal)
        
        return self._CalculatedSNRs(d_inner_h = d_inner_h, optimal_snr_squared = optimal_snr_squared)

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

        LogLikelisRatio = [0 for i in range(self.n_samp)]

        for i in range(self.n_samp):
            params = dict()
            for key in self.samples:
                params[key] = self.samples[key][i]

            # adapt all the parameters to account for lensing
            # here also the Morse factor 
            params['luminosity_distance'] = self.parameters['luminosity_distance']
            params['geocent_time'] = self.parameters['geocent_time'] 
            params['n_phase'] = self.parameters['n_phase']

            LogLikelisRatio[i] = self.log_likelihood_ratio_computation(params)

        # take the average over all samples to compute the Golum likelihood
        GolumLogLikelihoodRatio = scipy.special.logsumexp(LogLikelisRatio - np.log(len(LogLikelisRatio)))

        return GolumLogLikelihoodRatio

    def log_likelihood(self):
        """
        Function computing the log likelihood ratio using all the functions
        defined above 
        """
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

class DiscreteGravitationalWaveTransientImage2(GravitationalWaveTransientImage2_noLookup):
	"""
	class computing the likelihood in a discrete way
	"""

	def log_likelihood_ratio(self):
		"""
		Function computing the log likelihood ratio using the Golum 
		approximation 
		"""

		idx = int(self.parameters['indices'])
		params = dict()
		for key in self.samples:
			params[key] = self.samples[key][idx]
		params['luminosity_distance'] = np.sqrt(self.parameters['relative_magnification'])*params['luminosity_distance']
		params['geocent_time'] = self.parameters['delta_t'] + params['geocent_time']
		
		params['n_phase'] = self.parameters['delta_n'] + params['n_phase']

		LogLRatio = self.log_likelihood_ratio_computation(params)

		return LogLRatio

class DiscreteGravitationalWaveTransientImage2LookUp(GravitationalWaveTransientImage2):
    """
    Class computing the discrete likelihood but now using also 
    the lookup table (should be even faster)
    """

    def log_likelihood_ratio(self):
        """
        Overwritting the corresponding function to accomodate for
        other independant variables
        """
        if self.parameters['delta_n'] == 0:
            SNR_tab = self.snr_table_0.copy()
        elif self.parameters['delta_n'] == 0.5:
            SNR_tab = self.snr_table_0p5.copy()
        elif self.parameters['delta_n'] == 1:
            SNR_tab = self.snr_table_1.copy()
        elif self.parameters['delta_n'] == 1.5:
            SNR_tab = self.snr_table_1p5.copy()
        else:
            print("Unknown n_phase ...")

        SNR_tab_id = SNR_tab[int(self.parameters['indices'])]
        LogLikeliRatio = self.log_likelihood_ratio_computation_lookuptable((self.parameters['delta_t'] + self.samples['geocent_time'][int(self.parameters['indices'])]),
                           self.parameters['relative_magnification'], SNR_tab_id)

        return LogLikeliRatio


class JointGravitationalWaveTransient(Likelihood):
    """
    Class to compute the joint likelihood for 
    different lensed events
    """

    @attr.s
    class _CalculatedSNRs:
        d_inner_h = attr.ib()
        optimal_snr_squared = attr.ib()
        complex_matched_filter_snr = attr.ib()

    def __init__(self, interferometers_img1, interferometers_img2, waveform_generator, priors = None):
        """
        Initialization for the class

        ARGS:
        -----
        - interferometers_img1: the bilby interferometers to be used
                                 for the first image
        - interferometers_img2: the bilby interferometers to be used
                                for the second image
        - waveform_generator: the bilby waveform generator object used
                              for the analysis
        - priors: the priors used for the run, default is None
        """
        self.waveform_generator = waveform_generator
        super(JointGravitationalWaveTransient, self).__init__()
        self.img1_ifos = interferometers_img1
        self.img2_ifos = interferometers_img2
        self.priors = priors

        # to be filled by the sampler
        self.parameters = dict()

        # compute the noise log likelihood for the two events
        self.noise_log_l_imgs = self.noise_log_likelihood()

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
        for ifo in self.img1_ifos:
            mask = ifo.frequency_mask
            log_l -= noise_weighted_inner_product(ifo.frequency_domain_strain[mask],
                                                  ifo.frequency_domain_strain[mask],
                                                  ifo.power_spectral_density_array[mask],
                                                  self.waveform_generator.duration)/2.

        # for the second image 
        for ifo in self.img2_ifos:
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

    def log_likelihood_ratio(self):
        """
        Function computing the log likelihood ratio 
        for the two lensed images. 
        """
        params = dict()
        for key in self.parameters.keys():
            if key not in ['relative_magnification', 'delta_t', 'delta_n']:
                params[key] = self.parameters[key]

        waveform_polarizations_img1 = self.waveform_generator.frequency_domain_strain(params)

        # make the polarizations for the second image
        waveform_polarizations_img2 = dict()
        for key in waveform_polarizations_img1:
            waveform_polarizations_img2[key] = (np.sqrt(self.parameters['relative_magnification'])**(-1))*waveform_polarizations_img1[key]*\
                                               np.exp(-2j*np.pi*self.parameters['delta_t']*self.waveform_generator.frequency_array -
                                                      1j*np.pi*self.parameters['delta_n'])

        d_inner_h_img1 = 0.
        optimal_snr_squared_img1 = 0.
        complex_matched_filter_snr_img1 = 0.

        for ifo in self.img1_ifos:
            per_detector_snr = self.calcualte_snrs(waveform_polarizations = waveform_polarizations_img1,
                                                  interferometer = ifo, parameters = params)
            d_inner_h_img1 += per_detector_snr.d_inner_h
            optimal_snr_squared_img1 += per_detector_snr.optimal_snr_squared
            complex_matched_filter_snr_img1 += per_detector_snr.complex_matched_filter_snr

        Log_L_img1 = float((np.real(d_inner_h_img1) - optimal_snr_squared_img1 /2.).real)

        # do the same but for the second image
        d_inner_h_img2 = 0.
        optimal_snr_squared_img2 = 0.
        complex_matched_filter_snr_img2 = 0.

        params_2 = params.copy()
        params_2['luminosity_distance'] = np.sqrt(self.parameters['relative_magnification'])\
                                          *params['luminosity_distance']
        params_2['n_phase'] = params['n_phase'] + self.parameters['delta_n']
        params_2['geocent_time'] = params['geocent_time'] + self.parameters['delta_t']
        for ifo in self.img2_ifos:
            per_detector_snr = self.calcualte_snrs(waveform_polarizations = waveform_polarizations_img2,
                                                   interferometer = ifo, parameters = params_2)
            d_inner_h_img2 += per_detector_snr.d_inner_h
            optimal_snr_squared_img2 += per_detector_snr.optimal_snr_squared
            complex_matched_filter_snr_img2 += per_detector_snr.complex_matched_filter_snr

        Log_L_img2 = float((np.real(d_inner_h_img2) - optimal_snr_squared_img2 /2.).real)


        return Log_L_img1+Log_L_img2

    def log_likelihood(self):
        """
        Function computing the log likelihood for the event 
        pair using all the different info defined above
        """

        return self.log_likelihood_ratio() + self.noise_log_l_imgs


def reweight_samples_joint_likelihood(ifos1, ifos2, waveform_generator, img1_result, golum_result, n_points = int(1e5)):
    """
    Function doing the reweighing of the samples 
    using the joint likelihood samples. This is a way 
    to account for the information of the two events 
    in the final reweighing as to not just add up 
    the errors.

    ARGS:
    -----
    - ifos1: the list of interferometers used to 
             analyze the first image
    - ifos2: the list of interferometers used to 
             analyze the second image 
    - waveform_generator: bilby waveform generator
                          with the lensed waveform 
    - img1_result: the result object for the first image 
                       which should be used for the 
                       reweighing
    - golum_result: the golum result object to be used
                    for the reweighing 

    RETURNS:
    --------
    - img1_samps, golum_samps, log_weights: the samples 
     for the two images and their corresponding weights
     for the reweighing

    """

    # setup the joint PE likelihood 
    JPE_likeli = JointGravitationalWaveTransient(ifos1, ifos2, waveform_generator,
                                                 priors = None)

    img1_posts = img1_result.posterior.copy()
    gol_posts = golum_result.posterior.copy()

    # take random samples out of the posterios
    idxs1 = np.random.choice(len(img1_posts['geocent_time']), size = int(n_points))
    idxs2 = np.random.choice(len(gol_posts['delta_t']), size = int(n_points))
    log_weights = np.zeros(len(idxs2))

    # take the random samples 
    img1_samps = dict()
    gol_samps = dict()
    for key in img1_posts.keys():
        img1_samps[key] = [img1_posts[key][idd] for idd in idxs1]
    for key in gol_posts.keys():
        gol_samps[key] = [gol_posts[key][idd] for idd in idxs2]

    # compute the associated weight
    for i in range(len(idxs1)):
        if i%1000 == 0:
            print("Computing weight for samples %i / %i"%(i, len(idxs1)))

        id1 = idxs1[i]
        id2 = idxs2[i]
        log_likeli_img1 = img1_posts['log_likelihood'][id1]
        log_likeli_gol = gol_posts['log_likelihood'][id2]
        params = dict()
        for key in img1_posts.keys():
            if 'log_' not in key:
                params[key] = img1_posts[key][id1]
        for key in gol_posts.keys():
            if 'log_' not in key:
                params[key] = gol_posts[key][id2]

        # set the parameters for the joint likelihood parameters
        JPE_likeli.parameters = params.copy()
        log_likeli_JPE = JPE_likeli.log_likelihood()
        log_weights[i] = log_likeli_JPE - log_likeli_img1 - log_likeli_gol

    return img1_samps, gol_samps, log_weights

class SymmetricGravitationalWaveTransient(object):
    """
    Symetric version of the GOLUM likelihood. This
    does the analyis of one image using the other 
    in the most complete way. It should be much 
    closer of the result coming out of joint PE.
    """

    def __init__(self, interferometers_img1, file_img1, waveform_generator_img1, 
                 interferometers_img2, file_img2, waveform_generator_img2 = None,
                 n_samp_gol = 2500, n_samp_rew = int(1e5), seed = None,
                 priors = None, outdir = 'Outdir_GOLUM_run', label = 'GOLUM_run',
                 sampler = 'pymultinest', nact = 1, npool = 1, npoints = 2048,
                 maxmcmc = 10000):
        """
        Initialization of the class
        
        ARGS:
        -----
        - interferometers_img1: list of interferometers involved in the
                                observation of the first image
        - interferometers_img2: list of interferometers involved in the
                                observation of the second image
        - waveform_generator_img1: bilby waveform generator object used to analyze
                              the data for the first image. Should be using the LensedBBHmodel 
                              to account correctly for the Morse factor
        - waveform_generator_img2: same as for image 1 but for image 2. Default 
                                   is None, in which case we just use the same
                                   waveform generator for the two images
        - posteriors_img1: dictionary of the posterior samples coming from the 
                           16D run for the first image
        - posteriors_img2 : dictionary of the posterior samples coming from the 
                            16D run for the first image
        - n_samp_gol: int, number of samples to take out of the two posterior distributions
                  to do the GOLUM runs 
        - n_samp_rew; int, the number of samples to use when doing the reweighing processes
        - seed: int, default is None, can be set to take the same samples
                when starting the GOLUM run.
        - priors: the Bilby prior dict for the lensing parameters
        - outdir: name of the outdirectories in which the files will 
                  be stored 
        - label: extension to be given to the file names
        
        """

        super(SymmetricGravitationalWaveTransient, self).__init__()

        self.ifos_img1 = interferometers_img1
        self.ifos_img2 = interferometers_img2
        self.wf_gen_img1 = waveform_generator_img1
        if waveform_generator_img2 is None:
            self.wf_gen_img2 = waveform_generator_img1
        else:
            self.wf_gen_img2 = waveform_generator_img2

        self.file_img1 = file_img1
        self.file_img2 = file_img2
        self.img1_parameters, self.posteriors_img1 = utils.read_image1_file(file = self.file_img1)
        self.img2_parameters, self.posteriors_img2 = utils.read_image1_file(file = self.file_img2)
        self.n_samp_gol = n_samp_gol
        self.n_samp_rew = n_samp_rew
        self.outdir = outdir
        self.label = label
        self.seed = seed

        self.priors = priors
        self.sampler = sampler
        self.nact = nact 
        self.maxmcmc = maxmcmc
        self.npool = npool
        self.npoints = int(npoints)

        # do each GOLUM analysis needed
        self.first_image_analysis()
        self.second_image_analysis()

        # compute the lensed evidence for the two runs
        self.log_evidence = self.compute_lensed_evidence()

        # make the final joint parameters
        self.joint_posteriors = self.make_joint_posteriors()


    def first_image_analysis(self):
        """
        Function performing the first GOLUM run
        """

        label_img2img1 = '%s_img2img1'%self.label
        PriorDict = dict()
        for key in self.priors:
            if "_2" in key:
                spl = key.split('_')
                PriorDict['%s_%s'%(spl[0], spl[1])] = self.priors[key]
        LensingPriors = bilby.core.prior.PriorDict(dictionary = PriorDict)
        
        # setup the GOLUM likelihood for this run 
        LensingLikelihood = GravitationalWaveTransientImage2(self.ifos_img2,
                                                                        waveform_generator = self.wf_gen_img2,
                                                                        posteriors = self.posteriors_img1,
                                                                        n_samp = self.n_samp_gol,
                                                                        seed = self.seed,
                                                                        priors = LensingPriors)

        # compute the lensing parameters if needed
        if self.img1_parameters is not None and self.img2_parameters is not None:
            injected_parameters = dict()

        injected_parameters['relative_magnification'] = (self.img2_parameters['luminosity_distance']/self.img1_parameters['luminosity_distance'])**2
        injected_parameters['delta_t'] = self.img2_parameters['geocent_time'] - self.img1_parameters['geocent_time']
        injected_parameters['delta_n'] = self.img2_parameters['n_phase'] - self.img1_parameters['n_phase']

        if injected_parameters['delta_n'] == -0.5:
            injected_parameters['delta_n'] = 1.5
        elif injected_parameters['delta_n'] == -1:
            injected_parameters['delta_n'] = 1


        # run the sampler for this analysis
        self.results_img2img1 = bilby.run_sampler(likelihood = LensingLikelihood, priors = LensingPriors,
                                                  sampler = self.sampler, npoints = self.npoints,
                                                  nact = self.nact, npool = self.npool, 
                                                  maxmcmc = self.maxmcmc, outdir = self.outdir,
                                                  label = label_img2img1,
                                                  injection_parameters = injected_parameters)

        # plot the corner plot as well
        self.results_img2img1.plot_corner()

        # do the reweighing 
        self.img1_posteriors_img2img1, self.gol_posteriors_img2 = utils.WeightsEvent1(self.results_img2img1, LensingLikelihood,
                                                                                 self.ifos_img2, self.wf_gen_img2,
                                                                                 outdir = self.outdir, label = label_img2img1,
                                                                                 im1_posteriors = self.file_img1)

    def second_image_analysis(self):
        """
        Function performing the second GOLUM run
        """

        label_img1img2 = '%s_img1_img2'%self.label
        PriorDict = dict()
        for key in self.priors:
            if '_1' in key:
                splt = key.split('_')
                PriorDict['%s_%s'%(splt[0], splt[1])] = self.priors[key]
        LensingPriors = bilby.core.prior.PriorDict(dictionary = PriorDict)

        # setup the GOLUM likelihood for the second run 
        LensingLikelihood = GravitationalWaveTransientImage2(self.ifos_img1,
                                                                        waveform_generator = self.wf_gen_img1,
                                                                        posteriors = self.posteriors_img2,
                                                                        n_samp = self.n_samp_gol,
                                                                        seed = self.seed,
                                                                        priors = LensingPriors)

        # compute the lensing parameters if needed
        if self.img1_parameters is not None and self.img2_parameters is not None:
            injected_parameters = dict()
            injected_parameters['relative_magnification'] = (self.img1_parameters['luminosity_distance']/self.img2_parameters['luminosity_distance'])**2
            injected_parameters['delta_t'] = self.img1_parameters['geocent_time'] - self.img2_parameters['geocent_time']
            injected_parameters['delta_n'] = self.img1_parameters['n_phase'] - self.img2_parameters['n_phase']

            if injected_parameters['delta_n'] == -0.5:
                injected_parameters['delta_n'] = 1.5
            elif injected_parameters['delta_n'] == -1:
                injected_parameters['delta_n'] = 1

        # run the sampler for tha anlysis
        self.results_img1img2 = bilby.run_sampler(likelihood = LensingLikelihood,
                                                  priors = LensingPriors,
                                                  sampler = self.sampler, npoints = self.npoints,
                                                  nact = self.nact, npool = self.npool, 
                                                  maxmcmc = self.maxmcmc, outdir = self.outdir,
                                                  label = label_img1img2,
                                                  injection_parameters = injected_parameters)

        # make the corner plot for this image as wel
        self.results_img1img2.plot_corner()

        # do the reweighing
        self.img1_posteriors_img1img2, self.gol_posteriors_img1 = utils.WeightsEvent1(self.results_img1img2, LensingLikelihood,
                                                                                 self.ifos_img1, self.wf_gen_img1,
                                                                                 outdir = self.outdir, label = label_img1img2,
                                                                                 im1_posteriors = self.file_img2)
    def compute_lensed_evidence(self):
        """
        Function computing the lensed evidence
        based on the two runs
        """
        res_lens_img1 = bilby.result.read_in_result(filename = self.file_img1)
        res_lens_img2 = bilby.result.read_in_result(filename = self.file_img2)
        log_p_img2img1 = res_lens_img1.log_evidence + self.results_img2img1.log_evidence
        log_p_img1img2 = res_lens_img2.log_evidence + self.results_img1img2.log_evidence
        log_zs_arg = scipy.special.logsumexp([log_p_img2img1, log_p_img1img2])

        return np.log(1./2)+log_zs_arg

    def make_joint_posteriors(self):
        """
        Function computing the joint posteriors
        based on the two runs
        """

        # first need to compute the weights
        # and samples for the joint run 
        im1_results = bilby.result.read_in_result(filename = self.file_img1)
        img1_samps_img2img1, gol_samps_img2img1, log_w_img2img1 = reweight_samples_joint_likelihood(self.ifos_img1,
                                                                  self.ifos_img2, self.wf_gen_img2, im1_results,
                                                                  self.results_img2img1)

        im2_results = bilby.result.read_in_result(filename = self.file_img2)
        img1_samps_img1img2, gol_samps_img1img2, log_w_img1img2 = reweight_samples_joint_likelihood(self.ifos_img2,
                                                                    self.ifos_img1, self.wf_gen_img1, im2_results,
                                                                    self.results_img1img2)

        # need to put all the weights and posteriors together 
        all_log_w = []
        all_samps_img1 = {key : [] for key in img1_samps_img1img2.keys()}
        all_samps_gol = {key : [] for key in gol_samps_img1img2.keys()}

        for i in range(len(log_w_img2img1)):
            all_log_w.append(log_w_img2img1[i])
        for key in all_samps_img1:
            all_samps_img1[key].append(img1_samps_img2img1[key][i])
        for key in all_samps_gol:
            all_samps_gol[key].append(gol_samps_img2img1[key][i])

        # other image but alsi with correct conversion applied
        for i in range(len(log_w_img1img2)):
            all_log_w.append(log_w_img1img2[i])
            for key in all_samps_img1:
                all_samps_img1[key].append(img1_samps_img1img2[key][i])
            all_samps_gol['relative_magnification'].append(1./gol_samps_img1img2['relative_magnification'][i])
            all_samps_gol['delta_t'].append((-1)*gol_samps_img1img2['delta_t'])

            if ((-1)*gol_samps_img1img2['delta_n'][i]) == -0.5:
                all_samps_gol['delta_n'].append(1.5)
            elif ((-1)*gol_samps_img1img2['delta_n'][i]) == -1:
                all_samps_gol['delta_n'].append(1)
            else:
                all_samps_gol['delta_n'].append((-1)*gol_samps_img1img2['delta_n'][i])

        # choose between all the samples the 
        # ones that have the highest weight
        all_w = np.exp(all_log_w - max(all_log_w))
        idxs = np.random.choice(len(all_w), size = int(self.n_samp_rew),
                                p = all_w/np.sum(all_w))
        samps = dict()
        print("JJ  :  ", len(all_samps_img1['geocent_time']))
        print("JJ  :  ", len(all_samps_gol['delta_t']))
        for key in all_samps_img1.keys():
            samps[key] = [all_samps_img1[key][ii] for ii in idxs]
        for key in all_samps_gol.keys():
            if 'log_' not in key:
                print("JJ : ", key)
                samps[key] = [all_samps_gol[key][ii] for ii in idxs]

        return samps
