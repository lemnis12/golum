import bilby
from ..Lookup.LensingLookup import MorseFactorLookup, IndependantMorseFactorLookup
import numpy as np
import matplotlib.pyplot as plt
from bilby.core.utils import logger
import scipy.special
import matplotlib

try:
	from scipy.special import logsumexp
except ImportError:
	from scipy.misc import logsumexp

from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList
from bilby.gw.utils import noise_weighted_inner_product
from bilby.gw.waveform_generator import WaveformGenerator
from collections import namedtuple
from scipy.interpolate import CubicSpline
from bilby.core import utils
from bilby.core.series import CoupledTimeAndFrequencySeries
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.core.utils import logger

class GravitationalWaveTransientImage2(Likelihood):
	"""
	Class adapting the usual Bilby 'GravitationalWavetransient' to 
	a likelihood based on the Golum approximation for the lensing hypothesis
	and using the lookup table
	"""

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
		super(GravitationalWaveTransientImage2, self).__init__(dict())
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

		# make the lookup table for the different values of the Morse factor difference
		self.snr_table_0, self.time_array = MorseFactorLookup(self.interferometers, self.waveform_generator,
															  self.samples, 0., priors = self.priors)
		self.snr_table_0p5, _ = MorseFactorLookup(self.interferometers, self.waveform_generator,
												  self.samples, 0.5, priors = self.priors)
		self.snr_table_1, _ = MorseFactorLookup(self.interferometers, self.waveform_generator, 
												self.samples, 1, priors = self.priors)
		self.snr_table_1p5, _ = MorseFactorLookup(self.interferometers, self.waveform_generator, 
												  self.samples, 1.5, priors = self.priors)

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

	def log_likelihood_ratio_computation_lookuptable(self, geocent_time, mu_rel, snr_dict):
		"""
		Function computing the value of the of the log likelihood for a given 
		set of samples
		"""
		d_inner_h = 0.
		optimal_snr_squared = 0.
		idx = self.find_nearest(self.time_array, geocent_time)
		sqrt_mu_rel = np.sqrt(mu_rel)
		for ifo in self.interferometers:
			d_inner_h += np.real(snr_dict['d_inner_h_{}_real'.format(ifo.name)][idx] + 1j* snr_dict['d_inner_h_{}_img'.format(ifo.name)][idx] )  / sqrt_mu_rel
			optimal_snr_squared += np.real(snr_dict['h_inner_h_{}'.format(ifo.name)]) / mu_rel

		LogL = np.real(d_inner_h) - optimal_snr_squared /2.

		return LogL


	def log_likelihood_ratio(self):
		"""
		Function computing the log_likelihood ratio for the lensed hypothesis based
		on the GOLUM approximation and the lookup table
		"""

		# load the lookup table for the Morse phase under consideration 
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

		LogLikelisRatio = [self.log_likelihood_ratio_computation_lookuptable((self.parameters['delta_t'] + self.samples['geocent_time'][i]),
							self.parameters['relative_magnification'], SNR_tab[i]) for i in range(self.n_samp)]

		# take the average to approximate the log likelihood ratio under GOLUM approximation 
		GolumLogLikelihoodRatio = scipy.special.logsumexp(LogLikelisRatio - np.log(len(LogLikelisRatio)))

		return GolumLogLikelihoodRatio	

	def log_likelihood(self):
		"""
		Function computing the log likelihood based on the functions
		defined above and using the lookup (which assumes a BBH lens model)
		"""
		return self.log_likelihood_ratio() + self.noise_log_likelihood()

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
    Class used to compute the likelihood for the second image when we use Dl2, tc2 and n_2 as 
    parameters. Here, we directly use the adapted lookup table
    """
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
            self.samples[key] = [self.posteriors[key][idx] for idx in self.indices]

        # define dict to be filled by the sampler
        self.parameters = dict()

        # make the lookup table for the different values of the Morse factor for the second
        # image
        self.snr_table_0, self.time_array = IndependantMorseFactorLookup(self.interferometers, self.waveform_generator,
                                                                         self.samples, 0., priors = self.priors)
        self.snr_table_0p5, _ = IndependantMorseFactorLookup(self.interferometers, self.waveform_generator,
                                                             self.samples, 0.5, priors = self.priors)
        self.snr_table_1, _ = IndependantMorseFactorLookup(self.interferometers, self.waveform_generator,
                                                           self.samples, 1, priors = self.priors)

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

    def log_likelihood_ratio_computation_lookuptable(self, geocent_time, luminosity_distance, snr_dict):
        """
        Adaptation of the same function from GravitationalWaveTransientImage2 
        to the case where we use tc2, Dl2, n_2
        """
        d_inner_h = 0.
        optimal_snr_squared = 0.
        idx = self.find_nearest(self.time_array, geocent_time)
        for ifo in self.interferometers:
            d_inner_h += np.real(snr_dict['d_inner_h_{}_real'.format(ifo.name)][idx] \
                         + 1j* snr_dict['d_inner_h_{}_img'.format(ifo.name)][idx])/luminosity_distance
            optimal_snr_squared += np.real(snr_dict['h_inner_h_{}'.format(ifo.name)]) / luminosity_distance**2

        LogL = np.real(d_inner_h) - optimal_snr_squared /2. 

        return LogL

    def log_likelihood_ratio(self):
        """
        Overwritting the corresponding function to accomodate for
        other independant variables
        """
        if self.parameters['n_phase'] == 0:
            SNR_tab = self.snr_table_0.copy()
        elif self.parameters['n_phase'] == 0.5:
            SNR_tab = self.snr_table_0p5.copy()
        elif self.parameters['n_phase'] == 1:
            SNR_tab = self.snr_table_1.copy()
        else:
            print("Unknown n_phase used ...")

        LogLikelisRatio = [self.log_likelihood_ratio_computation_lookuptable(self.parameters['geocent_time'],
                           self.parameters['luminosity_distance'], SNR_tab[i]) for i in range(self.n_samp)]

        GolumLogLikelihoodRatio = scipy.special.logsumexp(LogLikelisRatio - np.log(len(LogLikelisRatio)))

        return GolumLogLikelihoodRatio

    def log_likelihood(self):
        """
        Function computing the log likelihood based on the functions
        defined above and using the lookup (which assumes a BBH lens model)
        """
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

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