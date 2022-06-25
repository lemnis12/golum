import bilby
import numpy as np

def find_nearest(a, a0):
    """
    Function to fine the index of the closest element in an 
    array.

    ARGS:
    -----
    - a: the array in which we want to search
    - a0: the value of the element for which we want to find the
           closest element

    RETURNS:
    --------
    - idx: the index corresponding to the closest value to the 
           element we want
    """
    idx = np.abs(a-a0).argmin()
    return idx


def MorseFactorLookup(interferometers, waveform_generator, samples, delta_n, priors = None):
    """
    Function making the lookup table as described in https://arxiv.org/pdf/2105.04536.pdf

    ARGS:
    -----
    - interferometers: the list of Bilby interferometers objects used for 
                       the analysis
    - waveform_generator: bilby waveform generator object used
    - samples: the samples from the first image used to do the 
               sub_sampling
    - delta_n: the value of the Morse factor difference for which the lookup
               table should be computed
    - priors: default is None, the priors used for the run. If None, then 
              the lookup table will be computed for all the times in 
              the data stretch. Else, it is computed only for the times
              relevant to the priors

    RETURNS:
    --------
    - snr_table: list of length N, each entry is a dictionary with the 
                 sum of the inner products for the samples
    - time_array: list where each point i is the GPS time for which the
                  information has been computed 
    """

    print("Lookup table generation for Delta_n = %.1f"%(delta_n))

    snr_table = []
    time_array = interferometers.start_time + np.linspace(0, interferometers.duration,
                                              int(interferometers.duration*waveform_generator.sampling_frequency + 1))[:-1]

    if priors is not None:
        print("INFO: Lookup table is computed for reduced time range based on priors")
        t_min = min(samples['geocent_time']) + priors['delta_t'].minimum
        t_max = max(samples['geocent_time']) + priors['delta_t'].maximum
        idx_min = find_nearest(time_array, t_min)
        idx_max = find_nearest(time_array, t_max)

    else:
        print("INFO: Lookup table is computed for full time, no prior given")
        idx_min = 0
        idx_max = -1

    for i in range(len(samples['geocent_time'])):
        snr_arrays = dict()
        snr_arrays['d_inner_h'] = 0.
        snr_arrays['h_inner_h'] = 0.
        params = dict()

        # sample upon the calibration priors if not
        # None
        if priors is not None:
            params = priors.sample()
            for k in list(params.keys()):
                if 'recalib' not in k:
                    params.pop(k)

        # load the values of the parameters from the samples
        for key in samples:
            params[key] = samples[key][i]

        # set the geocentric time to be the start of the array
        params['geocent_time'] = time_array[0]

        # need to modify the Morse factor to be the one of the second event
        params['n_phase'] = delta_n + params['n_phase']
        waveform_polarizations = waveform_generator.frequency_domain_strain(params)

        for ifo in interferometers:
            signal = dict()
            for mode in waveform_polarizations.keys():
                det_response = ifo.antenna_response(params['ra'], params['dec'],
                                                    params['geocent_time'],
                                                    params['psi'], mode)
                signal[mode] = waveform_polarizations[mode]*det_response

            signal_ifo = sum(signal.values())
            signal_ifo *= ifo.strain_data.frequency_mask
            time_shift = ifo.time_delay_from_geocenter(params['ra'], params['dec'],
                                                      params['geocent_time'])
            dt_geocent = params['geocent_time'] - ifo.strain_data.start_time
            dt = dt_geocent + time_shift

            signal_ifo[ifo.strain_data.frequency_mask] = signal_ifo[ifo.strain_data.frequency_mask] * np.exp(
                                                    -1j*2*np.pi*dt*ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask])
            signal_ifo[ifo.strain_data.frequency_mask] *= ifo.calibration_model.get_calibration_factor(
                                                            ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
                                                            prefix = 'recalib_{}_'.format(ifo.name), **params)
            w_plus = 2./waveform_generator.duration * ifo.frequency_domain_strain * signal_ifo.conjugate() / ifo.power_spectral_density_array
            w = np.zeros(2*(len(w_plus)-1), dtype = complex)
            w[0:len(w_plus)-1] = w_plus[:-1]
            w[len(w_plus)-1:] = np.flip(w_plus.conjugate())[:-1]

            snr_arrays['h_inner_h'] += ifo.optimal_snr_squared(signal = signal_ifo)
            z = len(w)*np.fft.ifft(w)
            snr_arrays['d_inner_h'] += (np.real(z)[idx_min:idx_max] + 1j*np.imag(z)[idx_min:idx_max])
        snr_table.append(snr_arrays)
    time_array = time_array[idx_min:idx_max]

    return snr_table, time_array

def IndependantMorseFactorLookup(interferometers, waveform_generator, samples, n, priors = None):
    """
    Function making the lookup as described in https://arxiv.org/pdf/2105.04536.pdf
    but adapted to account for a sampling in apparent time
    and distance for the second event

    ARGS:
    -----
    - interferometers: the list of Bilby interferometers objects used for 
                       the analysis
    - waveform_generator: bilby waveform generator object used
    - samples: the samples from the first image used to do the 
               sub_sampling
    - n: the Morse factor for the second event
    - priors: default is None, the priors used for the run.
              If None, the lookup table is computed for all the times
              in the data stretch. Else, it is computed only for
              the time relevant when compared to the priors

    RETURNS:
    --------
    - snr_table: list of length N, where each entry 
                 is a dictionary with the inner product of 
                 the data and the template and the template with itself
    - time_array: list whear each point i is the GPS time
                  for which the information has been computed
    """

    print("Lookup table generation for n_2 = %.1f"%n)

    snr_table = []
    time_array = interferometers.start_time + np.linspace(0, interferometers.duration,
                                              int(interferometers.duration*waveform_generator.sampling_frequency + 1))[:-1]

    if priors is not None:
        print("INFO: Lookup table is computed for reduced time range based on priors")
        t_min = priors['geocent_time'].minimum
        t_max = priors['geocent_time'].maximum
        idx_min = find_nearest(time_array, t_min)
        idx_max = find_nearest(time_array, t_max)

    else:
        print("INFO: Lookup table is computed for full time, no prior given")
        idx_min = 0
        idx_max = -1

    for i in range(len(samples['geocent_time'])):
        snr_arrays = dict()
        snr_arrays['d_inner_h'] = 0.
        snr_arrays['h_inner_h'] = 0.
        params = dict()

        # samples upon the calibration parameters if needed
        if priors is not None:
            params = priors.sample()
            for k in list(params.keys()):
                if 'recalib' not in k:
                    params.pop(k)

        # load the values of the parameters for the samples
        for key in samples:
            params[key] = samples[key][i]

        # set the different elements to their reference value
        params['geocent_time'] = time_array[0]
        params['n_phase'] = n
        params['luminosity_distance'] = 1. 

        waveform_polarizations = waveform_generator.frequency_domain_strain(params)

        for ifo in interferometers:
            signal = dict()
            for mode in waveform_polarizations.keys():
                det_response = ifo.antenna_response(params['ra'], params['dec'], 
                                                    params['geocent_time'], 
                                                    params['psi'], mode)
                signal[mode] = waveform_polarizations[mode]*det_response

            signal_ifo = sum(signal.values())
            signal_ifo *= ifo.strain_data.frequency_mask
            time_shift = ifo.time_delay_from_geocenter(params['ra'], params['dec'],
                                                       params['geocent_time'])
            dt_geocent = params['geocent_time'] - ifo.strain_data.start_time
            dt = dt_geocent + time_shift

            signal_ifo[ifo.strain_data.frequency_mask] = signal_ifo[ifo.strain_data.frequency_mask] * np.exp(
                                                    -1j*2*np.pi*dt*ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask])
            signal_ifo[ifo.strain_data.frequency_mask] *= ifo.calibration_model.get_calibration_factor(
                                                            ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
                                                            prefix = 'recalib_{}_'.format(ifo.name), **params)
            w_plus = 2./waveform_generator.duration * ifo.frequency_domain_strain * signal_ifo.conjugate() / ifo.power_spectral_density_array
            w = np.zeros(2*(len(w_plus)-1), dtype = complex)
            w[0:len(w_plus)-1] = w_plus[:-1]
            w[len(w_plus)-1:] = np.flip(w_plus.conjugate())[:-1]

            snr_arrays['h_inner_h'] += ifo.optimal_snr_squared(signal = signal_ifo)
            z = len(w)*np.fft.ifft(w)
            snr_arrays['d_inner_h'] += (np.real(z)[idx_min:idx_max] + 1j*np.imag(z)[idx_min:idx_max])
        snr_table.append(snr_arrays)
    time_array = time_array[idx_min:idx_max]

    return snr_table, time_array