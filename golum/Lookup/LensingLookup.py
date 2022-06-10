import bilby
import numpy as np


def MorseFactorLookup(interferometers, waveform_generator, samples, delta_n, priors = None):
    """
    Fuction making the lookup as described in ADD REF

    ARGS:
    -----
    - interferometers: the list of Bilby interferometers objects used
                                              for the analysis
    - waveform_generator: bilby waveform generator, supposed to take the LensedBBHmodel as
                                              frequency domain model
    - samples: dictionary containing N samples for each of the parameters from the first
                       event
    - delta_n: the difference in Morse factr between the two events for which
                       the lookup table should be computed
    - priors: (default is None); Bilby prior dictionary used for the second event run.
                      It is mainly used to sample upon the calibration parameters

    Returns:
    --------
    - snr_table: list of length N; each entry is a dictionary with the inner product,
                              real and imagnary SNR array for each detector
    - time_array: list where each point i the GPS time for which the SNR has been
                              computed

    Note:
    -----
    - The samples should be coherent in between the parameters
      (thus with the same number of samples each). For an example of lookup table making,
      see golum.PE.LensingLikelihoods.GravitationalWaveTransientImage2()
    """

    print("Generating the Lookup table for n_phase = %.1f" %(delta_n))

    snr_table = []
    time_array = interferometers.start_time + np.linspace(0, interferometers.duration,
                                            int(interferometers.duration*waveform_generator.sampling_frequency + 1))[:-1]

    for i in range(len(samples['geocent_time'])):
        snr_arrays = dict()
        params = dict()
        # sample upon the calibration parameters if needed
        if priors is not None:
            params = priors.sample()
            for k in list(params.keys()):
                if 'recalib' not in k:
                    params.pop(k)
        # load the value of the parameters from the samples
        for key in samples:
            params[key] = samples[key][i]

        # set the geocentric time to be the start of the time array
        params['geocent_time'] = time_array[0]

        # need to modify the Morse factor to be the one of the second event
        params['n_phase'] = delta_n + params['n_phase']
        waveform_polarizations = waveform_generator.frequency_domain_strain(params)

        for ifo in interferometers:
            signal = dict()
            for mode in waveform_polarizations.keys():
                det_response = ifo.antenna_response(params['ra'], params['dec'],
                                                                                        params['geocent_time'], params['psi'],
                                                                                        mode)
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

            snr_arrays['h_inner_h_{}'.format(ifo.name)] = ifo.optimal_snr_squared(signal = signal_ifo)
            z = len(w)*np.fft.ifft(w)
            snr_arrays['d_inner_h_{}_real'.format(ifo.name)] = np.real(z)
            snr_arrays['d_inner_h_{}_img'.format(ifo.name)] = np.imag(z)
            snr_arrays['param_idx'] = i

        snr_table.append(snr_arrays)

    return snr_table, time_array

def IndependantMorseFactorLookup(interferometers, waveform_generator, samples, n, priors = None):
    """
    Function making the lookup table in the case where we want to use Dl2, n2 and t2 as functions

    ARGS:
    -----
    - interferometers: the list of Bilby interferometers objects used
                                              for the analysis
    - waveform_generator: bilby waveform generator, supposed to take the LensedBBHmodel as
                                              frequency domain model
    - samples: dictionary containing N samples for each of the parameters from the first
                       event
    - n: the dMorse factor for which the lookup table should be computed
    - priors: (default is None); Bilby prior dictionary used for the second event run.
                      It is mainly used to sample upon the calibration parameters

    RETURNS:
    --------
    - snr_table: list of length N; each entry is a dictionary with the inner product,
                              real and imagnary SNR array for each detector
    - time_array: list where each point i the GPS time for which the SNR has been
                              computed
    """

    print("Generating the lookup table for n_2 = %.1f"%n)

    snr_table = []
    time_array = interferometers.start_time + np.linspace(0, interferometers.duration,
                 int(interferometers.duration*waveform_generator.sampling_frequency + 1))[:-1]

    for i in range(len(samples['geocent_time'])):
        snr_arrays = dict()
        params = dict()

        # sample upon the calibration paramerers if needed
        if priors is not None:
            params = priors.sample()
            for k in list(params.keys()):
                if 'recalib' not in k:
                    params.pop(k)

        # load the values of the parameters from samples
        for key in samples:
            params[key] = samples[key][i]

        # set the geocentric time, Morse factor and Dl to correct vamues
        params['geocent_time'] = time_array[0]
        params['n_phase'] = n
        params['luminosity_distance'] = 1. # will be corrected later by DL_2 in code

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

            snr_arrays['h_inner_h_{}'.format(ifo.name)] = ifo.optimal_snr_squared(signal = signal_ifo)
            z = len(w)*np.fft.ifft(w)
            snr_arrays['d_inner_h_{}_real'.format(ifo.name)] = np.real(z)
            snr_arrays['d_inner_h_{}_img'.format(ifo.name)] = np.imag(z)
            snr_arrays['param_idx'] = i

        snr_table.append(snr_arrays)

    return snr_table, time_array