from bilby.gw.source import *
import numpy as np

def LensedBBHmodel(frequency_array, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
        n_phase, **kwargs):
    """
    Wrapper around lal_binary_black_hole to make a lensed BBH model that
    can be used to perform the analysis

    ARGS:
    -----
    - frequency_array: the array of frequencies at which the waveform should
                                       be built
    - mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2,
            phi_jl, theta_jn, phase: usual BBH parameters. These will directly be passed
                                                             to the lal_binary_black hole source model from bilby
                                                             The luminosity distance is supposed to be the apparent
                                                             luminosity distance for the image and geocent_time should
                                                             be the time at which the image is observed
    - n_phase: the Morse factor for the image
    - **kwargs: extra parameters that will be passed the lal_binary_black_hole source model
                            dict with optional keywords "waveform_approximant, reference_frequency
            minimum_frequency, maximum_frequency, catch_waveform_errors, pn_spin_order,
            pn_tidal_order, pn_phase_order, pn_amplitude_order, mode_array"

    RETURNS:
    --------
    - waveform: dict, plus and cross polarization for the lensed BBH merger
    """

    # load the usual model
    frequency_domain_source_model = lal_binary_black_hole

    # generate the unlensed waveform
    waveform = frequency_domain_source_model(frequency_array, mass_1, mass_2,
                                                                                     luminosity_distance, a_1, tilt_1,
                                                                                     phi_12, a_2, tilt_2, phi_jl, theta_jn,
                                                                                     phase, **kwargs)
    # check that waveform is not None (waveform error catching)
    if waveform is None:
        return waveform

    # lens the waveform to get the correct model
    deltaPhi = n_phase * (np.pi/2.)
    for pola in waveform:
        waveform[pola] = np.exp(-2*1j*deltaPhi)*waveform[pola]

    return waveform
