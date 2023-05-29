import bilby 
import numpy as np
from ..pe.prior import MorseFactorPrior
from ..pe.likelihood import JointGravitationalWaveTransient
from bilby.core.likelihood import Likelihood
import attr


def prob_n(val, ref_val):
    """
    Function computing the probability to have 
    a given value for n when we have the
    reference value 

    This is 1 if val = ref_val and 0 in all other 
    cases

    """

    if val == ref_val:
        return 1.
    else:
        return 0.


def compute_log_bayes_factors_type_II_vs_type_I_and_vs_type_III(file = None):
    """
    Function computing the value of the log Bayes factor for 
    a type II image vs a type I image, and the Bayes factor
    for a type II image vs a type III image.

    ARGUMENTS:
    ----------
    - file: the result file

    Returns:
    --------
    - logB_IIvsI, logB_IIvsIII: the log Bayes factors
    """

    # load the result from the file
    results = bilby.result.read_in_result(filename = file)
    
    n_ph_array = np.array(results.posterior['n_phase'])

    # compute the prior
    Nprior = MorseFactorPrior(name = 'n_phase', latex_label = '$n_1$')

    wII = np.where(n_ph_array == 0.5, 1, 0)/Nprior.prob(n_ph_array)
    wI = np.where(n_ph_array == 0, 1, 0)/Nprior.prob(n_ph_array)
    wIII = np.where(n_ph_array == 1, 1, 0)/Nprior.prob(n_ph_array)

    logZ_II = results.log_evidence + np.log(np.mean(wII))
    logZ_I = results.log_evidence + np.log(np.mean(wI))
    logZ_III = results.log_evidence + np.log(np.mean(wIII))

    logB_IIvsI = logZ_II - logZ_I
    logB_IIvsIII = logZ_II - logZ_III

    return logB_IIvsI, logB_IIvsIII

def reweight_samples_with_joint_likelihood(interferometers_image_1, interferometers_image_2, waveform_generator, img1_posteriors, golum_result, n_points = int(1e5)):
    """
    Function doing the reweighting for the GOLUM samples using the 
    joint likelihood. This is a way to account for the joint nature
    of the signals when doing the symmetric GOLUM run (instead of 
    just summing the errors up)

    ARGS:
    -----
    - interferometers_image_1: the bilby interferometer list used
                               to analyze the first image
    - interferometers_image_2: the bilby interferometer list used
                               to analyze the second image
    - waveform_generator: the bilby waveform generator used for
                          the analysis
    - image_1_result: the bilby result object for the first image
                      run to be used for the reweighting
    - golum_result: the golum result object to be used for the
                    reweighting 
    - n_points: int, the number of points to be used in the reweighting 
                process.

    RETURNS:
    -------
    - image_1_samples, golum_samples, log_weights: the  reweighted samples for the
                                                   first image, the lensing samples,
                                                   and the corresponding log weights
    """

    # setup the joint pe likelihood 
    joint_pe_likelihood = JointGravitationalWaveTransient(interferometers_image_1,
                                                          interferometers_image_2,
                                                          waveform_generator,
                                                          priors = None)
    image_1_posteriors = img1_posteriors.copy()
    golum_posteriors = golum_result.posterior.copy()

    # take random samples out of the posteriors
    idxs1 = np.random.choice(len(image_1_posteriors['geocent_time']), size = int(n_points))
    idxs2 = np.random.choice(len(golum_posteriors['delta_t']), size = int(n_points))
    log_weights = np.zeros(len(idxs2))

    # take random samples
    image_1_samples = dict()
    golum_samples = dict()
    for key in image_1_posteriors.keys():
        image_1_samples[key] = [image_1_posteriors[key][idd] for idd in idxs1]
    for key in golum_posteriors.keys():
        golum_samples[key] = [golum_posteriors[key][idd] for idd in idxs2]

    # compute the associated weights
    for i in range(len(idxs1)):
        if i%1000 == 0:
            print("Computing weights for samples %i / %i"%(i, len(idxs1)))

        id1 = idxs1[i]
        id2 = idxs2[i]
        log_likelihood_image_1 = image_1_posteriors['log_likelihood'][id1]
        log_likelihood_golum = golum_posteriors['log_likelihood'][id2]
        params = dict()
        for key in image_1_posteriors.keys():
            if 'log_' not in key:
                params[key] = image_1_posteriors[key][id1]
        for key in golum_posteriors.keys():
            if 'log_' not in key:
                params[key] = golum_posteriors[key][id2]

        # set the parameters for the joint likelihood parameters
        joint_pe_likelihood.parameters = params.copy()
        log_likelihood_joint_pe = joint_pe_likelihood.log_likelihood()
        log_weights[i] = log_likelihood_joint_pe - log_likelihood_image_1 \
                         - log_likelihood_golum

    return image_1_samples, golum_samples, log_weights


def reweight_symmetric_to_joint_samples(img1_posteriors, img2_posteriors, results_img1img2, results_img2img1, ifos_img1, ifos_img2, waveform_generator, n_points = int(1e5)):
    """
    Function to reweight the posteriors coming from the symmetric run to joint posteriors

    ARGUMENTS:
    ----------
    - img1_posteriors: posterior dictionary for the first image
    - img2_posteriors: posterior dictionary for the second image
    - results_img1img2: bilby result object for the img1 - img2 run
    - results_img2img1: bilby result object for the img2 - img1 run
    - ifos_img1: the interferometers for the first image
    - ifos_img2: the interferometers for the second image
    - waveform_generator: the waveform generator used in the analysis
    - n_points: the number of points used in the reweighting

    RETURNS:
    --------
    - joint_samples: a dictionary with the joint samples
    """

    # combine the samples together
    if "relative_magnification" in results_img1img2.posterior.keys():
        effective_parameters = False
    else:
        effective_parameters = True

    image_1_samples_img2img1, golum_samples_img2img1, log_weights_img2img1 = reweight_samples_with_joint_likelihood(ifos_img1, ifos_img2, waveform_generator,
                                                                                                                    img1_posteriors, results_img2img1,
                                                                                                                    n_points = n_points)
    image_1_samples_img1img2, golum_samples_img1img2, log_weights_img1img2 = reweight_samples_with_joint_likelihood(ifos_img2, ifos_img1, waveform_generator,
                                                                                                                    img2_posteriors, results_img1img2,
                                                                                                                    n_points = n_points)
    
    # put all the weights and samples together
    all_log_weights = []
    all_samples_image_1 = {key : [] for key in image_1_samples_img1img2.keys()}
    all_samples_golum = {key : [] for key in golum_samples_img1img2.keys()}

    # first image is taken as reference
    for i in range(len(log_weights_img2img1)):
        all_log_weights.append(log_weights_img2img1[i])
        for key in image_1_samples_img2img1.keys():
            if 'log_' not in key:
                all_samples_image_1[key].append(image_1_samples_img2img1[key][i])
        for key in golum_samples_img2img1.keys():
            if 'log_' not in key:
                all_samples_golum[key].append(golum_samples_img2img1[key][i])

    # for the second image, need to also do
    # the correct conversions for the parameters affected by lensing
    for i in range(len(log_weights_img1img2)):
        all_log_weights.append(log_weights_img1img2[i])
        for key in image_1_samples_img1img2.keys():
            if 'log_' not in key and key not in ['luminosity_distance', 'geocent_time', 'n_phase']:
                all_samples_image_1[key].append(image_1_samples_img1img2[key][i])

        # lensing parameters all need to be modified
        all_samples_golum['relative_magnification'].append(1./golum_samples_img1img2['relative_magnification'][i])
        all_samples_golum['delta_t'].append((-1)*golum_samples_img1img2['delta_t'][i])

        if ((-1)*golum_samples_img1img2['delta_n'][i]) == -0.5:
            all_samples_golum['delta_n'].append(1.5)
        elif ((-1)*golum_samples_img1img2['delta_n'][i]) == -1:
            all_samples_golum['delta_n'].append(1)
        elif ((-1)*golum_samples_img1img2['delta_n']) == -1.5:
            all_samples_golum['delta_n'].append(1.5)
        else:
            all_samples_golum['delta_n'].append((-1)*golum_samples_img1img2['delta_n'][i])

    # make the posteriors for the luminosity distance
    idxs_dl2 = np.random.choice(len(image_1_samples_img1img2['luminosity_distance']),
                                size = len(image_1_samples_img1img2['luminosity_distance']))
    idxs_mu_rel = np.random.choice(len(golum_samples_img1img2['relative_magnification']),
                                   size = len(image_1_samples_img1img2['luminosity_distance']))

    image_1_samples_img1img2['luminosity_distance'] = np.array(image_1_samples_img1img2['luminosity_distance'])[idxs_dl2] * \
                                                      np.sqrt(np.array(golum_samples_img1img2['relative_magnification'])[idxs_mu_rel])

    for i in range(len(image_1_samples_img1img2['luminosity_distance'])):
        all_samples_image_1['luminosity_distance'].append(image_1_samples_img1img2['luminosity_distance'][i])

    # make the posteriors for the geocentric time
    idxs_tc2 = np.random.choice(len(image_1_samples_img1img2['geocent_time']),
                                size = len(image_1_samples_img1img2['geocent_time']))
    idxs_dt = np.random.choice(len(golum_samples_img1img2['delta_t']),
                               size = len(image_1_samples_img1img2['geocent_time']))
    image_1_samples_img1img2['geocent_time'] = np.array(image_1_samples_img1img2['geocent_time'])[idxs_tc2] + \
                                               np.array(golum_samples_img1img2['delta_t'])[idxs_dt]
    for i in range(len(image_1_samples_img1img2['geocent_time'])):
        all_samples_image_1['geocent_time'].append(image_1_samples_img1img2['geocent_time'][i])

    # make the posterior for the Morse factor
    idxs_n2 = np.random.choice(len(image_1_samples_img1img2['n_phase']),
                               size = len(image_1_samples_img1img2['n_phase']))
    idxs_dn = np.random.choice(len(golum_samples_img1img2['delta_n']),
                               size = len(image_1_samples_img1img2['n_phase']))

    image_1_samples_img1img2['n_phase'] = np.array(image_1_samples_img1img2['n_phase'])[idxs_n2] - \
                                          np.array(golum_samples_img1img2['delta_n'])[idxs_dn]

    for i in range(len(image_1_samples_img1img2['n_phase'])):
        if image_1_samples_img1img2['n_phase'][i] < -0.5:
            all_samples_image_1['n_phase'].append(image_1_samples_img1img2['n_phase'][i]+2)
        elif image_1_samples_img1img2['n_phase'][i] == -0.5:
            all_samples_image_1['n_phase'].append(0.5)
        else:
            all_samples_image_1['n_phase'].append(image_1_samples_img1img2['n_phase'][i])

    # now that all the conversions have been done
    # we can combine the samples and take those with the
    # highest weights
    all_weights = np.exp(all_log_weights - max(all_log_weights))
    idxs = np.random.choice(len(all_weights), size = n_points,
                            p = all_weights/np.sum(all_weights))
    samples = dict()
    

    for key in all_samples_image_1.keys():
        if 'log_' not in key:
            samples[key] = [all_samples_image_1[key][idd] for idd in idxs]
    for key in all_samples_golum.keys():
        if 'log_' not in key:
            samples[key] = [all_samples_golum[key][idd] for idd in idxs]

    return samples
