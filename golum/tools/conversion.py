import numpy as np
import scipy 
import bilby 

def change_in_prior(result_file, final_priors):
    """
    Function reweighing the posteriors for a 15D or 16D
    run one a change in posteriors. This enables one 
    to reweight the results to some standard prior and
    make for a more consistent comparison

    ARGS:
    -----
    - result_file: the bilby result file obtained 
                   via the unlensed run 
    - final_priors: Bilby prior dict corresponding to
                    the final desired priors

    RETURNS:
    --------
    - log_Z_final: the log evidence values for 
                   the new desired samples
    - samples_final: the samples reweighed to
                     account for the change in priors
    """

    # read the result file and extract the desired information 
    results = bilby.result.read_in_result(filename = result_file)
    old_priors = results.priors.copy()
    if 'geocent_time' not in list(final_priors.keys()) and 'geocent_time' in list(old_priors.keys()):
        old_priors.pop('geocent_time')
    ln_Z_old = results.log_evidence
    samples = results.posterior.copy()
    if 'geocent_time' not in list(final_priors.keys()) and 'geocent_time' in list(samples.keys()):
        samples.pop('geocent_time')
    log_priors = results.posterior['log_prior']
    for key in list(samples.keys()):
        if 'log_' in key or 'recalib' in key or 'SNR' in key or 'frequency' in key or 'waveform' in key:
            samples.pop(key)


    sub_samps = [{key : samples[key][i] for key in samples.keys()} 
                 for i in range(len(log_priors))]


    # compute the reweighted evidence
    p_samps_new = np.array([final_priors.prob(sub_samps[i]) for i in range(len(sub_samps))])
    weights = p_samps_new/np.exp(log_priors) 
    log_z_final = np.log(np.mean(weights)) + ln_Z_old

    # reweight the samples
    indices = len(weights)
    idxs = np.random.choice(indices, size = len(weights), p = weights/np.sum(weights))
    samples_final = dict()

    for key in samples:
        samples_final[key] = samples[key][idxs]

    return log_z_final, samples_final
