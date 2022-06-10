import bilby
import numpy as np
import matplotlib.pyplot as plt
from bilby.core.utils import logger
import scipy.special
import matplotlib
import json
from distutils.version import LooseVersion
import os
import copy
from ..PE.prior import Conditional_Dl2_prior_from_mu_rel_prior, condition_function

def MakeLensedParamsFromFirstEvent(img1_parameters, LensingParameters):
    """
    Function making the parameters to be injected for the second image, based on the
    parameters of the first event and the lensing parameters

    ARGS:
    -----
    - img1_parameters: dictionary containing the parameters injected for the
                                       first event
    - LensingParameters: dictionary containing the lensing paramters that GOLUM should
                                         recover (relative magnification, time delay, morse factor difference)

    RETURNS:
    --------
    - lensed_parameters: dictionary with updated parameters accounting for lensing and
                                              usable to inject the second event
    """

    lensed_parameters = img1_parameters.copy()
    lensed_parameters['luminosity_distance'] = np.sqrt(LensingParameters['relative_magnification']) * img1_parameters['luminosity_distance']
    lensed_parameters['geocent_time'] = LensingParameters['delta_t'] + img1_parameters['geocent_time']
    lensed_parameters['n_phase'] = LensingParameters['delta_n'] + img1_parameters['n_phase']

    return lensed_parameters

def read_image1_file(file):
    """
    Function reading the json file from the first run to
    extract the posteriors and the injection parameters.
    This is used instead of bilby.result.read_in_result
    to ensure backward compatibility. This can be replaced
    by the usual bily functions when the data has been
    generated in the same conditions

    ARGS:
    -----
    - file: name of the file (with full path) containing the
                results of the first run

    RETURNS:
    -------
    - injection_parameters: dictionary of the injected parameters.
                                                    is None if they are not included in the file
    - posteriors: dictionary containing the posterior samples of the
                              first run
    """

    with open(file) as f:
        result = json.load(f)

    if result['injection_parameters'] is not None:
        injection_parameters = result['injection_parameters']
    else:
        print("WARNING: no injection parameters in file, returning NONE")
        injection_parameters = None

    posteriors = dict()

    for key in result['posterior']['content']:
        posteriors[key] = result['posterior']['content'][key]

    if 'log_likelihood' in posteriors.keys():
        posteriors.pop('log_likelihood')
    if 'log_prior' in posteriors.keys():
        posteriors.pop('log_prior')

    return injection_parameters, posteriors

def ReweighingEvent1(LensedResults, LensingLikelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img):
    """
    Function doing the reweighing in practice. It can be used alone
    or is called by WeightsEvent1 where the info is also written into
    a json file.

    ARGS:
    -----
    - LensedResults: result object coming from the GOLUM run
    - LensingLikelihood: the GOLUM liklihood object used for the run
    - ifos: the interferometers used for the run
    - waveform_generator: waveform generator used for the run. It is
                                          supposed to take LensedBBHmodel in as
                                          freqency domain source model
    - n_points: int, the number of points to be used in the reweighing process
    - n_img: int, the number of the image used for the analysis (this is an
                     identifier for the run)
    - im1_posteriors, path to the first image file. If None, we reuse the samples from 
        the golum run

    RETURNS:
    --------
    Outdict with keys:
    - 'RefImgSamples%i'%n_img: the dictionary of reweighed samples
                         obtained via the reweighing process for the reference image
    - 'GolumSamples%i'%n_img: the dictionary with the reweighed GOLUM samples
    """
    idxs = np.random.randint(0, len(LensedResults.posterior['delta_t']), n_points)
    # sample the events
    samps_ev2 = {key : LensedResults.posterior[key][idxs].to_list() for key in ['delta_t', 'delta_n', 
                    'relative_magnification', 'log_likelihood']}
   

    # make the object needed to analyze the first image
    Ev1data = bilby.gw.GravitationalWaveTransient(ifos,
                                                  waveform_generator= waveform_generator)

    # draw random samples to evaluate the second likelihood
    
    if im1_posteriors is not None:
        results_1 = bilby.result.read_in_result(filename = im1_posteriors)
        samples_ev1 = results_1.posterior.copy()
        samples_ev1.pop('log_prior')
        samples_ev1.pop('log_likelihood')
    else:
        samples_ev1 = LensingLikelihood.samples.copy()

    idx2 = np.random.randint(0, len(samples_ev1['geocent_time']), n_points)


    LogL_Pd1 = np.zeros(n_points)
    for i in range(len(idx2)):
        if i%1000 == 0:
            print("Reweighing sample %i / %i"%(i, len(idx2)))
        ind1 = idxs[i]
        ind2 = idx2[i]
        for key in samples_ev1.keys():
            Ev1data.parameters[key] = samples_ev1[key][ind2]
        Ev1data.parameters['n_phase'] = samps_ev2['delta_n'][ind1] + Ev1data.parameters['n_phase']
        Ev1data.parameters['luminosity_distance'] = np.sqrt(samps_ev2['relative_magnification'][ind1])*\
                                                    Ev1data.parameters['luminosity_distance']
        Ev1data.parameters['geocent_time'] = samps_ev2['delta_t'][ind1] + Ev1data.parameters['geocent_time']
        LogL_Pd1[i] = Ev1data.log_likelihood()


    LogL_marg = np.array(samps_ev2['log_likelihood'])

    weights = np.exp((LogL_Pd1-LogL_marg) - np.max(LogL_Pd1-LogL_marg))
    
    # same the samples selected from the first event
    SampsImg1 = dict()
    for key in samples_ev1.keys():
        SampsImg1[key] = [samples_ev1[key][ind] for ind in idx2]

    # save the samples for the GOLUM part
    SampsGolum = dict()
    for key in LensedResults.posterior:
        SampsGolum[key] = [LensedResults.posterior[key][ind] for ind in idxs]

    # make the dictionary with reweighed samples
    inds = len(SampsImg1['geocent_time'])
    IdxsRew = np.random.choice(inds, size = n_points, p = weights/np.sum(weights))
    
    Img1RewSamp = dict()
    for key in SampsImg1.keys():
        Img1RewSamp[key] = list(np.array(SampsImg1[key])[IdxsRew])

    GolRewSamp = dict()
    for key in SampsGolum.keys():
        GolRewSamp[key] = list(np.array(SampsGolum[key])[IdxsRew])

    outDict = {'RefImgSamples%i'%n_img : Img1RewSamp,
               'GolumSamples%i'%n_img : GolRewSamp}

    return outDict

def ReweighingEvent1_Dl2Free(LensedResults, LensingLikelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img):
    """

    Same as other function, just for other parametrization
    Function doing the reweighing in practice. It can be used alone
    or is called by WeightsEvent1 where the info is also written into
    a json file.

    Updated version

    ARGS:
    -----
    - LensedResults: result object coming from the GOLUM run
    - LensingLikelihood: the GOLUM liklihood object used for the run
    - ifos: the interferometers used for the run
    - waveform_generator: waveform generator used for the run. It is
                                          supposed to take LensedBBHmodel in as
                                          freqency domain source model
    - n_points: int, the number of points to be used in the reweighing process
    - n_img: int, the number of the image used for the analysis (this is an
                     identifier for the run)
    - im1_posteriors, path to the first image file. If None, we reuse the samples from 
        the golum run

    RETURNS:
    --------
    - outDict: a dictionary containing the reweighed samples.
            - 'RefImgSamples%i'%n_img: the dictionary of reweighed samples
                         obtained via the reweighing process for the reference image
            - 'GolumSamples%i'%n_img: the dictionary with the reweighed 
                                      GOLUM samples                                                                                weights of the run
    """

    idxs = np.random.randint(0, len(LensedResults.posterior['geocent_time']), n_points)
    # sample the events
    samps_ev2 = dict()

    for key in  ['luminosity_distance', 'geocent_time', 'n_phase', 'log_likelihood']:
        samps_ev2[key] = [LensedResults.posterior[key][i] for i in idxs]
   

    # make the object needed to analyze the first image
    Ev1data = bilby.gw.GravitationalWaveTransient(ifos,
                                                  waveform_generator= waveform_generator)

    # draw random samples to evaluate the second likelihood
    
    if im1_posteriors is not None:
        results_1 = bilby.result.read_in_result(filename = im1_posteriors)
        samples_ev1 = results_1.posterior.copy()
        samples_ev1.pop('log_prior')
        samples_ev1.pop('log_likelihood')
    else:
        samples_ev1 = LensingLikelihood.samples.copy()

    idx2 = np.random.randint(0, len(samples_ev1['geocent_time']), n_points)


    LogL_Pd1 = np.zeros(n_points)
    for i in range(len(idx2)):
        if i%1000 == 0:
            print("Reweighing sample %i / %i"%(i, len(idx2)))
        ind1 = idxs[i]
        ind2 = idx2[i]
        for key in samples_ev1.keys():
            Ev1data.parameters[key] = samples_ev1[key][ind2]
        Ev1data.parameters['n_phase'] = samps_ev2['n_phase'][ind1]
        Ev1data.parameters['luminosity_distance'] = samps_ev2['luminosity_distance'][ind1]
        Ev1data.parameters['geocent_time'] = samps_ev2['geocent_time'][ind1]
        LogL_Pd1[i] = Ev1data.log_likelihood()


    LogL_marg = np.array(samps_ev2['log_likelihood'])

    weights = np.exp((LogL_Pd1-LogL_marg) - np.max(LogL_Pd1-LogL_marg))
    Weights = weights.tolist()
    LogLMarg = LogL_marg.tolist()
    LogL_Pd1 = LogL_Pd1.tolist()

    # same the samples selected from the first event
    SampsImg1 = dict()
    for key in samples_ev1.keys():
        SampsImg1[key] = [samples_ev1[key][ind] for ind in idx2]

    # save the samples for the GOLUM part
    SampsGolum = dict()
    for key in LensedResults.posterior:
        SampsGolum[key] = [LensedResults.posterior[key][ind] for ind in idxs]

    # make the dictionary with reweighed samples
    inds = len(SampsImg1['geocent_time'])
    IdxsRew = np.random.choice(inds, size = n_points, p = weights/np.sum(weights))
    
    Img1RewSamp = dict()
    for key in SampsImg1.keys():
        Img1RewSamp[key] = np.array(SampsImg1[key])[IdxsRew]

    GolRewSamp = dict()
    for key in SampsGolum.keys():
        GolRewSamp[key] = np.array(SampsGolum[key])[IdxsRew]

    outDict = {'RefImgSamples%i'%n_img : Img1RewSamp.to,
               'GolumSamples%i'%n_img : GolRewSamp}

    return outDict

def read_reweighted_posterior(file, n_img = 2):
    """
    Function to read the posterior files

    ARGS:
    -----
    - file: the file in which the posteriors are stored
    - n_img: the image for which the posteriors should be read

    RETURNS:
    --------
    - RefImgSamples: the reweighted samples for the
                     reference image
    - GolumSamples: the reweighed golum samples for the
                    run under consideration 
    """

    if os.path.isfile(file):
        with open(file) as f:
            result = json.load(f)

        if ('RefImgSamples%i'%(n_img)) in result.keys():
            # correct image and file are there
            RefImgSamples = result['RefImgSamples%i'%n_img]
            GolumSamples = result['GolumSamples%i'%n_img]

            return RefImgSamples, GolumSamples

        else:
            raise NameError("The imge %i does not exist in file"%n_img)

    else:
        raise NameError('File %s not found'%file)


def WeightsEvent1(LensedResults, LensingLikelihood, ifos, waveform_generator, outdir, label, im1_posteriors = None, n_points = int(1e5), dl2_free = False, n_img = 2):
    """
    Function reweighing the posteriors and saving the information into
    the reweighing file. It automatically adds the information in the
    existing file (if several images) or makes the file if it is not existing yet.

    If the reweighing has already been done with the same outdir, label and image
    numer, these values are given back.

    ARGS:
    -----
    - LensedResults: result object coming from the GOLUM run
    - LensingLikelihood: the GOLUM liklihood object used for the run
    - ifos: the interferometers used for the run
    - waveform_generator: waveform generator used for the run. It is
                                          supposed to take LensedBBHmodel in as
                                          freqency domain source model
    - outdir: the outdirectory of the run
    - label: label used for the run
    - n_points: int, the number of points to be used in the reweighing process
    - dl2_free: bool. If true, means the run was done with dl2, tc2 and mu as 
    - n_img: int, the number of the image used for the analysis (this is an
                     identifier for the run)

    RETURNS:
    --------
    - RefImgSamples: The reweighed samples for the reference image
    - GolumSamples: the reweighed samples for the GOLUM run 
    """

    # first check whether the reweighing has already been done
    if os.path.isfile('%s/%s_reweight.json' %(outdir, label)):
        with open('%s/%s_reweight.json' %(outdir, label))as f:
            result = json.load(f)

        if 'RefImgSamples%i'%n_img in result.keys():
            print("The run has already been done, LOADING the samples")
            RefImgSamples = result['RefImgSamples%i'%n_img]
            GolumSamples = result['GolumSamples%i'%n_img]

            return RefImgSamples, GolumSamples

        else:
            print("File already exists but no samples for image %i, computing samples for this image"%n_img)
            if dl2_free:
                outDict = ReweighingEvent1_Dl2Free(LensedResults, LensingLikelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)
            else:
                outDict = ReweighingEvent1(LensedResults, LensingLikelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)

            # add the new reweighed samples to the file
            with open('%s/%s_reweight.json'%(outdir, label), 'r+') as f:
                out = json.load(f)
                out.update(outDict)
                f.seek(0)
                json.dump(out, f, indent = 4)
            f.close()

            return outDict['RefImgSamples%i'%n_img], outDict['GolumSamples%i'%n_img]

    else:
        print("File %s/%s_reweight.json does not exist"%(outdir, label))
        if dl2_free:
            outDict = ReweighingEvent1_Dl2Free(LensedResults, LensingLikelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)
        else:
            outDict = ReweighingEvent1(LensedResults, LensingLikelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)

        # need to make the file
        with open('%s/%s_reweight.json' %(outdir, label), 'w') as f:
            json.dump(outDict, f, indent = 4)

        return outDict['RefImgSamples%i'%n_img], outDict['GolumSamples%i'%n_img]

def read_in_reweighted(file, n_img = 2):
    """
    Utility function to read in the reweighed posteriors
    from a file.

    ARGS:
    -----
    - file: the file that should be read
    - n_img: (default is 2): the image for which the posteriors 
                             are wanted
    """

    with open(file) as f:
        out = json.load(f)

    return out['RefImgSamples%i'%n_img], out['GolumSamples%i'%i]
