import bilby
import numpy as np
import json
import h5py
import os

def make_bbh_parameters_from_first_image_parameters_and_lensing_parameters(image_1_parameters, lensing_parameters):
    """
    Function making the parameters to be injected for the second image, based on the
    parameters of the first event and the lensing parameters

    ARGS:
    -----
    - img1_parameters: dictionary containing the parameters injected for the
                                       first event
    - lensing_parameters: dictionary containing the lensing paramters that GOLUM should
                                         recover (relative magnification, time delay, morse factor difference)

    RETURNS:
    --------
    - lensed_parameters: dictionary with updated parameters accounting for lensing and
                                              usable to inject the second event
    """

    lensed_parameters = image_1_parameters.copy()
    lensed_parameters['luminosity_distance'] = np.sqrt(lensing_parameters['relative_magnification']) * image_1_parameters['luminosity_distance']
    lensed_parameters['geocent_time'] = lensing_parameters['delta_t'] + image_1_parameters['geocent_time']
    lensed_parameters['n_phase'] = lensing_parameters['delta_n'] + image_1_parameters['n_phase']

    return lensed_parameters

def read_image1_file(file):
    """
    Function reading the first image run result. It first tries
    to read the usual bilby way, but overwrites it when 
    it cannot be done (generally due to version differences)

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

    if os.path.isfile(file):
        try:
            results = bilby.result.read_in_result(filename = file)
            injection_parameters = results.injection_parameters.copy()
            posteriors = dict()
            for key in results.posterior.keys():
                posteriors[key] = list(results.posterior[key].copy())
            if 'log_prior' in posteriors.keys():
                posteriors.pop('log_prior')

        except:
            # we fall back here in case the bilby has
            # a version compatibility issue
            # need to find the file extension 
            extension = os.path.splitext(file)[-1][1:]
            
            if extension == 'json':
                with open(file) as f:
                    results = json.load(f)

                if results['injection_parameters'] is not None:
                    injection_parameters = results['injection_parameters']
                else:
                    print("WARNING: no injection parameters in file, returning NONE")
                    injection_parameters = None

                posteriors = dict()

                for key in results['posterior']['content']:
                    posteriors[key] = results['posterior']['content'][key]

                if 'log_prior' in posteriors.keys():
                    posteriors.pop('log_prior')
            
            elif extension == 'hdf5':
                with h5py.File(file, 'r') as ff:
                    results = bilby.core.utils.recursively_load_dict_contents_from_group(ff, '/')

                injection_parameters = results['injection_parameters']

                posteriors = results['posterior'].copy()
                for key in posteriors.keys():
                    posteriors[key] = posteriors[key].tolist()
                if 'log_prior' in posteriors.keys():
                    posteriors.pop('log_prior')

            else:
                raise ValueError(f'Extension of {file} not understood')

        return injection_parameters, posteriors
    else:
        raise ValueError(f'Image 1 file {file} does not exist')


def reweighting_event_1(lensed_results, lensing_likelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img):
    """
    Function doing the reweighing in practice. It can be used alone
    or is called by weights_event_1 where the info is also written into
    a json file.

    ARGS:
    -----
    - lensed_results: result object coming from the GOLUM run
    - lensing_likelihood: the GOLUM liklihood object used for the run
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
    idxs = np.random.randint(0, len(lensed_results.posterior['delta_t']), n_points)
    # sample the events
    samps_ev2 = {key : lensed_results.posterior[key].to_list() for key in ['delta_t', 'delta_n', 
                    'relative_magnification', 'log_likelihood']}
   

    # make the object needed to analyze the first image
    ev1_data = bilby.gw.GravitationalWaveTransient(ifos,
                                                  waveform_generator= waveform_generator)

    # draw random samples to evaluate the second likelihood
    
    if im1_posteriors is not None:
        results_1 = bilby.result.read_in_result(filename = im1_posteriors)
        samples_ev1 = results_1.posterior.copy()
        samples_ev1.pop('log_prior')
        samples_ev1.pop('log_likelihood')
    else:
        samples_ev1 = lensing_likelihood.samples.copy()

    idx2 = np.random.randint(0, len(samples_ev1['geocent_time']), n_points)


    log_likelihood_pd1 = np.zeros(n_points)
    for i in range(len(idx2)):
        if i%1000 == 0:
            print("Reweighing sample %i / %i"%(i, len(idx2)))
        ind1 = idxs[i]
        ind2 = idx2[i]
        for key in samples_ev1.keys():
            ev1_data.parameters[key] = samples_ev1[key][ind2]
        
        ev1_data.parameters['n_phase'] = samps_ev2['delta_n'][ind1] + ev1_data.parameters['n_phase']
        ev1_data.parameters['luminosity_distance'] = np.sqrt(samps_ev2['relative_magnification'][ind1])*\
                                                    ev1_data.parameters['luminosity_distance']
        ev1_data.parameters['geocent_time'] = samps_ev2['delta_t'][ind1] + ev1_data.parameters['geocent_time']
        log_likelihood_pd1[i] = ev1_data.log_likelihood()


    log_likelihood_marg = np.array(samps_ev2['log_likelihood'])[idxs]

    weights = np.exp((log_likelihood_pd1-log_likelihood_marg) - np.max(log_likelihood_pd1-log_likelihood_marg))
    
    # same the samples selected from the first event
    samples_img1 = dict()
    for key in samples_ev1.keys():
        samples_img1[key] = [samples_ev1[key][ind] for ind in idx2]

    # save the samples for the GOLUM part
    samples_golum = dict()
    for key in lensed_results.posterior:
        samples_golum[key] = [lensed_results.posterior[key][ind] for ind in idxs]

    # make the dictionary with reweighed samples
    inds = len(samples_img1['geocent_time'])
    idxs_reweighted = np.random.choice(inds, size = n_points, p = weights/np.sum(weights))
    
    img1_reweighted_samples = dict()
    for key in samples_img1.keys():
        img1_reweighted_samples[key] = list(np.array(samples_img1[key])[idxs_reweighted])

    golum_reweighted_samples = dict()
    for key in samples_golum.keys():
        golum_reweighted_samples[key] = list(np.array(samples_golum[key])[idxs_reweighted])

    outDict = {'RefImgSamples%i'%n_img : img1_reweighted_samples,
               'GolumSamples%i'%n_img : golum_reweighted_samples}

    return outDict

def reweighting_event_1_effective_parameters(lensed_results, lensing_likelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img):
    """

    Same as other function, just for other parametrization
    Function doing the reweighing in practice. It can be used alone
    or is called by weights_event_1 where the info is also written into
    a json file.

    Updated version

    ARGS:
    -----
    - lensed_results: result object coming from the GOLUM run
    - lensing_likelihood: the GOLUM liklihood object used for the run
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

    idxs = np.random.randint(0, len(lensed_results.posterior['geocent_time']), n_points)
    # sample the events
    samps_ev2 = {key : lensed_results.posterior[key].to_list() for key in 
                 ['luminosity_distance', 'geocent_time', 'n_phase', 'log_likelihood']}

    # make the object needed to analyze the first image
    ev1_data = bilby.gw.GravitationalWaveTransient(ifos,
                                                  waveform_generator= waveform_generator)

    # draw random samples to evaluate the second likelihood
    
    if im1_posteriors is not None:
        results_1 = bilby.result.read_in_result(filename = im1_posteriors)
        samples_ev1 = results_1.posterior.copy()
        samples_ev1.pop('log_prior')
        samples_ev1.pop('log_likelihood')
    else:
        samples_ev1 = lensing_likelihood.samples.copy()

    idx2 = np.random.randint(0, len(samples_ev1['geocent_time']), n_points)


    log_likelihood_pd1 = np.zeros(n_points)
    for i in range(len(idx2)):
        if i%1000 == 0:
            print("Reweighing sample %i / %i"%(i, len(idx2)))
        ind1 = idxs[i]
        ind2 = idx2[i]
        for key in samples_ev1.keys():
            ev1_data.parameters[key] = samples_ev1[key][ind2]
        ev1_data.parameters['n_phase'] = samps_ev2['n_phase'][ind1]
        ev1_data.parameters['luminosity_distance'] = samps_ev2['luminosity_distance'][ind1]
        ev1_data.parameters['geocent_time'] = samps_ev2['geocent_time'][ind1]
        log_likelihood_pd1[i] = ev1_data.log_likelihood()


    log_likelihood_marg = np.array(samps_ev2['log_likelihood'])[idxs]

    weights = np.exp((log_likelihood_pd1-log_likelihood_marg) - np.max(log_likelihood_pd1-log_likelihood_marg))
    log_likelihood_pd1 = log_likelihood_pd1.tolist()

    # same the samples selected from the first event
    samples_img1 = dict()
    for key in samples_ev1.keys():
        samples_img1[key] = [samples_ev1[key][ind] for ind in idx2]

    # save the samples for the GOLUM part
    samples_golum = dict()
    for key in lensed_results.posterior:
        samples_golum[key] = [lensed_results.posterior[key][ind] for ind in idxs]

    # make the dictionary with reweighed samples
    inds = len(samples_img1['geocent_time'])
    idxs_reweighted = np.random.choice(inds, size = n_points, p = weights/np.sum(weights))
    
    img1_reweighted_samples = dict()
    for key in samples_img1.keys():
        img1_reweighted_samples[key] = list(np.array(samples_img1[key])[idxs_reweighted])

    golum_reweighted_samples = dict()
    for key in samples_golum.keys():
        golum_reweighted_samples[key] = list(np.array(samples_golum[key])[idxs_reweighted])

    outDict = {'RefImgSamples%i'%n_img : img1_reweighted_samples,
        'GolumSamples%i'%n_img : golum_reweighted_samples} # Note: There was something about a list here in the earlier version which did not run; should these be a list?

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


def weights_event_1(lensed_results, lensing_likelihood, ifos, waveform_generator, outdir, label, im1_posteriors = None, n_points = int(1e5), dl2_free = False, n_img = 2):
    """
    Function reweighing the posteriors and saving the information into
    the reweighing file. It automatically adds the information in the
    existing file (if several images) or makes the file if it is not existing yet.

    If the reweighing has already been done with the same outdir, label and image
    numer, these values are given back.

    ARGS:
    -----
    - lensed_results: result object coming from the GOLUM run
    - lensing_likelihood: the GOLUM liklihood object used for the run
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
                outDict = reweighting_event_1_effective_parameters(lensed_results, lensing_likelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)
            else:
                outDict = reweighting_event_1(lensed_results, lensing_likelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)

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
            outDict = reweighting_event_1_effective_parameters(lensed_results, lensing_likelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)
        else:
            outDict = reweighting_event_1(lensed_results, lensing_likelihood, ifos, waveform_generator, n_points, im1_posteriors, n_img)

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

    return out['RefImgSamples%i'%n_img], out['GolumSamples%i'%n_img]
