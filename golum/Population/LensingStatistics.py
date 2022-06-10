"""
file containing the reweighig schemes in case one wants 
to use the Lensing statistics results (such as Rgal 
and Mgal)
"""
import numpy as np
import scipy.stats as stat
import bilby
from . import snronpairs_quad_dbl as sn 
import scipy.interpolate as interp


class ReweighWithMgalCatalogResults(object):
    """
    Class to reweigh the results from a GOLUM run 
    by using the results from the Mgal statistcs paper
    (More & More, https://arxiv.org/pdf/2111.03091.pdf). 
    This is reweights the Coherence ratio by including the 
    lensing statistics based on the O3 population and an
    SIE lens model.

    This considers only doubles
    """

    def __init__(self, ev1_lensed_file, ev2_lensed_file, 
                 ev1_unlensed_file, ev2_unlensed_file, 
                 lensed_catalog_file, unlensed_catalog_file, 
                 n_samps = int(1e5)):
        """
        Initialization of the class. This leads to the making of
        the KDE estimators for the SIE model

        ARGS:
        -----
        - ev1_file: json file resulting from the first image run 
        - ev2_lensed_file: result file for the second image from 
                           the lensed run (GOLUM run)
        - ev2_unlensed_file: result file for the second image from
                             the unlensed run 
        - lensed_catalog_file: the results from the Mgal catalog for 
                               the lensed mu_rel and dt distribution 
        - unlensed_catalog_file: the results from the Mgal catalog
                                  for the unlensed mu_rel and dt 
                                  distributions
        - n_samps: the number of samples to take whenever resampling 
                   is needed
        - det: the detector appendix for which we want ot test the
               model
        """

        self.ev1_lensed_file = ev1_lensed_file
        self.ev2_lensed_file = ev2_lensed_file
        self.ev1_unlensed_file = ev1_unlensed_file
        self.ev2_unlensed_file = ev2_unlensed_file
        self.n_samps = n_samps
        self.lensed_catalog_file = lensed_catalog_file
        self.unlensed_catalog_file = unlensed_catalog_file

        day = 3600*24

        # make the kernel density estimator for the lensed case
        _, _, _, mag_lens, dt_lens, _, _ = np.loadtxt(self.lensed_catalog_file, unpack = True)
        idxs = np.arange(0, dt_lens.size, 1)
        mag_l1 = mag_lens[idxs%4==1]
        mu_rel_cat_lens = -mag_lens[idxs%4==2]/mag_l1
        dt_cat_lens = dt_lens[idxs%4==2]
        dt_cat_lens *= day
        cat_lens_val = np.vstack([mu_rel_cat_lens, dt_cat_lens])
        self.kde_mu_dt_lens = stat.gaussian_kde(cat_lens_val)

        # make the unlensed kernel density estimator
        mag_unl, dt_unl = np.loadtxt(self.unlensed_catalog_file, unpack = True)
        idxu = (dt_unl > 1e-3) & (mag_unl > 1e-3)
        mag_unl_cat = mag_unl[idxu]
        dt_unl_cat = dt_unl[idxu]*day
        cat_unl_val = np.vstack([mag_unl_cat, dt_unl_cat])
        self.kde_mu_dt_unl = stat.gaussian_kde(cat_unl_val)

        # load the result from the files
        self.ev1_lens_res = bilby.result.read_in_result(filename = self.ev1_lensed_file)
        self.ev2_lens_res = bilby.result.read_in_result(filename = self.ev2_lensed_file)
        self.ev1_unl_res = bilby.result.read_in_result(filename = self.ev1_unlensed_file)
        self.ev2_unl_res = bilby.result.read_in_result(filename = self.ev2_unlensed_file)

        self.lens_rew_samples, self.lens_rew_Z = self.lensed_hypo_reweigh()
        self.unl_rew_samples, self.unl_rew_Z = self.unlensed_hypo_reweigh()

        # now we can also make the lensing population reweighed coherence ratio
        self.lens_CLU = float(self.ev1_lens_res.log_evidence) +\
                        float(self.lens_rew_Z) -\
                        float(self.ev1_unl_res.log_evidence) -\
                        float(self.unl_rew_Z)


    def lensed_hypo_reweigh(self):
        """
        Function doing the reweighing for the lensed hypo using 
        the kernel density estimatore defined above

        RETURNS:
        --------
        - rew_par: the lensing reweighed samples 
                                     for mu_rel and dt 
                                     based on the weights computes
                                     for the lensing hypothesis
        - logZ: the reweighed evidence when accounting
                               for the lensing hypothesis
        """

        # info from golum run 
        samps_mu = self.ev2_lens_res.posterior['relative_magnification']
        samps_dt = self.ev2_lens_res.posterior['delta_t']
        p_mu_golum = self.ev2_lens_res.priors['relative_magnification']
        p_dt_golum = self.ev2_lens_res.priors['delta_t']

        # compute the golum probability 
        p_MuDt_golum = p_mu_golum.prob(samps_mu)*\
                       p_dt_golum.prob(samps_dt)

        # compute the probability from the lensed kde reconstruction
        p_MuDt_cat = self.kde_mu_dt_lens(np.vstack([samps_mu, samps_dt]))

        # compute the weights
        weights = p_MuDt_cat/p_MuDt_golum

        # use the weights to reweigh the samples
        inds = len(samps_mu)
        RewIdxs = np.random.choice(inds, size = self.n_samps,
                                   p = weights/np.sum(weights))
        rew_par = dict()
        rew_par['relative_magnification'] = [samps_mu[idx] for idx in RewIdxs]
        rew_par['delta_t'] = [samps_dt[idx] for idx in RewIdxs]

        # reweigh the log evidence
        logZ = float(self.ev2_lens_res.log_evidence) + np.log(np.mean(weights))

        return rew_par, logZ

    def unlensed_hypo_reweigh(self):
        """
        Function doing the reweighing under the unlensed
        hypothesis

        RETURNS:
        --------
        - rew_par: the reweighed mu and dt ditribution 
                   when accounting for the unlensed model
        - logZ: the evidence reweighed to acount for the 
                unlensed model
        """

        # since golum uses d_1, d_2, t_1, t_2 in the 
        # unlensed case, we need to make the mu_rel and dt
        # distribution here
        dl1_samps = self.ev1_unl_res.posterior['luminosity_distance']
        dl2_samps = self.ev2_unl_res.posterior['luminosity_distance']
        t1_samps = self.ev1_unl_res.posterior['geocent_time']
        t2_samps = self.ev2_unl_res.posterior['geocent_time']

        # take random samples to make the mu an dt distributions 
        idxs1 = np.random.choice(len(dl1_samps), size = self.n_samps)
        idxs2 = np.random.choice(len(dl2_samps), size = self.n_samps)
        dl1s = np.array(dl1_samps[idxs1])
        dl2s = np.array(dl2_samps[idxs2])
        t1s = np.array(t1_samps[idxs1])
        t2s = np.array(t2_samps[idxs2])

        mus_rel = (dl2s/dl1s)**2
        dts = t2s - t1s

        # compute the probabilities for mu and dt for the 
        # golum run 
        p_d2_unl = self.ev2_unl_res.priors['luminosity_distance']
        p_t2_unl = self.ev2_unl_res.priors['geocent_time']
        p_t1_unl = self.ev1_unl_res.priors['geocent_time']

        min_dt_prior = p_t2_unl.minimum - p_t1_unl.maximum
        max_dt_prior = p_t2_unl.maximum - p_t1_unl.minimum

        p_dt_prior = bilby.core.prior.Uniform(minimum = min_dt_prior,
                                              maximum = max_dt_prior)

        p_d2_samps = p_d2_unl.prob(dl2s)
        sqrt_mus = np.sqrt(mus_rel)
        p_mu_unlGol = (2*sqrt_mus/dl1s)*p_d2_samps
        p_dt_unlGol = p_dt_prior.prob(dts)

        p_MuDt_unlGol = p_mu_unlGol*p_dt_unlGol

        # make the probability distribution for a model 
        p_MuDt_unlCat = self.kde_mu_dt_unl(np.vstack([mus_rel, dts]))

        # compute the weights
        weights_unl = p_MuDt_unlCat/p_MuDt_unlGol

        # reweigh the mu and dt samples using the weights
        inds = len(mus_rel)
        RewIdxs = np.random.choice(inds, size = self.n_samps, p = weights_unl/np.sum(weights_unl))
        rew_par_unl = dict()
        rew_par_unl['relative_magnification'] = [mus_rel[i] for i in RewIdxs]
        rew_par_unl['delta_t'] = [dts[i] for i in RewIdxs]

        # reweigh the log evidence
        logZ_unl = np.float(self.ev2_unl_res.log_evidence) + np.log(np.mean(weights_unl))

        return rew_par_unl, logZ_unl

class ReweighWithMgalCatalogResults_FullO3(object):
    """
    Class to reweigh the results from a GOLUM run 
    by using the results from the Mgal statistcs paper
    (More & More, https://arxiv.org/pdf/2111.03091.pdf). 
    This is reweights the Coherence ratio by including the 
    lensing statistics based on the O3 population and an
    SIE lens model.

    This considers only doubles
    """

    def __init__(self, ev1_lensed_file, ev2_lensed_file, 
                 ev1_unlensed_file, ev2_unlensed_file,
                  unlensed_catalog_file, det = 'O3',
                 n_samps = int(1e5)):
        """
        Initialization of the class. This leads to the making of
        the KDE estimators for the SIE model

        ARGS:
        -----
        - ev1_file: json file resulting from the first image run 
        - ev2_lensed_file: result file for the second image from 
                           the lensed run (GOLUM run)
        - ev2_unlensed_file: result file for the second image from
                             the unlensed run 
        - lensed_catalog_file: the results from the Mgal catalog for 
                               the lensed mu_rel and dt distribution 
        - unlensed_catalog_file: the results from the Mgal catalog
                                  for the unlensed mu_rel and dt 
                                  distributions
        - n_samps: the number of samples to take whenever resampling 
                   is needed
        - det: the detector appendix for which we want ot test the
               model
        """

        self.ev1_lensed_file = ev1_lensed_file
        self.ev2_lensed_file = ev2_lensed_file
        self.ev1_unlensed_file = ev1_unlensed_file
        self.ev2_unlensed_file = ev2_unlensed_file
        self.n_samps = n_samps
        self.unlensed_catalog_file = unlensed_catalog_file

        day = 3600*24

        # make the kernel density estimator for the lensed case
        # change to combine all the values
        mag31,tdel31,mag32,tdel32,mag41,tdel41,mag42,tdel42,mag21,tdel21,mag43,tdel43,dbmg21,dbtd21= sn.getsnr_forpairs(det)
        mu_rel_cat = np.concatenate([mag31, mag32, mag41, mag42, mag21, mag43, dbmg21])
        dt_cat = np.concatenate([tdel31, tdel32, tdel41, tdel42, tdel21, tdel43, dbtd21])
        idxs = (mu_rel_cat> 5e-2) & (dt_cat>1e-3)
        mu_rel_cat_lens = mu_rel_cat[idxs]
        dt_cat_lens = dt_cat[idxs]
        dt_cat_lens *= day
        cat_lens_val = np.vstack([mu_rel_cat_lens, dt_cat_lens])
        self.kde_mu_dt_lens = stat.gaussian_kde(cat_lens_val)

        # make the unlensed kernel density estimator
        mag_unl, dt_unl = np.loadtxt(self.unlensed_catalog_file, unpack = True)
        idxu = (dt_unl > 1e-3) & (mag_unl > 1e-3)
        mag_unl_cat = mag_unl[idxu]
        dt_unl_cat = dt_unl[idxu]*day
        cat_unl_val = np.vstack([mag_unl_cat, dt_unl_cat])
        self.kde_mu_dt_unl = stat.gaussian_kde(cat_unl_val)

        # load the result from the files
        self.ev1_lens_res = bilby.result.read_in_result(filename = self.ev1_lensed_file)
        self.ev2_lens_res = bilby.result.read_in_result(filename = self.ev2_lensed_file)
        self.ev1_unl_res = bilby.result.read_in_result(filename = self.ev1_unlensed_file)
        self.ev2_unl_res = bilby.result.read_in_result(filename = self.ev2_unlensed_file)

        self.lens_rew_samples, self.lens_rew_Z = self.lensed_hypo_reweigh()
        self.unl_rew_samples, self.unl_rew_Z = self.unlensed_hypo_reweigh()

        # now we can also make the lensing population reweighed coherence ratio
        self.lens_CLU = float(self.ev1_lens_res.log_evidence) +\
                        float(self.lens_rew_Z) -\
                        float(self.ev1_unl_res.log_evidence) -\
                        float(self.unl_rew_Z)


    def lensed_hypo_reweigh(self):
        """
        Function doing the reweighing for the lensed hypo using 
        the kernel density estimatore defined above

        RETURNS:
        --------
        - rew_par: the lensing reweighed samples 
                                     for mu_rel and dt 
                                     based on the weights computes
                                     for the lensing hypothesis
        - logZ: the reweighed evidence when accounting
                               for the lensing hypothesis
        """

        # info from golum run 
        samps_mu = self.ev2_lens_res.posterior['relative_magnification']
        samps_dt = self.ev2_lens_res.posterior['delta_t']
        p_mu_golum = self.ev2_lens_res.priors['relative_magnification']
        p_dt_golum = self.ev2_lens_res.priors['delta_t']

        # compute the golum probability 
        p_MuDt_golum = p_mu_golum.prob(samps_mu)*\
                       p_dt_golum.prob(samps_dt)

        # compute the probability from the lensed kde reconstruction
        p_MuDt_cat = self.kde_mu_dt_lens(np.vstack([samps_mu, samps_dt]))

        # compute the weights
        weights = p_MuDt_cat/p_MuDt_golum

        # use the weights to reweigh the samples
        inds = len(samps_mu)
        RewIdxs = np.random.choice(inds, size = self.n_samps,
                                   p = weights/np.sum(weights))
        rew_par = dict()
        rew_par['relative_magnification'] = [samps_mu[idx] for idx in RewIdxs]
        rew_par['delta_t'] = [samps_dt[idx] for idx in RewIdxs]

        # reweigh the log evidence
        logZ = float(self.ev2_lens_res.log_evidence) + np.log(np.mean(weights))

        return rew_par, logZ

    def unlensed_hypo_reweigh(self):
        """
        Function doing the reweighing under the unlensed
        hypothesis

        RETURNS:
        --------
        - rew_par: the reweighed mu and dt ditribution 
                   when accounting for the unlensed model
        - logZ: the evidence reweighed to acount for the 
                unlensed model
        """

        # since golum uses d_1, d_2, t_1, t_2 in the 
        # unlensed case, we need to make the mu_rel and dt
        # distribution here
        dl1_samps = self.ev1_unl_res.posterior['luminosity_distance']
        dl2_samps = self.ev2_unl_res.posterior['luminosity_distance']
        t1_samps = self.ev1_unl_res.posterior['geocent_time']
        t2_samps = self.ev2_unl_res.posterior['geocent_time']

        # take random samples to make the mu an dt distributions 
        idxs1 = np.random.choice(len(dl1_samps), size = self.n_samps)
        idxs2 = np.random.choice(len(dl2_samps), size = self.n_samps)
        dl1s = np.array(dl1_samps[idxs1])
        dl2s = np.array(dl2_samps[idxs2])
        t1s = np.array(t1_samps[idxs1])
        t2s = np.array(t2_samps[idxs2])

        mus_rel = (dl2s/dl1s)**2
        dts = t2s - t1s

        # compute the probabilities for mu and dt for the 
        # golum run 
        p_d2_unl = self.ev2_unl_res.priors['luminosity_distance']
        p_t2_unl = self.ev2_unl_res.priors['geocent_time']
        p_t1_unl = self.ev1_unl_res.priors['geocent_time']

        min_dt_prior = p_t2_unl.minimum - p_t1_unl.maximum
        max_dt_prior = p_t2_unl.maximum - p_t1_unl.minimum

        p_dt_prior = bilby.core.prior.Uniform(minimum = min_dt_prior,
                                              maximum = max_dt_prior)

        p_d2_samps = p_d2_unl.prob(dl2s)
        sqrt_mus = np.sqrt(mus_rel)
        p_mu_unlGol = (2*sqrt_mus/dl1s)*p_d2_samps
        p_dt_unlGol = p_dt_prior.prob(dts)

        p_MuDt_unlGol = p_mu_unlGol*p_dt_unlGol

        # make the probability distribution for a model 
        p_MuDt_unlCat = self.kde_mu_dt_unl(np.vstack([mus_rel, dts]))

        # compute the weights
        weights_unl = p_MuDt_unlCat/p_MuDt_unlGol

        # reweigh the mu and dt samples using the weights
        inds = len(mus_rel)
        RewIdxs = np.random.choice(inds, size = self.n_samps, p = weights_unl/np.sum(weights_unl))
        rew_par_unl = dict()
        rew_par_unl['relative_magnification'] = [mus_rel[i] for i in RewIdxs]
        rew_par_unl['delta_t'] = [dts[i] for i in RewIdxs]

        # reweigh the log evidence
        logZ_unl = np.float(self.ev2_unl_res.log_evidence) + np.log(np.mean(weights_unl))

        return rew_par_unl, logZ_unl


class ReweighWithRgalCatalogStat(object):
    """
    Class reweighing the Golum results using the 
    Rgal statistics (https://arxiv.org/pdf/1807.07062.pdf). 
    This is based on an SIS model.
    """

    def __init__(self, ev1_lensed, ev2_lensed, ev1_unlensed,
                 ev2_unlensed, Rgal_file, Rz = 'Rzmax',  
                 obs_time = 3600*24*365.25, n_samps = int(1e5)):
        """
        Initialization of the class. This will automatically
        initiate the KDE and reweigh the results

        ARGS:
        -----
        - ev1_lensed: the file (with full path) containing the 
                      result for the first event under the 
                      lensed hypothesis
        - ev2_lensed: the file (with full path) containing the
                      results for the second image run under
                      the lensed hypothesis
        - ev1_unlensed: the file (with full path) containing the
                        result for the first image run under
                        the unlensed hypothesis
        - ev2_unlensed: the file (with full path) containing the 
                        results for the second image run under
                        the unlensed hypothesis
        - Rgal_file: the output file for the Rgal statististic 
                     from which one can make the interpolant
        - Rz: the Rz model that should be used. Default is Rzmax, 
                other option is Rzmin
        - N_samps: the number of samples to use to do the 
                   reweighing
        """

        self.ev1_lensed_file = ev1_lensed
        self.ev2_lensed_file = ev2_lensed
        self.ev1_unlensed_file = ev1_unlensed
        self.ev2_unlensed_file = ev2_unlensed
        self.n_samps = n_samps
        self.Rgal_file = Rgal_file
        self.obs_time = obs_time

        # make the interpolant for the KDE
        rgal_res = np.genfromtxt(self.Rgal_file, names = True)
        self.spline_interp = interp.splrep(rgal_res['t_sec'], rgal_res['Poft_%s'%Rz], s = 0)
        self.P_of_dt_from_t = lambda dt : interp.splev(dt, self.spline_interp, der = 0)

        # load the results from the files
        self.ev1_lens_res = bilby.result.read_in_result(filename = self.ev1_lensed_file)
        self.ev2_lens_res = bilby.result.read_in_result(filename = self.ev2_lensed_file)
        self.ev1_unl_res = bilby.result.read_in_result(filename = self.ev1_unlensed_file)
        self.ev2_unl_res = bilby.result.read_in_result(filename = self.ev2_unlensed_file)

        self.lens_rew_samps, self.lens_rew_Z = self.lensed_hypo_reweigh()
        self.unl_rew_samples, self.unl_rew_Z = self.unlensed_hypo_reweigh()

        # compute the reweighed Clu
        self.lens_CLU = float(self.ev1_lens_res.log_evidence) + \
                        float(self.lens_rew_Z) - \
                        float(self.ev1_unl_res.log_evidence) -\
                        float(self.unl_rew_Z)

    def lensed_hypo_reweigh(self):
        """
        Function reweighing the evidence and the samples 
        under the lensed hypothesis and assuming the 
        Rgal statistic describes the time delay

        RETURNS:
        --------
        - rew_par: the reweighed time delay parameters 
        - logZ: the reweighed natural log for the evidence 
        """

        samps_dt = self.ev2_lens_res.posterior['delta_t']
        p_dt_golum = self.ev2_lens_res.priors['delta_t']

        golum_dt_prob = np.array(p_dt_golum.prob(samps_dt))
        rgal_dt_prob = self.P_of_dt_from_t(samps_dt)
        w_lens = rgal_dt_prob/golum_dt_prob

        # reweigh the samples
        inds = len(samps_dt)
        RewIdxs = np.random.choice(inds, size = self.n_samps, 
                                   p = w_lens/np.sum(w_lens))
        rew_par = [samps_dt[idx] for idx in RewIdxs]

        # reweigh the evidence
        logZ = float(self.ev2_lens_res.log_evidence) + np.log(np.mean(w_lens))

        return rew_par, logZ

    def unlensed_hypo_reweigh(self):
        """
        Function reweihging the evidence and the samples under 
        the unlensed hypothesis
        """

        def unlensed_p_log10dt(dt, time_obs):
            """
            Function computing the probability for
            log10(dt)
            """
            p_log_t = 2*(time_obs-dt)*dt/np.power(time_obs, 2)/np.log10(np.e)
            return p_log_t
        def p_dt_unlensed(dt, time_obs):
            """
            Function computing the probability of having a 
            given dt

            ARGS:
            -----
            - dt: the difference in times of arrival (in s)
            - time_obs: the observation time under consideration
            """

            p_log10 = unlensed_p_log10dt(dt, time_obs)
    
            return p_log10*(np.log10(np.e))

        t2_samps = self.ev2_unl_res.posterior['geocent_time']
        t1_samps = self.ev1_unl_res.posterior['geocent_time']

        # make dt posterior for the unlensed case
        idxs1 = np.random.choice(len(t1_samps), size = self.n_samps)
        idxs2 = np.random.choice(len(t2_samps), size = self.n_samps)
        t1s = np.array(t1_samps[idxs1])
        t2s = np.array(t2_samps[idxs2])
        dts = t2s - t1s

        # compute the golum prior
        p_t2_unl = self.ev2_unl_res.priors['geocent_time']
        p_t1_unl = self.ev1_unl_res.priors['geocent_time']
        min_dt_prior = p_t2_unl.minimum - p_t1_unl.maximum
        max_dt_prior = p_t2_unl.maximum - p_t1_unl.minimum
        p_dt_prior = bilby.core.prior.Uniform(minimum = min_dt_prior,
                                              maximum = max_dt_prior)
        p_dt_samps = p_dt_prior.prob(dts)

        # compute the probability for rgal
        p_dt_rgal = p_dt_unlensed(dts, self.obs_time)

        # make weights
        w_unl = p_dt_rgal/p_dt_samps

        # reweihg dt posterior
        inds = len(dts)
        RewIdxs = np.random.choice(inds, size = self.n_samps, p = w_unl/np.sum(w_unl))
        rew_unls = [dts[idx] for idx in RewIdxs]

        # reweigh the evidence
        logZ = float(self.ev2_unl_res.log_evidence) + np.log(np.mean(w_unl))

        return w_unl, logZ        

