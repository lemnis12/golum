"""
Example file for how to reweight the results 
using lensing statistic models.

NOTE: Change pah to path in your file
"""
from golum.Population import LensingStatistics as LensStat
import matplotlib.pyplot as plt
import numpy as np
import bilby

ev1_file = 'event1_result.json'
ev2_lensed_file = 'Event2_result.json'
ev2_unl_file = 'UnlensedHypo_Event2_result.json'

# information to use the Mgal model
det = 'O3'
lensed_cat_file = "/home/justin/anaconda3/envs/GolumUpdate/lib/python3.9/site-packages/golum/Population/StatisticsModels/out_fincat_%s/quadall_imgprop.txt"%det
unlensed_cat_file = "/home/justin/anaconda3/envs/GolumUpdate/lib/python3.9/site-packages/golum/Population/StatisticsModels/unlensedpairs/unlensedpairs_tdmag_%s.txt"%(det)

# information to use for Rgal
rgal_file = "/home/justin/anaconda3/envs/GolumUpdate/lib/python3.9/site-packages/golum/Population/StatisticsModels/Rgal_dt.txt"

# do the reweighing fro Mgal
reweighed_Mgal = LensStat.ReweighWithMgalCatalogResults_FullO3(ev1_file, ev2_lensed_file, 
                                                               ev1_file, ev2_unl_file,
                                                               unlensed_cat_file)
# check effect on the Coherence ratio
print("The non-reweighed log Clu is %.3f" %(reweighed_Mgal.ev2_lens_res.log_evidence-reweighed_Mgal.ev2_unl_res.log_evidence))
print("The reweighed log Clu is %.3f" %(reweighed_Mgal.lens_CLU))

# also plot the different mus distribution 
plt.hist(reweighed_Mgal.ev2_lens_res.posterior['relative_magnification'], 
         bins = 50, density = True, 
         histtype = 'step', label = 'Initial mu_rel')
plt.hist(reweighed_Mgal.lens_rew_samples['relative_magnification'], 
         bins = 50, density = True, 
         histtype = 'step', label = 'Lensed Hypo reweigh')
plt.hist(reweighed_Mgal.unl_rew_samples['relative_magnification'], 
         bins = 50, density = True, 
         histtype = 'step', label = 'Unlensed Hypo reweigh')
plt.legend(loc = 'best')
plt.xlabel(r'$\mu_{rel}$')
plt.grid()

# do the same for Rgal
reweighed_Rgal = LensStat.ReweighWithRgalCatalogStat(ev1_file, ev2_lensed_file,
                                                     ev1_file, ev2_unl_file,
                                                     rgal_file)

# print the effect on the coherence ratio
print("The non reweighed log Clu is %.3f"%(reweighed_Rgal.ev2_lens_res.log_evidence-reweighed_Rgal.ev2_unl_res.log_evidence))
print("The reweighed log Clu is %.3f"%(reweighed_Rgal.lens_CLU))

plt.show()
