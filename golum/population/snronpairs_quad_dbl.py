import numpy as np
from pylab import plt
import scipy.stats as st
import sys
import os

## This func is for O3 detector
## Returns magnification and time delay distributions for all quad pair combinations and doubles

SNRthresh=8

def getsnr_forpairs(det):
   directory = os.path.dirname(os.path.abspath(__file__))
   lid,imid,uSNR=np.loadtxt("%s/statistics_models/out_fincat_%s/all_lensprop.txt"%(directory, det),usecols=(0,1,7),unpack=True)
   qdid,qdmag,qdtdel,srcqid=np.loadtxt("%s/statistics_models/out_fincat_%s/quadall_imgprop.txt"%(directory, det),usecols=(0,3,4,5),unpack=True)
   dbid,dbmag,dbtdel,srcdid=np.loadtxt("%s/statistics_models/out_fincat_%s/dblall_imgprop.txt"%(directory, det),usecols=(0,3,4,5),unpack=True)
       
   magr31=[]; tdel31=[] 
   magr32=[]; tdel32=[] 
   magr41=[]; tdel41=[] 
   magr42=[]; tdel42=[] 
   magr21=[]; tdel21=[] 
   magr43=[]; tdel43=[] 
   dbmg21=[]; dbtd21=[] 

   for ii in range(lid.size):
       indx =  (qdid == int(lid[ii])) #& (srcqid==int(imid[ii])) 
       indx1 = (dbid == int(lid[ii])) #& (srcdid==int(imid[ii])) 
       if(np.sum(indx)==0):
           mgnw=dbmag[indx1]
           tdnw=dbtdel[indx1]

           try: 
               dbtdel2= tdnw[1] 
               dbmag1 = mgnw[0]         
               dbmag2 = mgnw[1]
               dbmg21.append(-dbmag2/dbmag1)
               dbtd21.append(dbtdel2) 
           except IndexError:
               print(ii, int(lid[ii]))
           
       else:

           mgnw=qdmag[indx]
           tdnw=qdtdel[indx]
        
           ## quad
           qdtdel2 = tdnw[1] 
           qdtdel3 = tdnw[2] 
           qdtdel4 = tdnw[3] 
           
           qdmag1 = mgnw[0]         
           qdmag2 = mgnw[1]  
           qdmag3 = mgnw[2]  
           qdmag4 = mgnw[3]  
          
        
           ###########
           qdmin31 = min(np.absolute(qdmag3),np.absolute(qdmag1)) 
           qdmin32 = min(np.absolute(qdmag3),np.absolute(qdmag2)) 
           qdmin41 = min(np.absolute(qdmag4),np.absolute(qdmag1)) 
           qdmin42 = min(np.absolute(qdmag4),np.absolute(qdmag2)) 
           qdmin21 = min(np.absolute(qdmag2),np.absolute(qdmag1)) 
           qdmin43 = min(np.absolute(qdmag4),np.absolute(qdmag3)) 
    
           if( np.sqrt(qdmin31)*uSNR[ii] > SNRthresh ):
              magr31.append( -qdmag3/qdmag1 )
              tdel31.append( qdtdel3 )
        
           if( np.sqrt(qdmin32)*uSNR[ii] > SNRthresh ):
              magr32.append( -qdmag3/qdmag2 )
              tdel32.append( qdtdel3-qdtdel2 )
        
           if( np.sqrt(qdmin41)*uSNR[ii] > SNRthresh ):
              magr41.append( -qdmag4/qdmag1 )
              tdel41.append( qdtdel4 )
        
        
           if( np.sqrt(qdmin42)*uSNR[ii] > SNRthresh ):
              magr42.append( -qdmag4/qdmag2 )
              tdel42.append( qdtdel4-qdtdel2 )
        
           if( np.sqrt(qdmin21)*uSNR[ii] > SNRthresh ):
              magr21.append(qdmag2/qdmag1 )
              tdel21.append( qdtdel2 )
        
           if( np.sqrt(qdmin43)*uSNR[ii] > SNRthresh ):
              magr43.append(qdmag4/qdmag3 )
              tdel43.append( qdtdel4-qdtdel3 )
          
    
   magr31=np.array([magr31]); tdel31=np.array([tdel31])
   magr32=np.array([magr32]); tdel32=np.array([tdel32])
                                                       
   magr41=np.array([magr41]); tdel41=np.array([tdel41])
   magr42=np.array([magr42]); tdel42=np.array([tdel42])
                                                       
   magr21=np.array([magr21]); tdel21=np.array([tdel21])
   magr43=np.array([magr43]); tdel43=np.array([tdel43])

   dbmg21=np.array([dbmg21]); dbtd21=np.array([dbtd21])
 
   return magr31[tdel31>1e-3],tdel31[tdel31>1e-3],\
          magr32[ (tdel32>1e-3) & (magr32>0.) ],tdel32[ (tdel32>1e-3) & (magr32>0.) ],\
          magr41[tdel41>1e-3],tdel41[tdel41>1e-3],\
          magr42[ (tdel42>1e-3) & (magr42>0.) ],tdel42[ (tdel42>1e-3) & (magr42>0.) ],\
          magr21[ (tdel21>1e-3) & (magr21>0.) ],tdel21[ (tdel21>1e-3) & (magr21>0.) ],\
          magr43[tdel43>1e-3],tdel43[tdel43>1e-3],\
          dbmg21[dbtd21>1e-3],dbtd21[dbtd21>1e-3]

