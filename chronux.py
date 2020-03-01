#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:50:59 2019

@author: dana_z

translation of the chronux package to python because I'm too far in at this point ): 
"""
import os
os.chdir('/home/dana_z/ssd_2TB/6OHDA')
#import mpld3
#mpld3.enable_notebook()
import numpy as np
import scipy as sci
import scipy.fftpack
from scipy import signal
import warnings
from scipy.stats.distributions import chi2


def checkIfVector(v,shape='N'):
    # this function make sure v is a vector and returns v with the desired shape
    # input: 
    #   v - potential vector
    #   shape - the final shape to be returned (options:['(1,N)','(N,1)','N'])
    if v.ndim>2:
        return False, v
    elif v.ndim==2 and not (v.shape[1] == 1 or v.shape[0] == 1):
        return False, v
    else:
        if not shape in ['(1,N)','(N,1)','N']:
            shape == 'N'
        N = np.max(v.shape)
        v.shape = eval(shape)
        return True, v


def mtfftc(data,tapers,nfft,Fs):
    # python implementation of chronux mtfftc from:
    # http://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/continuous/mtfftc.html

    # Multi-taper fourier transform - continuous data
    # Input:
    #       data (in form samples x channels/trials or a single vector)
    #       tapers (precalculated tapers from dpss)
    #       nfft (length of padded data)
    #       Fs   (sampling frequency)
    #
    # Output:
    #       J (fft in form frequency index x taper index x channels/trials)

    isV, data = checkIfVector(data,shape='(N,1)')
    if not isV:
        warnings.warn('data is a matrix, assuming each column is a seperate trial. data.shape = '+str(data.shape))

    NC,C = data.shape # size of data
    NK, K = tapers.shape # size of tapers
    if NK!=NC:
        raise ValueError('length of tapers is incompatible with length of data')
    # add channel indices to tapers
    tapers = np.expand_dims(tapers, axis=2)
    tapers = np.tile(tapers, [1,1,C])
    # add taper indices to data
    data = np.expand_dims(data, axis=2)
    data = np.tile(data, [1,1,K])
    data = np.moveaxis(data, [0,1,2], [0,2,1])


    data_proj = np.multiply(data, tapers)  # product of data with tapers
    J=sci.fftpack.fft(data_proj,int(nfft),axis=0)/Fs;   # fft of projected data
    return J

def getfgrid(Fs,nfft,fpass):
    # This function is a python implementation of getfgrid from chronux:
    # http://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/helper/getfgrid.html
    
    # Helper function that gets the frequency grid associated with a given fft based computation
    # Called by spectral estimation routines to generate the frequency axes
    # Usage: [f,findx]=getfgrid(Fs,nfft,fpass)
    # Inputs:
    # Fs        (sampling frequency associated with the data)-required
    # nfft      (number of points in fft)-required
    # fpass     (band of frequencies at which the fft is being calculated [fmin fmax] in Hz)-required
    # Outputs:
    # f         (frequencies)
    
    
     df = Fs/nfft
     f = np.arange(0,np.ceil(Fs/df))*df # all possible frequencies
     f = f[0:int(np.floor(nfft))]
     if len(fpass)!=1:
        func = lambda x: (x >= fpass[0]) and (x<=fpass[-1])
        findx = [i for (i, val) in enumerate(f) if func(val)]
        f=f[findx]
     else:
        findx =np.argmin(np.abs(f-fpass));
        f=[f[findx]]
     
     
     return f, findx

def dpsschk(tapers,N,Fs):
    # python implementation of chronux dpsschk:
    # http://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/helper/dpsschk.html
    # Helper function to calculate tapers and, if precalculated tapers are supplied,
    # to check that they (the precalculated tapers) the same length in time as
    # the time series being studied. The length of the time series is specified
    # as the second input argument N. Thus if precalculated tapers have
    # dimensions [N1 K], we require that N1=N.
    # Usage: tapers=dpsschk(tapers,N,Fs)
    # Inputs:
    #   tapers : tapers in the form of:
    #                 (i) precalculated tapers or,
    #                 (ii) [NW K] - time-bandwidth product, number of tapers)
    #
    #   N : number of samples
    #   Fs : sampling frequency - this is required for nomalization of
    #           tapers: we need tapers to be such that integral of the square 
    #           of each taper equals 1 dpss computes tapers such that the
    #           SUM of squares equals 1 - so we need to multiply the dpss 
    #           computed tapers by sqrt(Fs) to get the right normalization
    # Outputs:
    #   tapers : calculated or precalculated tapers
    #   eigs : eigenvalues
    
     
     if isinstance(tapers,list) and len(tapers)==2:
         tapers,eigs=sci.signal.windows.dpss(int(N),tapers[0],Kmax=tapers[1],return_ratios=True)
         tapers = tapers*np.sqrt(Fs)
         return tapers, eigs
     elif N!=tapers.shape[1]:
         raise ValueError('''seems to be an error in your dpss calculation; 
                          the number of time points is different from the length 
                          of the tapers''')
     return tapers, None

def mtspecgramc(data,movingwin,tapers=[2,3],pad=0,Fs=20,fpass=None,err=0,trialave=0):
    # This function is a python implementation of mtspecgramc from chronux: 
    # http://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/continuous/mtspecgramc.html#_top
    # Input: 
    #   data : the data (must be a cloumn vector)
    #   tapers : precalculated tapers from dpss or in the one of the following
    #           forms:
    #            (1) A numeric vector [TW K] where TW is the
    #                   time-bandwidth product and K is the number of
    #                   tapers to be used (less than or equal to
    #                   2TW-1).
    #            (2) A numeric vector [W T p] where W is the
    #                   bandwidth, T is the duration of the data and p
    #                   is an integer such that 2TW-p tapers are used. In
    #                   this form there is no default i.e. to specify
    #                   the bandwidth, you have to specify T and p as
    #                   well. Note that the units of W and T have to be
    #                   consistent: if W is in Hz, T must be in seconds
    #                   and vice versa. Note that these units must also
    #                   be consistent with the units of params.Fs: W can
    #                   be in Hz if and only if params.Fs is in Hz.
    #                   The default is to use form 1 with TW=3 and K=5
    #                 Note that T has to be equal to movingwin(1).
    #   pad : (padding factor for the FFT) - optional (can take values -1,0,1,2...).
    #         -1 corresponds to no padding, 0 corresponds to padding to the 
    #         next highest power of 2 etc.
    #         e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
    #              to 512 points, if pad=1, we pad to 1024 points etc.
    #   Fs : sampling frequency (default 20Hz) 
    #   fpass : frequency band to be used in the calculation in the form
    #           [fmin fmax].
    #           Default all frequencies between 0 and Fs/2
    #   err : error calculation [1 p] - Theoretical error bars; [2 p] - Jackknife error bars
    #        [0 p] or 0 - no error bars.
    #   trialave : average over trials/channels when 1, don't average when 0
    # Output:
    #   S : spectrum in form time x frequency x channels/trials if trialave=0;
    #       in the form time x frequency if trialave=1
    #   t : times
    #   f : frequencies
    #   Serr : error bars only for err(1)>=1
 
    
    
    # check that data is a vector
    isV, data = checkIfVector(data,shape='(N,1)')
    if not isV:
        warnings.warn('data is a matrix, assuming each column is a seperate trial. data.shape = '+str(data.shape))
    
    # intialize fpass
    if fpass is None: 
        fpass = [0,Fs/2]
        
    # check valid tapers:
    if len(tapers)==3 and movingwin[0]!=tapers[1]:
         raise ValueError('''Duration of data in tapers is inconsistent with
                          movingwin[0], modify tapers[1] to proceed''')    
    
    # if err==0, no error bars would be calculated, Serr irrelevant 
    if err==0 or err[0]==0:
        Serr = None
        errV = False
    else:
        errV = True
    
    #start code:
    N, Ch = data.shape
    Nwin = np.round(Fs*movingwin[0]) # number of samples in window
    Nstep = np.round(movingwin[1]*Fs) # number of samples to step through
    nfft = np.nanmax((2**(np.ceil(np.log2(np.abs(Nwin))))+pad,Nwin))
    f,_ = getfgrid(Fs,nfft,fpass) 
    Nf = len(f)
    tapers,_ =dpsschk(tapers,Nwin,Fs)
    winstart = np.arange(0,(N-Nwin+1)/Nstep)*Nstep;
    wn = len(winstart)
    if trialave:
        S = np.zeros((wn,Nf));
        if errV:
            Serr=np.zeros((2,wn,Nf))
    else:
        S = np.zeros((wn,Nf,Ch))
        if errV:
            Serr=np.zeros((2,wn,Nf,Ch))
             
    for n in range(0,wn):
        indx = np.arange(winstart[n],winstart[n]+Nwin,dtype=int)
        datawin = data[indx,:]
        if errV:
            s,f,serr= mtspectrumc(datawin,tapers,pad,Fs,fpass,err,trialave)
            if serr.ndim==2:
                serr = np.expand_dims(serr, axis=-1)
            Serr[0,n,:,:]=serr[0,:,:]
            Serr[1,n,:,:]=serr[1,:,:]
        else:
            s,f,_ =mtspectrumc(datawin,tapers,pad,Fs,fpass,err,trialave)
        if s.ndim==1:
            s = np.expand_dims(s, axis=-1)
        S[n,:,:]=s

    S=np.squeeze(S) 
    if errV:
        Serr=np.squeeze(Serr)
    winmid=winstart+np.round(Nwin/2)
    t=winmid/Fs
    
    
    return S,t,f,Serr

def mtspectrumc(data,tapers=[2,3],pad=0,Fs=20,fpass =None,err=0,trialave=0):
    # python implementation of chronux mtspectrumc from:
    # http://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/continuous/mtspectrumc.html

    # Multi-taper spectrum - continuous process
    #
    # Usage:
    #
    # [S,f,Serr]=mtspectrumc(data,params)
    # Input:
    # Note units have to be consistent. See chronux.m for more information.
    #   data : in form samples x channels/trials
    #   tapers : precalculated tapers from dpss or in the one of the following
    #            forms:
    #               (1) A numeric vector [TW K] where TW is the time-bandwidth 
    #                   product and K is the number of tapers to be used (less
    #                   than or equal to 2TW-1).
    #               (2) A numeric vector [W T p] where W is the bandwidth, T 
    #                   is the duration of the data and p is an integer such 
    #                   that 2TW-p tapers are used. In this form there is no 
    #                   default i.e. to specify the bandwidth, you have to 
    #                   specify T and p as well. Note that the units of W and 
    #                   T have to be consistent: if W is in Hz, T must be in 
    #                   seconds and vice versa. Note that these units must also
    #                   be consistent with the units of params.Fs: W can
    #                   be in Hz if and only if params.Fs is in Hz.
    #
    #   pad : padding factor for the FFT.  can take values -1,0,1,2...).
    #         -1 corresponds to no padding, 0 corresponds to padding to the 
    #         next highest power of 2 etc.
    #           e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we 
    #                pad the FFT to 512 points, if pad=1, we pad to 1024 points etc.
    #  Fs : sampling frequency
    #  fpass : frequency band to be used in the calculation in the form [fmin fmax]
    #          Default: all frequencies between 0 and Fs/2
    #  err : error calculation [1 p] - Theoretical error bars; 
    #                          [2 p] - Jackknife error bars
    #                          [0 p] or 0 - no error bars
    #  trialave : average over trials/channels when 1, don't average when 0
    # Output:
    #   S : spectrum in form frequency x channels/trials if trialave=0;
    #       in the form frequency if trialave=1)
    #   f : frequencies
    #   Serr : error bars -- only for err(1)>=1
     



    if err==0 or err[0]==0:
        Serr = None
        errV = False
    else:
        errV = True

    isV, data = checkIfVector(data,shape='(N,1)')
    if not isV:
        warnings.warn('data is a matrix, assuming each column is a seperate trial. data.shape = '+str(data.shape))

    N=data.shape[0]
    nfft = np.nanmax((2**(np.ceil(np.log2(np.abs(N))))+pad,N))

    f,findx = getfgrid(Fs,nfft,fpass) 
    tapers,_=dpsschk(tapers,N,Fs) 
    tapers = tapers.T# check tapers
    J=mtfftc(data,tapers,nfft,Fs)
    J=J[findx,:,:]
    S=np.squeeze(np.nanmean(np.abs(np.multiply(np.conj(J),J)),axis=1))
    if trialave:
        S=np.squeeze(np.nanmean(S,axis=1))
    if errV:
        Serr=specerr(S,J,err,trialave);
    
    return S, f, Serr

    



def specerr(S,J,err,trialave,numsp=None):
    # python implementation of the specerr function from chronux:
    # http://chronux.org/chronuxFiles/Documentation/chronux/spectral_analysis/helper/specerr.html
    
    # Function to compute lower and upper confidence intervals on the spectrum
    #
    # Inputs:
    # S - spectrum
    # J - tapered fourier transforms
    # err - [errtype p] (errtype=1 - asymptotic estimates; errchk=2 - Jackknife estimates;
    #                   p - p value for error estimates)
    # trialave - 0: no averaging over trials/channels
    #            1 : perform trial averaging
    # numsp    - number of spikes in each channel. specify only when finite
    #            size correction required (and of course, only for point
    #            process data)
    #
    # Outputs:
    # Serr - error estimates. Only for err(1)>=1. If err=[1 p] or [2 p] Serr(...,1) and Serr(...,2)
    # contain the lower and upper error bars with the specified method.

    if err[0]==0:
        raise ValueError('''Need err=[1 p] or [2 p] for error bar calculation. 
                         Make sure you are not asking for the output of Serr''')
    nf,K,C = J.shape
    errchk = err[0]
    p = err[1]
    pp = 1-p/2
    qq = 1-pp
     
    if trialave:
        dim = K*C
        C = 1
        dof = 2*dim
        if numsp is not None:
            dof = np.floor(np.abs(1/(1/dof + 1/(2*np.sum(numsp)))))*np.sign(1/dof + 1/(2*np.sum(numsp)))
        J=np.reshape(J,(nf,dim))
    else:
        dim = K
        dof=2*dim*np.ones((1,C))
        for ch in range(0,C):
            if numsp is not None:
                dof[ch] = np.floor(np.abs(1/(1/dof + 1/(2*numsp(ch))))) * np.sign(1/dof + 1/(2*numsp(ch))) 

    Serr = np.zeros((2,nf,C))
    if errchk==1:
        Qp = chi2.ppf(pp, df=dof)
        Qq = chi2.ppf(qq,df=dof)
        dofS = np.tile(dof,[nf,1])
        if S.ndim < dofS.ndim:
            S = np.expand_dims(S, axis=-1)
        Serr[0,:,:] = np.divide(np.multiply(dofS,S),np.tile(Qp,[nf,1]))
        Serr[1,:,:] = np.divide(np.multiply(dofS,S),np.tile(Qq,[nf,1]))
    elif errchk==2:
        tcrit = sci.stats.t.ppf(pp,dim-1)
        for k in range(0,dim):
            indices = [x for x in np.arange(0,dim) if x != k]
            Jjk = J[:,indices,:] # 1-drop projection
            eJjk = np.squeeze(np.sum(np.multiply(Jjk,np.conj(Jjk)),axis=1))
            if 'Sjk' not in locals():
                Sjk = np.empty_like(eJjk)
                Sjk = np.expand_dims(Sjk, axis=0)
                Sjk = np.tile(Sjk, [dim,1,1])
            Sjk[k,:,:] = eJjk/(dim-1) # 1-drop spectrum

        sigma = np.sqrt(dim-1)*np.squeeze(np.std(np.log(Sjk),axis=0))
        if sigma.ndim==1:
            sigma.shape = (sigma.shape[0],1) 
        conf=np.multiply(np.matlib.repmat(tcrit,nf,C),sigma)
        conf=np.squeeze(conf) 
        if conf.ndim==1:
            conf.shape = (conf.shape[0],1)
        if S.ndim==1:
            S.shape = (S.shape[0],1) 
        Serr[0,:,:]=np.multiply(S,np.exp(-conf))
        Serr[1,:,:]=np.multiply(S,np.exp(conf))

    Serr=np.squeeze(Serr)
    return Serr
