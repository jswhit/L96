import numpy as np
#from numpy.linalg import eigh # scipy.linalg.eigh broken on my mac
from scipy.linalg import eigh, cho_solve, cho_factor, svd, pinvh

def symsqrt_psd(a, inv=False):
    """symmetric square-root of a symmetric positive definite matrix"""
    evals, eigs = eigh(a)
    symsqrt =  (eigs * np.sqrt(np.maximum(evals,0))).dot(eigs.T)
    if inv:
        inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
        return symsqrt, inv
    else:
        return symsqrt

def symsqrtinv_psd(a):
    """inverse and inverse symmetric square-root of a symmetric positive
    definite matrix"""
    evals, eigs = eigh(a)
    symsqrtinv =  (eigs * (1./np.sqrt(np.maximum(evals,0)))).dot(eigs.T)
    inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
    return symsqrtinv, inv

def serial_ensrf(xmean,xprime,h,obs,oberrvar,covlocal,obcovlocal):
    """serial potter method"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        hxprime = np.dot(xprime,h[nob])
        hxmean = np.dot(h[nob],xmean)
        # state space update
        hxens = hxprime.reshape((nanals, 1))
        D = (hxens**2).sum()/(nanals-1) + oberrvar
        gainfact = np.sqrt(D)/(np.sqrt(D)+np.sqrt(oberrvar))
        pbht = (xprime.T*hxens[:,0]).sum(axis=1)/float(nanals-1)
        kfgain = covlocal[nob,:]*pbht/D
        xmean = xmean + kfgain*(ob-hxmean)
        xprime = xprime - gainfact*kfgain*hxens
    return xmean, xprime

def serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z):
    """serial potter method"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]

    # if True, use gain from modulated ensemble to
    # update perts.  if False, use gain from original ensemble.
    update_xprime = True
    if z is None:
        # set ensemble to square root of localized Pb
        Pb = covlocal*np.dot(xprime.T,xprime)/(nanals-1)
        evals, eigs = eigh(Pb)
        evals = np.where(evals > 1.e-10, evals, 1.e-10)
        nanals2 = eigs.shape[0]
        xprime2 = np.sqrt(nanals2-1)*(eigs*np.sqrt(evals)).T
    else:
        # modulation ensemble
        neig = z.shape[0]; nanals2 = neig*nanals; nanal2 = 0
        xprime2 = np.zeros((nanals2,ndim),xprime.dtype)
        for j in range(neig):
            for nanal in range(nanals):
                xprime2[nanal2,:] = xprime[nanal,:]*z[neig-j-1,:]
                # unmodulated member is j=1, scaled by z[-1]
                nanal2 += 1
        xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2

    # update xmean using full xprime2
    # update original xprime using gain from full xprime2
    for nob,ob in zip(np.arange(nobs),obs):
        # forward operator.
        hxprime = np.dot(xprime2,h[nob])
        hxprime_orig = np.dot(xprime,h[nob])
        hxmean = np.dot(h[nob],xmean)
        # state space update
        hxens = hxprime.reshape((nanals2, 1))
        hxens_orig = hxprime_orig.reshape((nanals, 1))
        D = (hxens**2).sum()/(nanals2-1) + oberrvar
        gainfact = np.sqrt(D)/(np.sqrt(D)+np.sqrt(oberrvar))
        pbht = (xprime2.T*hxens[:,0]).sum(axis=1)/float(nanals2-1)
        kfgain = pbht/D
        xmean = xmean + kfgain*(ob-hxmean)
        xprime2 = xprime2 - gainfact*kfgain*hxens
        if not update_xprime:
            D = (hxens_orig**2).sum()/(nanals-1) + oberrvar
            gainfact = np.sqrt(D)/(np.sqrt(D)+np.sqrt(oberrvar))
            pbht = (xprime.T*hxens_orig[:,0]).sum(axis=1)/float(nanals-1)
            kfgain = covlocal[nob,:]*pbht/D
        xprime  = xprime  - gainfact*kfgain*hxens_orig
    return xmean, xprime

def bulk_ensrf(xmean,xprime,h,obs,oberrvar,covlocal,denkf=False):
    """bulk potter method"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    R = oberrvar*np.eye(nobs)
    Rsqrt = np.sqrt(oberrvar)*np.eye(nobs)
    Pb = np.dot(np.transpose(xprime),xprime)/(nanals-1)
    Pb = covlocal*Pb
    D = np.dot(np.dot(h,Pb),h.T)+R
    if not denkf:
        Dsqrt,Dinv = symsqrt_psd(D,inv=True)
    else:
        Dinv = cho_solve(cho_factor(D),np.eye(nobs))
    kfgain = np.dot(np.dot(Pb,h.T),Dinv)
    if not denkf:
        tmp = Dsqrt + Rsqrt
        tmpinv = cho_solve(cho_factor(tmp),np.eye(nobs))
        gainfact = np.dot(Dsqrt,tmpinv)
        reducedgain = np.dot(kfgain, gainfact)
    else:
        reducedgain = 0.5*kfgain
    xmean = xmean + np.dot(kfgain, obs-np.dot(h,xmean))
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal])
    xprime = xprime - np.dot(reducedgain,hxprime.T).T
    return xmean, xprime

def bulk_enkf(xmean,xprime,h,obs,oberrvar,covlocal,rs):
    """bulk enkf method with perturbed obs"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    R = oberrvar*np.eye(nobs)
    Rsqrt = np.sqrt(oberrvar)*np.eye(nobs)
    Pb = np.dot(np.transpose(xprime),xprime)/(nanals-1)
    Pb = covlocal*Pb
    D = np.dot(np.dot(h,Pb),h.T)+R
    Dinv = cho_solve(cho_factor(D),np.eye(nobs))
    kfgain = np.dot(np.dot(Pb,h.T),Dinv)
    xmean = xmean + np.dot(kfgain, obs-np.dot(h,xmean))
    obnoise = np.sqrt(oberrvar)*rs.standard_normal(size=(nanals,nobs))
    obnoise_var = ((obnoise-obnoise.mean(axis=0))**2).sum(axis=0)/(nanals-1)
    obnoise = np.sqrt(oberrvar)*obnoise/np.sqrt(obnoise_var)
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal]) + obnoise[nanal]
    xprime = xprime - np.dot(kfgain, hxprime[:,:,np.newaxis]).T.squeeze()
    return xmean, xprime

def etkf(xmean,xprime,h,obs,oberrvar):
    """ETKF (use only with full rank ensemble, no localization)"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    # forward operator.
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal])
    hxmean = np.dot(h,xmean)
    Rinv = (1./oberrvar)*np.eye(nobs)
    YbRinv = np.dot(hxprime,Rinv)
    pa = (nanals-1)*np.eye(nanals)+np.dot(YbRinv,hxprime.T)
    pasqrt_inv, painv = symsqrtinv_psd(pa)
    kfgain = np.dot(xprime.T,np.dot(painv,YbRinv))
    enswts = np.sqrt(nanals-1)*pasqrt_inv
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    xprime = np.dot(enswts.T,xprime)
    return xmean, xprime

def getkf(xmean,xprime,h,obs,oberrvar):
    """GETKF (use only with full rank ensemble, no localization)"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    # forward operator.
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal])
    hxmean = np.dot(h,xmean)
    sqrtoberrvar_inv = 1./np.sqrt(oberrvar)
    YbRsqrtinv = hxprime * sqrtoberrvar_inv
    u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')
    sp = s**2+nanals-1
    painv =  (u * (1./sp)).dot(u.T)
    kfgain = np.dot(xprime.T,np.dot(painv,YbRsqrtinv*sqrtoberrvar_inv))
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    reducedgain = np.dot(xprime.T,u)*(1.-np.sqrt((nanals-1)/sp))
    # ETKF form
    # method 1
    #pasqrt_inv =  (u * (np.sqrt((nanals-1)/sp))).dot(u.T)
    #xprime = np.dot(xprime.T, pasqrt_inv).T
    # method 2
    #xprime = xprime - np.dot(reducedgain,u.T).T
    # this is equivalent to above, since u.T = np.dot((v.T/s).T,YbRsqrtinv.T)
    #xprime = xprime - np.dot(reducedgain,np.dot((v.T/s).T,YbRsqrtinv.T)).T
    # GETKF form
    reducedgain = np.dot(reducedgain,(v.T/s).T)*sqrtoberrvar_inv
    xprime = xprime - np.dot(reducedgain,hxprime.T).T
    return xmean, xprime

def getkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z):
    """GETKF with modulated ensemble"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    if z is None:
        raise ValueError('z not specified')
    # modulation ensemble
    neig = z.shape[0]; nanals2 = neig*nanals; nanal2 = 0
    xprime2 = np.zeros((nanals2,ndim),xprime.dtype)
    for j in range(neig):
        for nanal in range(nanals):
            xprime2[nanal2,:] = xprime[nanal,:]*z[neig-j-1,:]
            nanal2 += 1
    xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2
    # forward operator.
    hxprime = np.empty((nanals2, nobs), xprime2.dtype)
    hxprime_orig = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals2):
        hxprime[nanal] = np.dot(h,xprime2[nanal])
    for nanal in range(nanals):
        hxprime_orig[nanal] = np.dot(h,xprime[nanal])
    hxmean = np.dot(h,xmean)
    sqrtoberrvar_inv = 1./np.sqrt(oberrvar)
    YbRsqrtinv = hxprime * sqrtoberrvar_inv
    u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')
    sp = s**2+nanals2-1
    painv =  (u * (1./sp)).dot(u.T)
    kfgain = np.dot(xprime2.T,np.dot(painv,YbRsqrtinv*sqrtoberrvar_inv))
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    reducedgain = np.dot(xprime2.T,u)*(1.-np.sqrt((nanals2-1)/sp))
    reducedgain = np.dot(reducedgain, (v.T/s).T)*sqrtoberrvar_inv
    xprime = xprime - np.dot(reducedgain,hxprime_orig.T).T
    return xmean, xprime

def etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z,rs=None,po=False,ss=False,adjust_obnoise=False,denkf=False):
    """ETKF with modulated ensemble."""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    if z is None:
        raise ValueError('z not specified')
    # modulation ensemble
    neig = z.shape[0]; nanals2 = neig*nanals; nanal2 = 0
    xprime2 = np.zeros((nanals2,ndim),xprime.dtype)
    for j in range(neig):
        for nanal in range(nanals):
            xprime2[nanal2,:] = xprime[nanal,:]*z[neig-j-1,:]
            nanal2 += 1
    normfact = np.sqrt(float(nanals2-1)/float(nanals-1))
    xprime2 = normfact*xprime2
    #var = ((xprime**2).sum(axis=0)/(nanals-1)).mean()
    #var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    # 1st nanals members are original members multiplied by scalefact
    # (which is proportional to 1st eigenvector of cov local matrix)
    scalefact = normfact*z[-1]
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(80), z[-1])
    #plt.title('1st eigenvector of localization matrix')
    #plt.xlabel('j')
    #plt.ylabel('eigenvector amplitude')
    #plt.ylim(-1.2,0.2)
    #plt.axhline(0,color='k')
    #plt.savefig('eig1.png')
    #print np.abs(z[-1]).max()/np.abs(z[-1]).min()
    #plt.show()
    #raise SystemExit
    # forward operator.
    hxprime = np.empty((nanals2, nobs), xprime2.dtype)
    hxprime_orig = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals2):
        hxprime[nanal] = np.dot(h,xprime2[nanal])
    for nanal in range(nanals):
        hxprime_orig[nanal] = np.dot(h,xprime[nanal])
    hxmean = np.dot(h,xmean)

    YbRinv = np.dot(hxprime,(1./oberrvar)*np.eye(nobs))
    pa = (nanals2-1)*np.eye(nanals2)+np.dot(YbRinv,hxprime.T)
    pasqrt_inv, painv = symsqrtinv_psd(pa)
    kfgain = np.dot(xprime2.T,np.dot(painv,YbRinv))
    xmean = xmean + np.dot(kfgain, obs-hxmean)
    if denkf:
        xprime = xprime - np.dot(0.5*kfgain,hxprime_orig.T).T
    elif po: # use perturbed obs to update ensemble perts
        if rs is None:
            raise ValueError('must pass random state if po=True')
        # generate obnoise, make sure it has zero mean
        obnoise = rs.standard_normal(size=(nanals,nobs))
        obnoise = obnoise - obnoise.mean(axis=0)
        if adjust_obnoise:
            # remove part of obnoise that lies in suspace of hxprime
            cxy = np.dot(obnoise, hxprime_orig.T)
            cxx = np.dot(hxprime_orig,hxprime_orig.T)
            # pseudo-inverse of a symmetrix matrix (same as above)
            cxxinv = pinvh(cxx)
            # compute multivariate regression matrix, find part of obnoise that
            # is linearly related to hxprime, subtract from obnoise.
            obnoise = obnoise - np.dot(np.dot(cxy,cxxinv), hxprime_orig)
            # make sure mean is still zero
            obnoise = obnoise - obnoise.mean(axis=0)
        # rescale so obnoise has expected variance.
        #obnoise=np.sqrt(oberrvar/((obnoise**2).sum(axis=0)/(nanals-1)))*obnoise
        obnoise=np.sqrt(oberrvar/(((obnoise**2).sum(axis=0)/(nanals-1))).mean())*obnoise
        # check that cross-covariance really is zero
        #cxy = np.dot(obnoise, hxprime_orig.T)
        #print cxy.min(), cxy.max()
        #raise SystemExit
        hxprime = obnoise  + hxprime_orig
        xprime = xprime - np.dot(kfgain,hxprime.T).T
    elif ss: # use stochastic subsampling to select posterior perturbations.
        pasqrt_inv, painv = symsqrtinv_psd(pa)
        enswts = np.sqrt(nanals2-1)*pasqrt_inv
        xprime_full = np.dot(enswts.T,xprime2)
        # deterministic sub-sampling
        #xprime = xprime_full[np.random.choice(nanal2, nanals, replace=False)]
        # stochastic sub-sampling (nanals random linear combos of nanals
        # posterior perturbations)
        #print ((xprime_full**2).sum(axis=0)/(nanals2-1)).mean()
        ranwts = rs.standard_normal(size=(nanals,nanals2))/np.sqrt(nanals2-1)
        ranwts_mean = ranwts.mean(axis=1)
        # make sure weights sum to zero and stdev=1
        ranwts = ranwts - ranwts_mean[:,np.newaxis]
        ranwts_stdev = np.sqrt((ranwts**2).sum(axis=1))
        ranwts = ranwts/ranwts_stdev[:,np.newaxis]
        xprime = np.dot(ranwts,xprime_full)
        #print ((xprime**2).sum(axis=0)/(nanals-1)).mean()
        #raise SystemExit
        xprime = xprime - xprime.mean(axis=0)
    else:
        # compute reduced gain to update perts
        #D = np.dot(hxprime.T, hxprime)/(nanals2-1) + oberrvar*np.eye(nobs)
        #Dsqrt = symsqrt_psd(D) # symmetric square root of pos-def sym matrix
        #tmp = Dsqrt + np.sqrt(oberrvar)*np.eye(nobs)
        #tmpinv = cho_solve(cho_factor(tmp),np.eye(nobs))
        #gainfact = np.dot(Dsqrt,tmpinv)
        #kfgain = np.dot(kfgain, gainfact)
        #hxprime = hxprime[0:nanals]/scalefact
        #xprime = xprime - np.dot(kfgain,hxprime.T).T
        # update modulated ensemble perts with ETKF weights
        pasqrt_inv, painv = symsqrtinv_psd(pa)
        enswts = np.sqrt(nanals2-1)*pasqrt_inv
        # just use 1st nanals posterior members, rescaled.
        #xprime2 = np.dot(enswts.T,xprime2)
        #xprime = xprime2[0:nanals]/scalefact
        # this is equivalent, but a little faster
        xprime = np.dot(enswts[:,0:nanals].T,xprime2)/scalefact
        #xprime_mean = np.abs(xprime.mean(axis=0))
        #xprime = xprime-xprime_mean
        # make sure mean of posterior perts is zero
        #if xprime_mean.max() > 1.e-6:
        #    raise ValueError('nonzero perturbation mean')

    return xmean, xprime

def letkf(xmean,xprime,h,obs,oberrvar,obcovlocal):
    """LETKF (with observation localization)"""
    nanals, ndim = xprime.shape; nobs = obs.shape[-1]
    # forward operator.
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h,xprime[nanal])
    hxmean = np.dot(h,xmean)
    obcovlocal = np.where(obcovlocal < 1.e-13,1.e-13,obcovlocal)
    xprime_prior = xprime.copy(); xmean_prior = xmean.copy()
    # brute force application of ETKF for each state element - very slow!
    # assumes all state variables observed.
    ominusf = obs - np.dot(h,xmean_prior)
    for n in range(ndim):
        Rinv = np.diag(obcovlocal[n,:]/oberrvar)
        YbRinv = np.dot(hxprime,Rinv)
        pa = (nanals-1)*np.eye(nanals)+np.dot(YbRinv,hxprime.T)
        evals, eigs = eigh(pa)
        painv = np.dot(np.dot(eigs,np.diag(np.sqrt(1./evals))),eigs.T)
        kfgain = np.dot(xprime_prior[:,n].T,np.dot(np.dot(painv,painv.T),YbRinv))
        enswts = np.sqrt(nanals-1)*painv
        xmean[n] = xmean[n] + np.dot(kfgain, ominusf)
        xprime[:,n] = np.dot(enswts.T,xprime_prior[:,n])
    return xmean, xprime
