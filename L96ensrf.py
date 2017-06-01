"""Ensemble square-root filters for the Lorenz 96 model"""
import numpy as np
import sys
from L96 import L96
from enkf import serial_ensrf, bulk_ensrf, etkf, letkf, etkf_modens,\
                 serial_ensrf_modens, bulk_enkf, getkf, getkf_modens
np.seterr(all='raise') # raise error when overflow occurs

if len(sys.argv) < 3:
    msg="""python L96ensrf.py covlocal method smooth_len gaussian

all variables are observed, assimilation interval given by dtassim,
nens ensemble members, observation error standard deviation = oberrstdev,
observation operator is smooth_len pt boxcar running mean or gaussian.

time mean error and spread stats printed to standard output.

covlocal:  localization distance (distance at which Gaspari-Cohn polynomial goes
to zero).

method:  =0 for serial Potter method
         =1 for bulk Potter method (all obs at once)
         =2 for ETKF (no localization applied)
         =3 for LETKF (using observation localization)
         =4 for serial Potter method with localization via modulation ensemble
         =5 for ETKF with modulation ensemble
         =6 for ETKF with modulation ensemble and perturbed obs
         =7 for serial Potter method using sqrt of localized Pb ensemble
         =8 for bulk EnKF (all obs at once) with perturbed obs.
         =9 for GETKF (no localization applied)
         =10 for GETKF (with modulated ensemble)
         =11 for ETKF with modulation ensemble and stochastic subsampling
         =12 for ETKF with modulation ensemble and 'adjusted' perturbed obs
         =13 for 'DEnKF' approx to ETKF with modulated ensemble
         =14 for 'DEnKF' approx to bulk potter method.

covinflate1,covinflate2:  (optional) inflation parameters corresponding
to a and b in Hodyss and Campbell.  If not specified, a=b=1. If covinflate2
<=0, relaxation to prior spread (RTPS) inflation used with a relaxation
coefficient equal to covinflate1."""
    raise SystemExit(msg)

corrl = float(sys.argv[1])
method = int(sys.argv[2])
smooth_len = int(sys.argv[3])
use_gaussian = bool(int(sys.argv[4]))
covinflate1=1.; covinflate2=1.
if len(sys.argv) > 5:
    # if covinflate2 > 0, use Hodyss and Campbell inflation,
    # otherwise use RTPS inflation.
    covinflate1 = float(sys.argv[5])
    covinflate2 = float(sys.argv[6])

ntstart = 1000 # time steps to spin up truth run
ntimes = 11000 # ob times
nens = 8 # ensemble members
oberrstdev = 0.1; oberrvar = oberrstdev**2 # ob error
verbose = False # print error stats every time if True
# Gaussian or running average smoothing in H.
# for running average, smooth_len is half-width of boxcar.
# for gaussian, smooth_len is standard deviation.
thresh = 0.99 # threshold for modulated ensemble eigenvalue truncation.
# model parameters...
# for truth run
dt = 0.05; npts = 80
dtassim = dt # assimilation interval
diffusion_truth_max = 2.5
diffusion_truth_min = 0.5
# for forecast model (same as above for perfect model expt)
# for simplicity, assume dt and npts stay the same.
diffusion_max = diffusion_truth_max
diffusion_min = diffusion_truth_min

rstruth = np.random.RandomState(42) # fixed seed for truth run
rsens = np.random.RandomState() # varying seed for ob noise and ensemble initial conditions

# model instance for truth (nature) run
F = 8; deltaF = 1./8.; Fcorr = np.exp(-1)**(1./3.) # efolding over n timesteps, n=3
model = L96(n=npts,F=F,deltaF=deltaF,Fcorr=Fcorr,dt=dt,diff_max=diffusion_truth_max,diff_min=diffusion_truth_min,rs=rstruth)
# model instance for forecast ensemble
ensemble = L96(n=npts,F=F,deltaF=deltaF,Fcorr=Fcorr,members=nens,dt=dt,diff_max=diffusion_truth_max,diff_min=diffusion_truth_min,rs=rsens)
for nt in range(ntstart): # spinup truth run
    model.advance()

# sample obs from truth, compute climo stats for model.
xx = []; tt = []
for nt in range(ntimes):
    model.advance()
    xx.append(model.x[0]) # single member
    tt.append(float(nt)*model.dt)
xtruth = np.array(xx,np.float)
timetruth = np.array(tt,np.float)
xtruth_mean = xtruth.mean()
xprime = xtruth - xtruth_mean
xvar = np.sum(xprime**2,axis=0)/(ntimes-1)
xtruth_stdev = np.sqrt(xvar.mean())
if verbose:
    print 'climo for truth run:'
    print 'x mean =',xtruth_mean
    print 'x stdev =',xtruth_stdev
# forward operator.
# identity obs.
ndim = ensemble.n
h = np.eye(ndim)
# smoothing in forward operator
# gaussian or heaviside kernel.
if smooth_len > 0:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i-j)
            if i-j < -(ndim/2): rr = float(ndim-j+i)
            if i-j > (ndim/2): rr = float(i-ndim-j)
            r = np.fabs(rr)/smooth_len
            if use_gaussian:
                h[j,i] = np.exp(-r**2) # Gaussian
            else: # running average (heaviside kernel)
                if r <= 1:
                    h[j,i] = 1.
                else:
                    h[j,i] = 0.
        # normalize H so sum of weight is 1
        h[j,:] = h[j,:]/h[j,:].sum()
obs = np.empty(xtruth.shape, xtruth.dtype)
for nt in range(xtruth.shape[0]):
    obs[nt] = np.dot(h,xtruth[nt])
obs = obs + oberrstdev*rsens.standard_normal(size=obs.shape)

# spinup ensemble
ntot = xtruth.shape[0]
nspinup = ntstart
for n in range(ntstart):
    ensemble.advance()

nsteps = np.int(dtassim/model.dt) # time steps in assimilation interval
if verbose:
    print 'ntstart, nspinup, ntot, nsteps =',ntstart,nspinup,ntot,nsteps
if nsteps % 1  != 0:
    raise ValueError, 'assimilation interval must be an integer number of model time steps'
else:
    nsteps = int(nsteps)

def ensrf(ensemble,xmean,xprime,h,obs,oberrvar,covlocal,method=1,z=None):
    if method == 0: # ensrf with obs one at time
        return serial_ensrf(xmean,xprime,h,obs,oberrvar,covlocal,covlocal)
    elif method == 1: # ensrf with all obs at once
        return bulk_ensrf(xmean,xprime,h,obs,oberrvar,covlocal)
    elif method == 2: # etkf (no localization)
        return etkf(xmean,xprime,h,obs,oberrvar)
    elif method == 3: # letkf
        return letkf(xmean,xprime,h,obs,oberrvar,covlocal)
    elif method == 4: # serial ensrf using 'modulated' ensemble
        return serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z)
    elif method == 5: # etkf using 'modulated' ensemble
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z)
    elif method == 6: # etkf using 'modulated' ensemble w/pert obs
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z,rs=rsens,po=True)
    elif method == 7: # serial ensrf using sqrt of localized Pb
        return serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,None)
    elif method == 8: # enkf with perturbed obs all at once
        return bulk_enkf(xmean,xprime,h,obs,oberrvar,covlocal,rsens)
    elif method == 9: # getkf with no localization
        return getkf(xmean,xprime,h,obs,oberrvar)
    elif method == 10: # getkf with modulated ensemble
        return getkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z)
    elif method == 11: # etkf using 'modulated' ensemble and stochastic subsample
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z,rs=rsens,ss=True)
    elif method == 12: # etkf using 'modulated' ensemble w/ 'adjusted' pert obs
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z,rs=rsens,po=True,adjust_obnoise=True)
    elif method == 13: # 'DEnKF'approx to ETKF with modulated ensemble.
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z,denkf=True)
    elif method == 14: # 'DEnKF'approx to bulk potter method
        return bulk_ensrf(xmean,xprime,h,obs,oberrvar,covlocal,denkf=True)
    else:
        raise ValueError('illegal value for enkf method flag')

# define localization matrix.
covlocal = np.eye(ndim)
# zonally varying localization
xdep = model.diff_min + (model.diff_max-model.diff_min)*model.blend
#xdep[:] = 1.0 # set back to constant
#import matplotlib.pyplot as plt
#plt.plot(np.arange(80), xdep*corrl)
#plt.show()
#raise SystemExit
# constant localization
#xdep = np.ones(ndim)

if corrl < 2*ndim:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i-j)
            if i-j < -(ndim/2): rr = float(ndim-j+i)
            if i-j > (ndim/2): rr = float(i-ndim-j)
            r = np.fabs(rr)/(xdep[i]*corrl); taper = 0.0
            # Bohman taper (Gneiting 2002, doi:10.1006/jmva.2001.2056,
            # equation, eq 21)
            #if r < 1.:
            #    taper = (1.-r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi
            # Gaspari-Cohn polynomial (Gneiting 2002, eq 23).
            # Figure 3 of that paper compares Bohman and GC tapers.
            rr = 2.*r
            if r <= 0.5:
                taper = ((( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2+1.
            elif r > 0.5 and r < 1.:
                taper = (((( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                        + 4.0 - 2.0 / (3.0 * rr)
            covlocal[j,i]=taper
    # symmetrize the covariance localization matrix
    covlocal = 0.5*(covlocal + covlocal.transpose())

# compute square root of covlocal
if method in [4,5,6,10,11,12,13]:
    evals, eigs = np.linalg.eigh(covlocal)
    evals = np.where(evals > 1.e-10, evals, 1.e-10)
    evalsum = evals.sum(); neig = 0
    frac = 0.0
    while frac < thresh:
        frac = evals[ndim-neig-1:ndim].sum()/evalsum
        neig += 1
    #print 'neig = ',neig
    zz = (eigs*np.sqrt(evals/frac)).T
    z = zz[ndim-neig:ndim,:]
else:
    neig = 0
    z = None

# run assimilation.
fcsterr = []
fcsterr1 = []
fcstsprd = []
analerr = []
analsprd = []
diverged = False
fsprdmean = np.zeros(ndim,np.float)
fsprdobmean = np.zeros(ndim,np.float)
asprdmean = np.zeros(ndim,np.float)
ferrmean = np.zeros(ndim,np.float)
aerrmean = np.zeros(ndim,np.float)
corrmean = np.zeros(ndim,np.float)
corrhmean = np.zeros(ndim,np.float)
for nassim in range(0,ntot,nsteps):
    # assimilate obs
    xmean = ensemble.x.mean(axis=0)
    xmean_b = xmean.copy()
    xprime = ensemble.x - xmean
    # calculate background error, sprd stats.
    ferr = (xmean - xtruth[nassim])**2
    if np.isnan(ferr.mean()):
        diverged = True
        break
    fsprd = (xprime**2).sum(axis=0)/(ensemble.members-1)
    corr = (xprime.T*xprime[:,ndim/2]).sum(axis=1)/float(ensemble.members-1)
    hxprime = np.dot(xprime,h)
    fsprdob = (hxprime**2).sum(axis=0)/(ensemble.members-1)
    corrh = (xprime.T*hxprime[:,ndim/2]).sum(axis=1)/float(ensemble.members-1)
    if nassim >= nspinup:
        fsprdmean = fsprdmean + fsprd
        fsprdobmean = fsprdobmean + fsprdob
        corrmean = corrmean + corr
        corrhmean = corrhmean + corrh
        ferrmean = ferrmean + ferr
        fcsterr.append(ferr.mean()); fcstsprd.append(fsprd.mean())
        fcsterr1.append(xmean - xtruth[nassim])
    # update state estimate.
    xmean,xprime =\
    ensrf(ensemble,xmean,xprime,h,obs[nassim,:],oberrvar,covlocal,method=method,z=z)
    # calculate analysis error, sprd stats.
    aerr = (xmean - xtruth[nassim])**2
    asprd = (xprime**2).sum(axis=0)/(ensemble.members-1)
    if nassim >= nspinup:
        asprdmean = asprdmean + asprd
        aerrmean = aerrmean + aerr
        analerr.append(aerr.mean()); analsprd.append(asprd.mean())
    if verbose:
        print nassim,timetruth[nassim],np.sqrt(ferr.mean()),np.sqrt(fsprd.mean()),np.sqrt(aerr.mean()),np.sqrt(asprd.mean())
    if covinflate2 > 0:
        # Hodyss and Campbell inflation.
        inc = xmean - xmean_b
        inf_fact = np.sqrt(covinflate1 + \
        (asprd/fsprd**2)*((fsprd/ensemble.members) + covinflate2*(2.*inc**2/(ensemble.members-1))))
    else:
        # relaxation to prior spread inflation
        asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
        inf_fact = 1.+covinflate1*(fsprd-asprd)/asprd
    xprime *= inf_fact
    # run forecast model.
    ensemble.x = xmean + xprime
    for n in range(nsteps):
        ensemble.advance()

# print out time mean stats.
# error and spread are normalized by observation error.
if diverged:
    print method,len(fcsterr),corrl,covinflate1,covinflate2,oberrstdev,np.nan,np.nan,np.nan,np.nan,neig
else:
    ncount = len(fcstsprd)
    fcsterr = np.array(fcsterr)
    fcsterr1 = np.array(fcsterr1)
    fcstsprd = np.array(fcstsprd)
    analerr = np.array(analerr)
    analsprd = np.array(analsprd)
    fstdev = np.sqrt(fcstsprd.mean())
    astdev = np.sqrt(analsprd.mean())
    asprdmean = asprdmean/ncount
    aerrmean = aerrmean/ncount
    fsprdmean = fsprdmean/ncount
    fsprdobmean = fsprdobmean/ncount
    corrmean = corrmean/ncount
    corrhmean = corrhmean/ncount
    fstd = np.sqrt(fsprdmean)
    fstdob = np.sqrt(fsprdobmean)
    covmean = corrmean; corrmean = corrmean/(fstd*fstd[ndim/2])
    covhmean = corrhmean; corrhmean = corrhmean/(fstd*fstdob[ndim/2])
    fcsterrcorr =\
    (fcsterr1.T*fcsterr1[:,ndim/2]).sum(axis=1)/float(fcsterr1.shape[0]-1)
    ferrstd = np.sqrt((fcsterr1**2).sum(axis=0)/float(fcsterr1.shape[0]-1))
    errcovmean = fcsterrcorr
    fcsterrcorr = fcsterrcorr/(ferrstd*ferrstd[ndim/2])
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(ndim),corrmean,color='k',label='r')
    #plt.plot(np.arange(ndim),corrhmean,color='b',label='r (x vs hx)')
    #plt.plot(np.arange(ndim),h[:,ndim/2]/h.max(),color='r',label='H')
    #plt.plot(np.arange(ndim),covlocal[:,ndim/2],'k:',label='L')
    #plt.xlim(0,ndim)
    #plt.legend()
    #plt.figure()
    #plt.plot(np.arange(ndim),covmean,color='b',label='mean ens cov')
    #plt.plot(np.arange(ndim),errcovmean,color='r',label='ens mean errcov')
    #plt.xlim(0,ndim)
    #plt.legend()
    #plt.show()
    print method,ncount,corrl,covinflate1,covinflate2,oberrstdev,np.sqrt(fcsterr.mean()),fstdev,\
          np.sqrt(analerr.mean()),astdev,neig
