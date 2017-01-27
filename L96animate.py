"""Lorenz 1996 model animation (with zonally varying damping)
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from L96 import L96

nspinup = 1000 # time steps to spin up truth run
nmax = 10000 # number of ob times.

np.random.seed(42) # fix random seed for reproducibility

F = 8; deltaF = 1./8.; Fcorr = np.exp(-1)**3 # efolding over n timesteps, n=3
model = L96(n=80,F=F,deltaF=deltaF,Fcorr=Fcorr,diff_max=2.5,diff_min=0.5) # model instance for truth run
for nt in range(nspinup): # spinup truth run
    model.advance()

uu = []; tt = []
N = model.n
x = np.arange(N)
fig, ax = plt.subplots()
line, = ax.plot(x, model.x.squeeze())
ax.set_xlim(0,N-1)
#ax.set_ylim(3,3)
#Init only required for blitting to give a clean slate.
def init():
    global line
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

def updatefig(n):
    global tt,uu,vspec
    model.advance()
    u = model.x.squeeze()
    line.set_ydata(u)
    print n,u.min(),u.max()
    uu.append(u); tt.append(n*model.dt)
    return line,

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, updatefig, np.arange(1,nmax+1), init_func=init,
                              interval=25, blit=True, repeat=False)
#ani.save('KS.mp4',writer=writer)
plt.show()

plt.figure()
# make contour plot of solution, plot spectrum.
ncount = len(uu)
uu = np.array(uu); tt = np.array(tt)
print tt.min(), tt.max()
print uu.shape
uup = uu - uu.mean(axis=0)
print uup.shape
cov = np.dot(uup.T,uup)/(ncount-1)
print 'cov',cov.min(), cov.max(), cov.shape
nplt = 500
print uu[:nplt].min(), uu[:nplt].max()
plt.contourf(x,tt[:nplt],uu[:nplt],np.linspace(-18,18,41),cmap=plt.cm.bwr,extend='both')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.title('time-longitude snapshot of modified L96 model')
plt.savefig('hovmuller.png')

plt.figure()
print cov.min(), cov.max()
plt.pcolormesh(x,x,cov,cmap=plt.cm.bwr,vmin=-30,vmax=30)
plt.title('climatological covariance matrix for modified L96 model')
plt.xlabel('grid index')
plt.ylabel('grid index')
plt.colorbar()
plt.savefig('covmat.png')


plt.show()
