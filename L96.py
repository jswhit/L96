"""Lorenz 1996 model (with zonally varying damping)
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18."""
import numpy as np

class L96:

    def __init__(self,members=1,n=40,dt=0.05,F=8,deltaF=0,Fcorr=1.,rs=None):
        self.n = n; self.dt = dt
        self.F = F; self.members = members
        self.deltaF = deltaF; self.Fcorr = Fcorr
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs
        self.x = F+0.1*self.rs.standard_normal(size=(members,n))
        self.xwrap = np.zeros((self.members,self.n+3), np.float)
        if self.deltaF == 0:
            self.forcing = self.F
        else:
            self.forcing = self.rs.gamma(self.F/self.deltaF, self.deltaF, size=(self.members,self.n))

    def shiftx(self):
        xwrap = self.xwrap
        xwrap[:,2:self.n+2] = self.x; xwrap[:,1]=self.x[:,-1]
        xwrap[:,0]=self.x[:,-2]; xwrap[:,-1]=self.x[:,0]
        xm2=xwrap[:,0:self.n]; xm1=xwrap[:,1:self.n+1]; xp1=xwrap[:,3:self.n+3]
        return xm2,xm1,xp1

    def dxdt(self):
        xm2,xm1,xp1 = self.shiftx()
        return (xp1-xm2)*xm1 - self.x + self.forcing

    def advance(self):
        # gamma distributed forcing with mean = self.F.  Approaches constant
        # value self.F as deltaF approaches zero, more variability as deltaF
        # increases. Forcing is random at every grid point, but correlated
        # in time with lag-1 correlation of self.Fcorr.  Forcing
        # is assumed constant over one time step (does not vary inside RK4).
        # standard deviation is sqrt(F*deltaF)
        if self.deltaF == 0:
            self.forcing = self.F
        else:
            # adjust deltaF to preserve expected variance.
            deltaF = self.deltaF/(1.-np.sqrt(self.Fcorr))
            forcing = self.Fcorr*self.forcing + (1.-self.Fcorr)*self.rs.gamma(self.F/deltaF, deltaF, size=(self.members,self.n))
            self.forcing = forcing
        h = self.dt; hh = 0.5*h; h6 = h/6.
        x = self.x
        dxdt1 = self.dxdt()
        self.x = x + hh*dxdt1
        dxdt2 = self.dxdt()
        self.x = x + hh*dxdt2
        dxdt = self.dxdt()
        self.x = x + h*dxdt
        dxdt2 = 2.0*(dxdt2 + dxdt)
        dxdt = self.dxdt()
        self.x = x + h6*(dxdt1 + dxdt + dxdt2)
