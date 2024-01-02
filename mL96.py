"""multi-level Lorenz 1996 model from
https://www.frontiersin.org/articles/10.3389/fams.2019.00003/full"""
import numpy as np

class mL96:

    def __init__(self,members=1,n=40,nlevs=32,dt=0.05,gamma=1,Fbot=8,Ftop=4):
        self.n = n; self.nlevs=nlevs; self.dt = dt
        self.Fbot = Fbot; self.Ftop=Ftop; self.members = members
        self.gamma = gamma
        self.forcing=(1-(np.arange(nlevs)/(nlevs-1)))*Fbot + (np.arange(nlevs)/(nlevs-1))*Ftop
        self.x = self.forcing+0.1*self.rs.standard_normal(size=(members,nlevs,n))

    def shiftx(self):
        xwrap = self.xwrap
        xwrap[:,: 2:self.n+2] = self.x; xwrap[:,:,1]=self.x[:,:,-1]
        xwrap[:,:,0]=self.x[:,:,-2]; xwrap[:,:,-1]=self.x[:,:,0]
        xm2=xwrap[:,:,0:self.n]; xm1=xwrap[:,:,1:self.n+1]; xp1=xwrap[:,:,3:self.n+3]
        xzp1[:,:,0:self.nlevs-1]=self.x[:,:,1:self.nlevs]
        xzm1[:,:,1:self.nlevs]=self.x[:,:,0:self.nlevs-1]
        xzp1[:,:,-1] = self.x[:,:,-1]; xzm1[:,:,0] = self.x[:,:,0]
        return xm2,xm1,xp1,xzm1,xzp1

    def dxdt(self):
        xm2,xm1,xp1,xzm1,xzp1 = self.shiftx()
        return (xp1-xm2)*xm1 - self.x + self.forcing +\
                self.gamma*(xzm1 - self.x) +\
                self.gamma*(xzp1 - self.x)

    def advance(self):
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
