## This file contains user provided data/functions needed for numerically exact quantum dynamics performed in evolve_QD.py

import numpy as np

## DVR grid ('position' basis)
ndvr=501
mass=2000
hbar=1.0

## Potential parameters
AA=0.01
BB=0.6
CC=0.001
DD=1.0
E0=0.05

## Evolution parameters
dt=50.0
total_time=4000.0
nprint=2            ## Prints rho after every nprint steps
nsave=20            ## Prints snapshots of the wavefunction every nsave steps

## Position grid
xgrid=np.linspace(-10,25,ndvr)

## User defined potential as a function of coordinate x
def pot(x):
    V=np.zeros((2,2))
    V[0,0]=AA*np.tanh(BB*x)
    V[1,1]=-V[0,0]
    V[0,1]=CC*np.exp(-DD*x*x)
    V[1,0]=V[0,1]
    return(V)

## User defined initial wavefunction
def init_psi(n,xgrid):
    psi=np.zeros(2*n,dtype=np.complex)
    ## Initial momentum value
    k=np.sqrt(2*mass*0.03)
    sigma=1.0
    ## psi(t=0)=N exp((x-x0)**2/sigma**2) . e(ikx)
    for i in range(n):
        psi[i]=np.exp(-(xgrid[i]+5)**2/(sigma**2)) * np.exp(1.j*k*xgrid[i])
    ## Normalizing the wavefunction
    psi=psi/np.sqrt(np.vdot(psi,psi))
    return(psi)
