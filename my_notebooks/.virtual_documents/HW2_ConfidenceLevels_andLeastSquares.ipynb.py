N=4000
import numpy as np
import scipy as sp; import matplotlib.pyplot as plt
import scipy.optimize as op
import ROOT


x = sp.random.normal(0,1 , size=N)
y = sp.random.normal(0, 1, size=N)
z = [x*y for x,y in zip(x,y)]
plt.hist(np.array(z), label='$z=xy$')
plt.hist(x, label='$x \~ \mathcal{N}(0, 1)$'); plt.hist(y,label='$y \~ \mathcal{N}(0, 1)$')
plt.legend()


def pdf_analytical(t, N=1000, zmin=0.0, zmax=1.0):
    def integrand(t,z):
        tt = t*t
        zz = (z/(1 - z))**2
        return np.exp(-(zz + tt/zz)/2)/(z*(1 - z))
    #now return the integrated pdf (integtate)
    dz = (zmax-zmin)/N
    a = dz / np.pi
    return a * sum([integrand(t, zmin + (i+0.5)*dz) for i in range(N)])
print('get_ipython().run_line_magic("10.9f'", " % pdf_analytical(1))")
    


nbins=600 
xmin=-6; xmax=6
ymin= 0; ymax=2.0
ftsize=24 
color=(0.1,0.3,0.8) 
fgsize=(8,5)
plt.figure(figsize=fgsize)


y, x, _ = plt.hist(z, 
                   bins=nbins, 
                   color=color,
                   alpha=0.20,
                   range=(xmin, xmax), 
                   density=True, label='simulation by sp.random.normal' )
print('x are the bin edges, x=\n', x)


# convert bin edges to centers of bins
m = (x[:-1]+x[1:])/2.0
print(m)


for i in m[:5]:
    print('pdf analytical at this bin is', pdf_analytical(i), '\n')


#even though the bin edges where from the simulation, we can still use them to evaluate the pdf at those points
r = np.array([pdf_analytical(i) for i in m])
y, x, _ = plt.hist(z, 
                   bins=nbins, 
                   color=color,
                   alpha=0.20,
                   range=(xmin, xmax), 
                   density=True, label='simulation by sp.random.normal' )
plt.plot(m, r, label='analytical pdf')
plt.legend()


def DR(n, mu):
    return sp.special.gammainc(n, mu)
def DL(n, mu):
    return 1- sp.special.gammainc(n+1, mu)


def FR(n,mu):
    return DR(n, mu) - (1-CL)/2
def FL(n,mu):
    return DL(n,mu) - (1-CL)/2


def compute_interval(n):
    dn = 2*np.sqrt(n)
    # find muL
    amin = max(0, n - dn) # lower bound of range to search for solution
    amax = n # upper bound of range to search for solution
    muL = op.brentq(FR, amin, amax, args=(n,))
    # find muU
    amin = n
    amax = n + dn + 5
    muU = op.brentq(FL, amin, amax, args=(n,))
    return (muL, muU)


N=4
CL=0.693
muL, muU = compute_interval(N)
print('N = 4, mu in [get_ipython().run_line_magic("3.1f,", " %3.1f] @ 68%s CL' % (muL, muU, '%'))")


# standard system modules
import os, sys

# standard table manipulation module
import pandas as pd

# standard array manipulation module
import numpy as np

# standard scientific python module
import scipy as sp
import scipy.stats as st
import scipy.optimize as op

# standard symbolic algebra pakage
import sympy as sm

# standard plotting module
import matplotlib as mp
import matplotlib.pyplot as plt

# make plots appear inline
get_ipython().run_line_magic("matplotlib", " inline")

# update fonts
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20
        }
mp.rc('font', **font)
mp.rc('xtick', labelsize='x-small')
mp.rc('ytick', labelsize='x-small')

# set usetex = False if Latex is not available on your system
mp.rc('text', usetex=True)

# set a seed to ensure reproducibility 
# on a given machine
seed = 111
rnd = np.random.RandomState(seed)


ndata=
np.random.standard_normal(ndata)#this draws from gauss(mean=0, std=1)
