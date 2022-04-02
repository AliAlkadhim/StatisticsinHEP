# standard system modules
import os, sys

# standard table manipulation module
#import pandas as pd

# standard array manipulation module
import numpy as np

# standard scientific python module
import scipy as sp
import scipy.stats as st
import scipy.optimize as op

# Minuit 
import iminuit as im

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


# read all records
records = [x.strip() for x in open('likelihood_data.txt').readlines()]

# get model formula
model   = records[0].split(': ')[-1]

# get true parameters (a, b, and p)
A, B, P = [float(x.split()[-1]) for x in records[1:4]]

# read data
X       = np.array([float(x) for x in records[4:]])

model, A, B, P, X


record = '''
def F(x, a, b):
    return get_ipython().run_line_magic("s", "")
''' % model
print(record)

exec(record)


def plot_data(d, f, 
              nbins=20, 
              xmin=0, xmax=40, 
              ymin=0, ymax=0.4, 
              ftsize=20, 
              color=(0,0,1), 
              fgsize=(5, 5)):
    
    h = (xmax-xmin)/nbins
    x = np.arange(xmin, xmax, h)
    
    # set size of figure
    fig = plt.figure(figsize=fgsize)

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    
    # annotate axes
    plt.xlabel(r'$x$', fontsize=ftsize)
    plt.ylabel(r'$f(x, a, b)$', fontsize=ftsize)
    
    # histogram data
    # returns y, x, o
    # y: counts
    # x: bin boundaries
    # o: objects
    
    w = np.ones(len(d))/len(d)
    plt.hist(d, 
             weights=w,
             bins=nbins, 
             color='steelblue', 
             alpha=0.3,
             range=(xmin, xmax)) 
    
    g = f(x, A, B) * h
    plt.plot(x, g, color='red')
    
    fig.tight_layout()
    plt.savefig('fig_exp_data.pdf')
    plt.show()


plot_data(X, F)


def nll(params):
    a, b = params
    f    = np.log(F(X, a, b)) 
    return -f.sum()

nll.errdef = im.Minuit.LIKELIHOOD


guess  = [2.0,  15.0] # estimates of a and b
results= im.minimize(nll, guess)

# estimate of parameters (best-fit values)
a, b = results.x

# estimate of covariance matrix
cov  = results.hess_inv

da   = np.sqrt(cov[0][0])
db   = np.sqrt(cov[1][1])
print('a: get_ipython().run_line_magic("6.1f", " +/- %4.1f' % (a, da))")
print('b: get_ipython().run_line_magic("6.1f", " +/- %4.1f' % (b, db))")

# get value of -2*log(model) at best-fit value
nll0 = 2*nll(results.x)

# get value of Gaussian approximation of likelihood about maximum
mvn0 = st.multivariate_normal.pdf(results.x, 
                                  results.x, 
                                  results.hess_inv)


def dnlmvg(params):
    a, b = params
    f  = st.multivariate_normal.pdf(params, 
                                    results.x, 
                                    results.hess_inv) / mvn0
    return -np.log(f)

def dnll(params):
    return nll(params) - nll0


AMIN = 0.1
AMAX = 15
BMIN = 0.1
BMAX = 30

def compute_lhood(f, 
                  nsteps=100, 
                  xmin=AMIN, xmax=AMAX,
                  ymin=BMIN, ymax=BMAX):
    
    # 1. first create a mesh grid
    xdelta = float(xmax-xmin)/nsteps
    ydelta = float(ymax-ymin)/nsteps
    x      = np.arange(xmin, xmax, xdelta)
    y      = np.arange(ymin, ymax, ydelta)
    x, y   = np.meshgrid(x, y)
    
    # 2. compute likelihoods at all mesh grid points
    z = [f((a, b)) for a, b in zip(x.flatten(), y.flatten())]
    z = np.array(z).reshape(x.shape)
    z = np.exp(-z)

    return (x, y, z)


ll_true   = compute_lhood(dnll)
ll_approx = compute_lhood(dnlmvg)


def plot_lhood(points,
               approx=None,
               nsteps=50, 
               xmin=AMIN, xmax=AMAX, 
               ymin=BMIN, ymax=BMAX, 
               ftsize=20, 
               fgsize=(6, 6)):

    # set size of figure
    fig = plt.figure(figsize=fgsize)

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    
    # annotate axes
    plt.xlabel(r'$a$', fontsize=ftsize)
    plt.ylabel(r'$b$', fontsize=ftsize)
    
    # plot contours
    rainbow = plt.get_cmap('rainbow')
    x, y, z = points
    c1 = plt.contour(x, y, z, cmap=rainbow)
    c1.collections[0].set_label('true likelihood')
    
    if approx get_ipython().getoutput("= None:")
        # plot contours
        earth   = plt.get_cmap('gist_earth')
        x, y, z = approx
        c2 = plt.contour(x, y, z, cmap=earth)
        c2.collections[0].set_label('Gaussian approx.')

    # display legends
    plt.legend(loc='upper right', fontsize=18)
    
    fig.tight_layout()
    plt.savefig('fig_likelihood.pdf')
    plt.show()


plot_lhood(ll_true, ll_approx)



