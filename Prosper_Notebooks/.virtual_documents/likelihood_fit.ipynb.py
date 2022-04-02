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


scale =  10.0
ndata = 200
data  = scale * rnd.standard_exponential(ndata)
data[:4]


def plotData(data, nbins, 
             xmin=0, xmax=50,
             ymin=0, ymax=60,
             ftsize=20, 
             color=(0,0,1), 
             fgsize=(8,5)):

    # set size of figure
    plt.figure(figsize=fgsize)
    
    # histogram data
    # returns y, x, o
    # y: counts
    # x: bin boundaries
    # o: objects (not used, hence the use of "_")
    y, x, _ = plt.hist(data, 
                       bins=nbins, 
                       color=color,
                       alpha=0.3,
                       range=(xmin, xmax), 
                       density=False)
    
    # convert bin boundaries to bin centers
    # Note: x[1:]  = x[1], x[2], ..., x[n-1]
    #       x[:-1] = x[0], x[1], ..., x[n-2]
    x = (x[:-1]+x[1:])/2
    
    # add simple "error" bars sqrt(N)
    y_err = np.sqrt(y)
    
    plt.errorbar(x, y, yerr=y_err, 
                 fmt='o', 
                 ecolor='blue', 
                 markersize=10,
                 color='steelblue')

    # add legends in the order in which plot objects
    # are created
    plt.legend(['data'])
    
    # set up x, y limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    # add x and y labels
    plt.xlabel('$x$', fontsize=24)
    plt.ylabel('counts', fontsize=24)
    
    # tighten layout so that image is fully
    # contained within viewport
    plt.tight_layout()
    plt.savefig("fig_likelihood_data.pdf")
    plt.show()
    
    # convert lists to numpy arrays
    return (np.array(x), np.array(y))


xx, yy = plotData(data, nbins=25, 
                  xmin=0, xmax=50,
                  ymin=0, ymax=60)

# bin width
h = xx[1]-xx[0]


# exclude bins with a count of zero
select = yy > 5
x = xx[select]
y = yy[select]
m = len(x)
print("number of data points with more than 5 counts: get_ipython().run_line_magic("d"", " % m)")


mu = 1
k  = 2
y0 = -2*(-mu + k*np.log(mu) - np.log(np.math.factorial(k)))
y1 = -2*st.poisson.logpmf(k, mu)
y1,y0


def func(params, *args):
    x, n, h = args
    a, b = params
    mu= a * b * np.exp(-b*x) * h
    f = -st.poisson.logpmf(n, mu) + st.poisson.logpmf(n, n)
    return 2*np.sqrt(f)


guess = [50.0, 5.0]    # estimates of a and b
lower = [0.0, 0.0]     # lower bounds of a and b
upper = [500.0, 500.0] # upper bounds of a and b


results = op.least_squares(func, guess, 
                           args=(x, y, h),
                           bounds=[lower, upper])


# best-fit parameters
a, b  = results.x

# compute approximate covariance matrix
J     = results.jac
H     = np.dot(J.T, J)   # approximate Hessian
cov   = np.linalg.inv(H)

# get approximate standard "errors"
da    = np.sqrt(cov[0][0])
db    = np.sqrt(cov[1][1])

chisq = results.cost     # minimum chisquare
NDF   = m - 2            # number of degrees of freedom
chisqNDF = chisq/NDF
print('''
a = get_ipython().run_line_magic("6.2f", " +/- %5.2f")
b = get_ipython().run_line_magic("6.2f", " +/- %5.2f")
chisq / NDF = get_ipython().run_line_magic("5.2f", " / %d = %5.2f")
''' % (a, da, b, db, chisq, NDF, chisqNDF))


pvalue = 1 - st.chi2.cdf(chisq, NDF)
print('''
p-value of fit: get_ipython().run_line_magic("6.2f", "")
''' % pvalue)


def plotResult(x, y, params, nbins=25, 
               xmin=0, xmax=50, 
               ymin=0, ymax=60, 
               ftsize=20, 
               color=(0,0,1), 
               fgsize=(8, 5)):
    a, b = params
    
    # set size of figure
    plt.figure(figsize=fgsize)
    
    # plot points with simple error bars sqrt(N)
    y_err = [np.sqrt(z) for z in y]
    plt.errorbar(x, y, yerr=y_err, fmt='o', 
                 ecolor='blue', markersize=10,
                 color='steelblue')
    
    h = x[1]-x[0]
    f = a*b*np.exp(-b*x)*h
    plt.plot(x, f, 'r-')
    
    # add legends in order in which plot objects
    # are created
    plt.legend(['fit', 'data'])
    
    # set up x, y limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
   
    # add x and y labels
    plt.xlabel('$x$', fontsize=ftsize)
    plt.ylabel('counts', fontsize=ftsize)
    
    # annotate 
    xwid = (xmax-xmin)/10
    ywid = (ymax-ymin)/10

    xpos = 3*xwid
    ypos = 9*ywid
    
    plt.text(xpos, ypos, 
             r'$g(x, \theta) = a b \exp(-b x)$', fontsize=ftsize)
    
    ypos -= ywid*2
    plt.text(xpos, ypos, 
             r'$a = get_ipython().run_line_magic("6.1f", " \pm %5.1f$' % (a, da), fontsize=ftsize)")
    
    ypos -= ywid
    plt.text(xpos, ypos, 
             r'$b = get_ipython().run_line_magic("6.2f", " \pm %5.2f$' % (b, db), fontsize=ftsize)")
    
    ypos -= ywid
    plt.text(xpos, ypos, 
             r'$\chi^2 \, /\,$ NDF $= get_ipython().run_line_magic("6.1f", " \, / \, %d$' % (chisq, NDF), ")
             fontsize=ftsize)

    ypos -= ywid
    plt.text(xpos, ypos, 
             r'$p$-value = get_ipython().run_line_magic("6.2f'", " % pvalue, ")
             fontsize=ftsize)
 
    # tighten layout so that image is fully
    # contained within viewport
    plt.tight_layout()
    plt.savefig("fig_likelihood_fit.pdf")
    plt.show()


plotResult(xx, yy, results.x, 
           nbins=25, xmin=0, xmax=50, ymin=0, ymax=60)



