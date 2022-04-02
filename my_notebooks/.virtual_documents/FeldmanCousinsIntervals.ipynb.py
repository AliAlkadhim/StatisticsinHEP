# standard system modules
import os, sys

# standard array manipulation module
import numpy as np

# standard scientific python module
import scipy as sp
import scipy.stats as st

# standard symbolic algebra module
import sympy as sm
sm.init_printing()

# standard plotting module
import matplotlib as mp
import matplotlib.pyplot as plt

# arbitrary precision real and complex calculation
#import mpmath

# make plots appear inline
get_ipython().run_line_magic("matplotlib", " inline")


# update fonts
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18
        }
mp.rc('font', **font)
mp.rc('text', usetex=True)

# set a seed to ensure reproducibility 
# on a given machine
seed = 111
rnd  = np.random.RandomState(seed)


CL    = 0.683
MUMIN =  0
MUMAX = 25
MUSTEP= 0.01


def compute_obs_interval(mu, n1, cache, cl):
    
    # step 1: define range to scan
    n2 = n1 + int(6*np.sqrt(mu))
    
    # steps 2, 3: compute ratios and sort
    rn = []
    for n in range(n1, n2+1) :
        p = st.poisson.pmf(n, mu)
        q = st.poisson.pmf(n, n)
        rn.append((p / q, p, n))
    rn.sort()
    rn.reverse()

    # step 4: cache mu for every n in interval
    m = []
    p = 0.0
    for _, pn, n in rn:
        if n not in cache:
            cache[n] = []
        cache[n].append(mu)
        m.append(n)
        p += pn
        if p >= cl:
            break
            
    # check that counts are contiguous
    m.sort()
    m  = np.array(m)
    n1 = m.min()
    n2 = m.max()
    
    # u =  n1 + ... + n2 + n2 + ... + n1
    # u = len(m)*(n1+n2)
    u = (np.arange(n1,n2+1) + np.arange(n2,n1-1,-1)).sum()
    w = len(m) * (n1 + n2)
    if u get_ipython().getoutput("= w:")
        print('** non-contiguous interval: get_ipython().run_line_magic("s\tmu", " = %5.2f' % (m, mu))")

    return (p, n1, n2)


def compute_intervals(cl, 
                      mumin=MUMIN, mumax=MUMAX, mustep=MUSTEP):
    # mumin:  minimum mean value
    # mumax:  maximum mean value
    # mustep: step in mean value

    # number of points in mu at which to construct an interval in
    # the space of counts.
    N = int(1.5*(mumax-mumin)/mustep)

    # cache the value of mu associated with each count.
    # the lower and upper limits of the confidence interval
    # associated with a count are just the minimum and
    # maximum values of mu for a given count.
    cache = {}

    mu = [] # mu values
    p  = [] # coverage probability
    n1 = 0  # lower bound of interval in space of observations

    for i in range(N):
        x = (i+1) * mustep
        q, n1, n2 = compute_obs_interval(x, n1, cache, cl)
    
        # accumulate coverage vs. mu
        if x <= mumax:
            if i % 5 == 0:
                mu.append(x)
                p.append(q)

        if i % 500 == 0:
            print('get_ipython().run_line_magic("10.1f", " %10.3f %5d %5d' % (x, q, n1, n2))")
        
    # get confidence intervals
    intervals = []
    for n in range(mumax+1):
        lower = min(cache[n])
        upper = max(cache[n])
        intervals.append((lower, upper))
    return (intervals, mu, p)


intervals, mu, p = compute_intervals(CL)


def plot_coverage(x, y, cl,
                  xmin=MUMIN, xmax=MUMAX, 
                  ymin=0, ymax=1, 
                  ftsize=20, 
                  fgsize=(6, 5)):
   
    # set size of figure
    plt.figure(figsize=fgsize)
    
    # plot points 
    plt.plot(x, y, color='steelblue', 
             label='Feldman-Cousins intervals')
    
    percent = 'get_ipython().run_line_magic("s'", " % '%'")
    icl = int(100*cl)
    plt.plot([xmin, xmax], [cl, cl], 
             color='magenta',linestyle='--', 
             label='CL = get_ipython().run_line_magic("5.3f'", " % cl)")

    # add legends 
    plt.legend(fontsize=18)
    
    # set up x, y limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
   
    # add x and y labels
    plt.xlabel(r'$\mu$', fontsize=ftsize)
    plt.ylabel(r'Pr[$\,\mu \in (\mu_L, \, \mu_U)\,$]', 
               fontsize=ftsize)
 
    # tighten layout so that image is fully
    # contained within viewport
    plt.tight_layout()
    
    filename = "fig_poisson_FC_coverage_get_ipython().run_line_magic("2.2d.pdf"", " % icl")
    print(filename)
    plt.savefig(filename)
    plt.show()


plot_coverage(mu, p, CL)


def plot_intervals(intervals, cl, 
                   xmin=MUMIN, xmax=MUMAX, 
                   ymin=0, ymax=MUMAX, 
                   ftsize=20, 
                   fgsize=(6, 5)):

    # set size of figure
    plt.figure(figsize=fgsize)
    
    # plot points
    for n, y in enumerate(intervals[:-1]):
        x = (n, n)
        if n < 10:
            print('get_ipython().run_line_magic("5d", " %5.2f, %5.2f' % (n, y[0], y[1]))")
        plt.plot(x, y, color='steelblue', linewidth=2)
    n = len(intervals)
    x = (n, n)
    y = intervals[-1]
    plt.plot(x, y, color='steelblue', linewidth=2, 
             label='get_ipython().run_line_magic("5.3f", " CL F-C intervals' % cl)")
    
    # add legends 
    plt.legend()
    
    # set up x, y limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
   
    # add x and y labels
    plt.xlabel(r'$N$', fontsize=ftsize)
    plt.ylabel(r'$[\,\mu_L(N), \, \mu_U(N)\,]$', 
               fontsize=ftsize)
 
    # tighten layout so that image is fully
    # contained within viewport
    plt.tight_layout()
    
    icl = int(100*cl)
    filename = "fig_poisson_FC_intervals_get_ipython().run_line_magic("d.pdf"", " % icl")
    print(filename)
    plt.savefig(filename)
    plt.show()


plot_intervals(intervals, CL)



