import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab

def draw_grid():
    root = 'results/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
        
    cat = os.listdir(root)
    paths = []
    for c in cat:
        if 'movielens' not in c and 'covtype' not in c and 'yahoo' not in c and 'simulation' not in c: continue
        folders = os.listdir(root+c)
        for folder in folders:
            paths.append(root + c + '/' + folder + '/')

    for path in paths:
        fn = path.split('/')[-3] 
        algo = path.split('/')[-2]
        if algo == 'linucb': prefix = 'LinUCB-'
        elif algo == 'lints': prefix = 'LinTS-'
        elif algo == 'glmucb': prefix = 'GLMUCB-'
        if 'simulation' in fn:
            _, dstr, Kstr = fn.split('_')
            d = int(dstr[1:])
            K = int(Kstr[1:])
            title = 'Cumulative regret v.s. Exploration rates for {}'.format(prefix[:-1])# .format(d, K)
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 18, 'axes.labelsize': 18, 'font.size': 12, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8)}
        pylab.rcParams.update(params)

        keys = os.listdir(path)
        if 'grid_all' not in keys:
            continue

        key = 'grid_all'
        data = np.loadtxt(path+key)
        regret = data[:,-1]
        T = len(regret)
        plot.plot((list(range(T))), regret, linestyle = '-', color = 'black', linewidth = 2)

        plot.xlabel('Exploration rates')
        y_label = 'Cumulative Regret'
        plot.ylabel(y_label)
        
        if 'lin' in path:
            dates = list(np.arange(0, 10.1, 1))
            dates = list(str(int(x)) for x in np.arange(0, 10.1, 1))
            # plot.xticks(list(range(T)), dates)
            tmpx = list(range(T))
            plot.xticks([0] + tmpx[9::10], dates)
        elif 'glm' in path:
            dates = list(np.arange(0, 5.1, 0.5))
            # dates = list(str(int(x)) for x in np.arange(0, 10.1, 1))
            # plot.xticks(list(range(T)), dates)
            tmpx = list(range(T))
            plot.xticks(tmpx, dates)
            
        plot.title(title)
        fig.savefig('plots/{}_{}_{}.pdf'.format(algo, fn, 'grid_all'), dpi=300, bbox_inches = "tight")
        print('file in path {} plotted and saved as {}_{}_{}.pdf'.format(path, algo, fn, 'grid_all'))

draw_grid()