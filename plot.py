import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab

def draw_figure():
    plot_style = {
            'theory': ['--', 'orange', 'Theoretical-Explore'],
            'auto': ['--', 'red', 'Auto-Tuning'],
            'op': [':', 'blue', 'OP'],
        }
    plot_prior = {
            'theory': 2,
            'op': 3,
            'auto': 4,
        }
    root = 'results/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
        
    cat = os.listdir(root)
    paths = []
    for c in cat:
        if 'movielens' not in c and 'netflix' not in c and 'simulation' not in c: continue
        folders = os.listdir(root+c)
        for folder in folders:
            paths.append(root + c + '/' + folder + '/')
    for path in paths:
        algo = path.split('/')[-2]
        fn = path.split('/')[-3]
        if 'simulation' in fn:
            title = 'Simulation, d={}, K={}'.format(20, 10)
        elif 'movie' in fn or 'netflix' in fn:
            title = 'Movielens, d=20, K=100'
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 18, 'axes.labelsize': 18, 'font.size': 12, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8)}
        pylab.rcParams.update(params)
        leg = []
        keys = os.listdir(path)
        if 'grid_all' in keys:
            keys.remove('grid_all')
        keys = sorted(keys, key=lambda kv: plot_prior[kv])
        y_label = 'Cumulative Regret'
        for key in keys:
            if key not in plot_style.keys(): continue
            if algo == 'linucb': prefix = 'LinUCB-'
            elif algo == 'lints': prefix = 'LinTS-'
            elif algo == 'glmucb': prefix = 'GLMUCB-'
            leg += [prefix + plot_style[key][-1]]
            data = np.loadtxt(path+key)
            T = len(data)
            plot.plot((list(range(T))), data, linestyle = plot_style[key][0], color = plot_style[key][1], linewidth = 2)
        plot.legend((leg), loc='upper left', fontsize=8, frameon=False)
        plot.xlabel('Iterations')
        plot.ylabel(y_label)
        plot.title(title)
        fig.savefig('plots/{}_{}.pdf'.format(algo, fn), dpi=300, bbox_inches = "tight")
        print('file in path {} plotted and saved as {}_{}.pdf'.format(path, algo, fn))

draw_figure()