import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab

def draw_grid():
    '''
    plot_style = {
            1: ['--', 'orange', 'Repeat1'],
            2: ['-', 'red', 'Repeat2'],
            3: [':', 'blue', 'Repeat3'],
            4: ['-.', 'purple', 'Repeat4'],
            5: ['--', 'green', 'Repeat5'],
    }
    '''
    plot_style = {
            'normal': ['--', 'black', 'Truncated Normal'],
            'uniform': ['-', 'red', 'Uniform'],
    }
    
    root = 'results/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
    
    cat = os.listdir(root)
    paths = []
    for c in cat:
        if 'grid' not in c: continue
        folders = os.listdir(root+c)
        for folder in folders:
            paths.append(root + c + '/' + folder + '/')

    for path in paths:
        algo = path.split('/')[-2]
        if algo == 'linucb': prefix = 'LinUCB'
        elif algo == 'lints': prefix = 'LinTS'
        elif algo == 'glmucb': prefix = 'UCBGLM'
        fn = algo + '_grid'
        title = 'Cumulative regret v.s. Exploration rates for {}'.format(prefix)
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 18, 'axes.labelsize': 18, 'font.size': 12, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8)}
        pylab.rcParams.update(params)
        
        leg = []
        keys = os.listdir(path)
        for key in keys:
            if key not in plot_style: continue        
            data = np.loadtxt(path+key)
            explore = data[:,0]
            plot.plot(explore, data[:,1], linestyle = plot_style[key][0], color = plot_style[key][1], linewidth = 2)
            leg += [plot_style[key][2]]
        plot.legend(leg, loc='best', fontsize=12, frameon=False)
        plot.xlabel('Exploration rates')
        plot.ylabel('Cumulative Regret')

        dates = list(np.arange(0, 10.1, 1))
        xlabel = np.arange(0, explore[-1]+0.1, 1)
        plot.xticks(xlabel, dates)
        
        plot.title(title)
        fig.savefig('plots/{}.pdf'.format(fn), dpi=300, bbox_inches = "tight")
        print('file in path {} plotted and saved as {}.pdf'.format(path, fn))

draw_grid()