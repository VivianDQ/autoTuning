import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab

def draw_grid():
    plot_style = {
            1: ['--', 'orange', 'Repeat1'],
            2: ['-', 'red', 'Repeat2'],
            3: [':', 'blue', 'Repeat3'],
            4: ['-.', 'purple', 'Repeat4'],
            5: ['--', 'green', 'Repeat5'],
    }
    
    root = 'results/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
    
    cat = os.listdir(root)
    paths = []
    for c in cat:
        if 'grid' not in c: continue
        files = os.listdir(root+c)
        for file in files:
            paths.append(root + c + '/' + file)

    for path in paths:
        algo = path.split('/')[-1]
        if algo == 'linucb': prefix = 'LinUCB'
        elif algo == 'lints': prefix = 'LinTS'
        elif algo == 'glmucb': prefix = 'UCBGLM'
        fn = algo + '_grid'
        if 'simulation' in fn:
            _, dstr, Kstr = fn.split('_')
            d = int(dstr[1:])
            K = int(Kstr[1:])
            title = 'Cumulative regret v.s. Exploration rates for {}'.format(prefix)
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 18, 'axes.labelsize': 18, 'font.size': 12, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8)}
        pylab.rcParams.update(params)

        data = np.loadtxt(path)
        explore = data[0,:]
        for i in range(1, data.shape[1]):
            plot.plot(explore, data[i,:], linestyle = plot_style[i][0], color = plot_style[i][1], linewidth = 2)

        plot.xlabel('Exploration rates')
        plot.ylabel('Cumulative Regret')
        if data.shape[1] > 2:
            plot.legend(plot_style[i][2], loc='best', fontsize=12, frameon=False)
            
        if 'lin' in path:
            dates = list(np.arange(0, 10.1, 1))
            xlable = np.arange(0, len(explore), 2)
            plot.xticks(xlabel, dates)
            
        plot.title(title)
        fig.savefig('plots/{}.pdf'.format(fn), dpi=300, bbox_inches = "tight")
        print('file in path {} plotted and saved as {}.pdf'.format(path, fn))

draw_grid()