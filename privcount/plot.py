#!/usr/bin/env python
# See LICENSE for licensing information

import sys, os, argparse, json
from itertools import cycle
from math import sqrt
# NOTE see plotting imports below in import_plotting()

"""
python plot.py --help

compare output from multiple privcount results files. run like:
python privcount/plot.py -d results1.txt test1 -d results2.txt test2 ...
"""

LINEFORMATS="k,r,b,g,c,m,y"

class PlotDataAction(argparse.Action):
    '''
    a custom action for passing in experimental data directories when plotting
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        # extract the path to our data, and the label for the legend
        datapath = os.path.abspath(os.path.expanduser(values[0]))
        label = values[1]
        # check the path exists
        if not os.path.exists(datapath): raise argparse.ArgumentError(self, "The supplied path to the plot data does not exist: '{0}'".format(datapath))
        # remove the default
        if "_didremovedefault" not in namespace:
            setattr(namespace, self.dest, [])
            setattr(namespace, "_didremovedefault", True)
        # append out new experiment path
        dest = getattr(namespace, self.dest)
        dest.append((datapath, label))

def main():
    parser = argparse.ArgumentParser(
        description='Utility to help plot results from PrivCount',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_plot_args(parser)
    args = parser.parse_args()
    run_plot(args)

def add_plot_args(parser):
    parser.add_argument('-d', '--data',
        help="""Append a PATH to a privcount tallies.json file,
                and the LABEL we should use for the graph legend for this
                set of experimental results""",
        metavar=("PATH", "LABEL"),
        nargs=2,
        required="True",
        action=PlotDataAction, dest="experiments")

    parser.add_argument('-p', '--prefix',
        help="a STRING filename prefix for graphs we generate",
        metavar="STRING",
        action="store", dest="prefix",
        default=None)

    parser.add_argument('-f', '--format',
        help="""A comma-separated LIST of color/line format strings to cycle to
                matplotlib's plot command (see matplotlib.pyplot.plot)""",
        metavar="LIST",
        action="store", dest="lineformats",
        default=LINEFORMATS)

def import_plotting():
    global matplotlib
    import matplotlib; matplotlib.use('Agg') # for systems without X11
    global PdfPages
    from matplotlib.backends.backend_pdf import PdfPages
    global pylab
    import pylab

    pylab.rcParams.update({
        'backend': 'PDF',
        'font.size': 16,
        'figure.figsize': (6,4.5),
        'figure.dpi': 100.0,
        'figure.subplot.left': 0.15,
        'figure.subplot.right': 0.95,
        'figure.subplot.bottom': 0.15,
        'figure.subplot.top': 0.95,
        'grid.color': '0.1',
        'axes.grid' : True,
        'axes.titlesize' : 'small',
        'axes.labelsize' : 'small',
        'axes.formatter.limits': (-4,4),
        'xtick.labelsize' : 'small',
        'ytick.labelsize' : 'small',
        'lines.linewidth' : 2.0,
        'lines.markeredgewidth' : 0.5,
        'lines.markersize' : 10,
        'legend.fontsize' : 'x-small',
        'legend.fancybox' : False,
        'legend.shadow' : False,
        'legend.borderaxespad' : 0.5,
        'legend.numpoints' : 1,
        'legend.handletextpad' : 0.5,
        'legend.handlelength' : 1.6,
        'legend.labelspacing' : .75,
        'legend.markerscale' : 1.0,
        # turn on the following to embedd fonts; requires latex
        #'ps.useafm' : True,
        #'pdf.use14corefonts' : True,
        #'text.usetex' : True,
    })

    try: pylab.rcParams.update({'figure.max_num_figures':50})
    except: pylab.rcParams.update({'figure.max_open_warning':50})
    try: pylab.rcParams.update({'legend.ncol':1.0})
    except: pass

def run_plot(args):
    import_plotting()

    lflist = args.lineformats.strip().split(",")
    lfcycle = cycle(lflist)

    plot_info = {}
    for (path, label) in args.experiments:
        dataset_color = lfcycle.next()
        dataset_label = label
        fin = open(path, 'r')
        histograms = json.load(fin)
        fin.close()

        for name in histograms.keys():
            plot_info.setdefault(name, {'datasets':[], 'error':0, 'dataset_colors':[], 'dataset_labels':[], 'bin_labels':[]})
            plot_info[name]['dataset_colors'].append(dataset_color)
            plot_info[name]['dataset_labels'].append(dataset_label)

            dataset = []
            bin_labels = []
            for (left, right, val) in histograms[name]['bins']:
                if right == float('inf'):
                    right = '{}'.format(r'$\infty$')
                elif 'Ratio' not in name:
                    right = int(right)
                if left == float('-inf'):
                    left = '{}'.format(r'$-\infty$')
                elif 'Ratio' not in name:
                    left = int(left)
                bin_labels.append("[{},{})".format(left, right))
                dataset.append(val)
            plot_info[name]['datasets'].append(dataset)

            if 'sigma' in histograms[name]:
                sigma = float(histograms[name]['sigma'])
                plot_info[name]['error'] = int(round(2 * sqrt(3) * sigma)) %  1000000000000000

            if len(plot_info[name]['bin_labels']) == 0:
                plot_info[name]['bin_labels'] = bin_labels

    page = PdfPages("{0}privcount.results.pdf".format(args.prefix+'.' if args.prefix is not None else ''))
    # test data
    '''
    datasets = [[5, 10, 12, 7, 4], [3, 4, 5, 6, 7]]
    dataset_labels = ["tor", "shadow"]
    dataset_colors = ["red", "green"]
    bar_xlabels = ['[0,128)', '[128,256)', '[256,512)', '[512,1024)', '[1024,\n2048)']
    plot_bar_chart(page, datasets, dataset_labels, dataset_colors, bar_xlabels, title="test", xlabel="test_x", ylabel="test_y")
    '''
    for name in sorted(plot_info.keys()):
        dat = plot_info[name]
        plot_bar_chart(page, dat['datasets'], dat['dataset_labels'], dat['dataset_colors'], dat['bin_labels'], err=dat['error'], title=name)
    page.close()

def plot_bar_chart(page, datasets, dataset_labels, dataset_colors, x_group_labels, err=0, title=None, xlabel='Bins', ylabel='Counts'):
    assert len(datasets) == len(dataset_colors) == len(dataset_labels)
    for dataset in datasets:
        assert len(dataset) == len(datasets[0])
        assert len(dataset) == len(x_group_labels)

    num_x_groups = len(datasets[0])
    x_group_locations = pylab.arange(num_x_groups)
    width = 1.0 / float(len(datasets)+1)

    figure = pylab.figure()
    axis = figure.add_subplot(111)
    bars = []

    for i in xrange(len(datasets)):
        bar = axis.bar(x_group_locations + (width*i), datasets[i], width, yerr=err, color=dataset_colors[i], error_kw=dict(ecolor='pink', lw=3, capsize=6, capthick=3))
        bars.append(bar)

    if title is not None:
        axis.set_title(title)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    if xlabel is not None:
        axis.set_xlabel(xlabel)

    axis.set_xticks(x_group_locations + width*len(datasets)/2)
    x_tick_names = axis.set_xticklabels(x_group_labels)
    rot = 0 if num_x_groups == 1 else 15
    pylab.setp(x_tick_names, rotation=rot, fontsize=10)
    axis.set_xlim(-width, num_x_groups)
    y_tick_names = axis.get_yticklabels()
    pylab.setp(y_tick_names, rotation=0, fontsize=10)

    axis.legend([bar[0] for bar in bars], dataset_labels)
    page.savefig()
    pylab.close()

if __name__ == '__main__': sys.exit(main())
