import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os, sys
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

import functions.ground.base_am_fxns as bf #anthonys base functions
from functions.testing_plotting_functions import *

# To recreate 3 level set plots in paper submission:
#   1. Run this script from command line:
#       `python src/ICML_scripts/make_plots.py`
#   2. look at the pretty plots you just made in 'ICML_results'

###
# Script makes level set plots in paper--
# Creates plot for AM algo on a 50x50 uniformly sampled mesh on [-1,-1]^2
# for each of 2d test functions f1. f2. f3 as defined in paper
# builds /tests manifold using same start pt [.01, .43] and same 2000\500 test train split
#
# writes out figure files 'f1_eg_plot_500tps.png', 'f2_eg_plot_500tps.png', 'f3_eg_plot_500tps.png'
# to folder 'ICML_results'
###

if __name__ == '__main__':

    outdir = os.path.abspath(os.path.join(os.getcwd(), 'ICML_results'))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    fig = plot_level_set_path_results(numPoints=50, f =bf.f1, grad_f=bf.gradf1,
        stepsize = None,test_train_split_seed=0, init_pt_seed = 0, test_size=.2,seedPoint = None,
        outpath=os.path.join(outdir,'f1_eg_plot_500pts.png'))

    fig = plot_level_set_path_results(numPoints=50, f =bf.f2, grad_f=bf.gradf2,
        stepsize = None,test_train_split_seed=0, init_pt_seed = 0, test_size=.2,seedPoint = None,
        outpath=os.path.join(outdir,'f2_eg_plot_500pts.png'))

    fig = plot_level_set_path_results(numPoints=50, f =bf.f3, grad_f=bf.gradf3,
        stepsize = None,test_train_split_seed=0, init_pt_seed = 0, test_size=.2,seedPoint = None,
        outpath=os.path.join(outdir,'f3_eg_plot_500pts.png'))
