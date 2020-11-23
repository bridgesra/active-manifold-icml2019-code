# This file contains functions for making the plots in the examples

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm

# Basically replaces everything else
def splinePlot(activeManifold, fValues, func, sub):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of ${}$ Along $\gamma_{}(t)$'.format(func, sub), fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: ${}\,(\gamma_{}\,(t))$'.format(func, sub), fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return plt.savefig(str(func) + str(sub) + 'FitSpline' + '.pdf')

################## toy examples ##############
def splinefitEll(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $L$ Along $\gamma(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $L\,(\gamma\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return plt.savefig('EllFitSpline.pdf')


def splinefitQ(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $Q$ Along $\gamma(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $Q$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $Q\,(\gamma\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return plt.savefig('CubeFitSpline.pdf')


def splinefitSS(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $S$ Along $\gamma(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $S\,(\gamma\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return plt.savefig('SquareFitSpline.pdf')


def splinefitf3(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $f_3$ Along $\gamma(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $f_3\,(\gamma\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return fig

################ PV ################
def PVsplinefit(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $P_{max}$ Along $\gamma(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $P_{max}\,(\gamma\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return plt.savefig('PVFitSpline.pdf')


############### MHD ###############
def splinefitHB(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $B_{ind}$ Along $\gamma_{HB}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $B_{ind}\,(\gamma_{HB}\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return fig
	# return plt.savefig('Hartmann_BFitSpline.pdf')


def splinefitHu(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $u_{avg}$ Along $\gamma_{Hu}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $u_{avg}\,(\gamma_{Hu}\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return fig
	# return plt.savefig('Hartmann_uFitSpline.pdf')


def splinefitB(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $B_{ind}$ Along $\gamma_{B}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $B_{ind}\,(\gamma_{B}\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return fig
	# return plt.savefig('MHD_BFitSpline.pdf')


def splinefitu(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $u_{avg}$ Along $\gamma_{u}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $u_{avg}\,(\gamma_{u}\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return fig
	# return plt.savefig('MHD_uFitSpline.pdf')


############ Ebola #############
def splinefitL(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $R_0$ Along $\gamma_{L}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $R_0\,(\gamma_{L}\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return plt.savefig('LFitSpline.pdf')


def splinefitSL(activeManifold, fValues):
	numpts = len(activeManifold)

	sValues = np.linspace(0., numpts, numpts) / (numpts)
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	spline = PchipInterpolator(sValues,fValues)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $R_0$ Along $\gamma_{SL}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $R_0\,(\gamma_{SL}\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, spline(xp), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return plt.savefig('SLFitSpline.pdf')




################# Out of date versions #############
def model_fitB(model, activeManifold, fValues):
	"""
	Compute and plot a least-squares fit to active manifold, given a model function

	Input:	model -- function to use for fit: f(x, *params)
			activeManifold -- array, points of active manifold, arranged vertically
			fValues -- array, function values of active manifold points

	Output: params -- array, optimal fit parameters for the model function
			variance -- array, variance in the parameters
	"""

	numpts = len(activeManifold)

	# Create sValues, evenly spaced in [0,1]
	sValues = np.linspace(0., numpts, numpts) / (numpts)

	# Fit fValues to a model function
	params, covariance = curve_fit(model, sValues, fValues)

	# Parameters for visualization
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $B_{ind}$ Along $\gamma_{B}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $B_{ind}\,(\gamma_B\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=15)
	plt.plot(xp, model(xp, *params), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return params, covariance, plt.savefig('MHD_BFit.png', dpi=1200)


def model_fitLib(model, activeManifold, fValues):
	"""
	Compute and plot a least-squares fit to active manifold, given a model function

	Input:	model -- function to use for fit: f(x, *params)
			activeManifold -- array, points of active manifold, arranged vertically
			fValues -- array, function values of active manifold points

	Output: params -- array, optimal fit parameters for the model function
			variance -- array, variance in the parameters
	"""

	numpts = len(activeManifold)

	# Create sValues, evenly spaced in [0,1]
	sValues = np.linspace(0., numpts, numpts) / (numpts)

	# Fit fValues to a model function
	params, covariance = curve_fit(model, sValues, fValues)

	# Parameters for visualization
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $R_0$ Along $\gamma_{L}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $R_0\,(\gamma_{L}\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='xkcd:puce', s=30)
	plt.plot(xp, model(xp, *params), '-', c='xkcd:pine green', lw = 2)
	plt.xlim(0,1)
	plt.grid(True)
	return params, covariance, plt.savefig('LiberiaFit.png')


def PV_flexModel(model, activeManifold, fValues):
	"""
	Compute and plot a least-squares fit to active manifold, given a model function

	Input:	model -- function to use for fit: f(x, *params)
			activeManifold -- array, points of active manifold, arranged vertically
			fValues -- array, function values of active manifold points

	Output: params -- array, optimal fit parameters for the model function
			variance -- array, variance in the parameters
	"""

	numpts = len(activeManifold)

	# Create sValues, evenly spaced in [0,1]
	sValues = np.linspace(0., numpts, numpts) / (numpts)

	# Fit fValues to a model function
	params, covariance = curve_fit(model, sValues, fValues)

	# Parameters for visualization
	xp = np.linspace(0., numpts, 2*(numpts)) / (numpts)

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Value of $P_{max}$ Along $\gamma(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	ax.set_ylabel(r'Function Value: $P_{max}\,(\gamma\,(t))$', fontsize = 13)
	plt.scatter(sValues,fValues, c='#fc8d62', s=17)
	plt.plot(xp, model(xp, *params), '-', c='#66c2a5', lw = 3)
	plt.xlim(0,1)
	plt.grid(True)
	return params, covariance, plt.savefig('PVfit.png')
