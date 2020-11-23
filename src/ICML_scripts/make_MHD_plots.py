import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import pandas as pn
import sys, os
from sklearn import preprocessing
from scipy import spatial
sys.path.insert(0, 'src')

# from functions.main_base import *
from functions.plotfuncs import *
import functions.fast_funcs as ff #mikis optimized base/main functions

############# gotta run this first ###############

coth = lambda x: 1./np.tanh(x)
sech = lambda x: 1./np.cosh(x)
csch = lambda x: 1./np.sinh(x)

l = 1.0
mu0 = 1.0

lb = np.log(np.array([.05, 1, .5, .5, .1])); ub = np.log(np.array([.2, 5, 3, 3, 1]))

#average velocity
def uavg(x):
	x = np.exp(x)
	return -x[:,2]*x[:,3]/x[:,4]**2*(1 - x[:,4]*l/np.sqrt(x[:,3]*x[:,0])*coth(x[:,4]*l/\
									np.sqrt(x[:,3]*x[:,0])))

#induced magnetic field
def Bind(x):
	x = np.exp(x)
	return x[:,2]*l*mu0/(2*x[:,4])*(1 - 2*np.sqrt(x[:,3]*x[:,0])/(x[:,4]*l)*np.tanh(x[:,4]*\
				l/(2*np.sqrt(x[:,3]*x[:,0]))))

#gradient of average velocity
def uavg_grad(x):
	x = np.exp(x)
	mu = x[:,0]; rho = x[:,1]; dp0 = x[:,2]; eta = x[:,3]; B0 = x[:,4]

	dudmu = -dp0*eta/B0**2*(B0*l/(2*np.sqrt(eta))*mu**(-3./2)*coth(B0*l/np.sqrt(eta*mu)) - \
			(B0*l)**2*np.sqrt(eta/mu)*csch(B0*l/np.sqrt(eta*mu))**2/(2*(eta*mu)**(3./2)))
	dudrho = 0.0*np.empty(x.shape[0])
	duddp0 = -eta/B0**2*(1 - B0*l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu)))
	dudeta = -dp0/B0**2*(1 - B0*l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu))) - \
		dp0*eta/B0**2*(B0*l/(2*np.sqrt(mu))*eta**(-3./2)*coth(B0*l/np.sqrt(eta*mu)) - \
			(B0*l)**2*np.sqrt(mu/eta)*csch(B0*l/np.sqrt(eta*mu))**2/(2*(eta*mu)**(3./2)))
	dudB0 = 2*dp0*eta/B0**3*(1 - B0*l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu))) - \
		dp0*eta/B0**2*(-l/np.sqrt(eta*mu)*coth(B0*l/np.sqrt(eta*mu)) + B0*l**2/(eta*mu)*\
												csch(B0*l/np.sqrt(eta*mu))**2)

	dudmu = dudmu[:,None]; dudrho = dudrho[:,None]; duddp0 = duddp0[:,None]
	dudeta = dudeta[:,None]; dudB0 = dudB0[:,None]
	return np.hstack((dudmu, dudrho, duddp0, dudeta, dudB0))*x*(ub - lb).reshape((1, 5))/2.

#gradient of induced magnetic field
def Bind_grad(x):
	x = np.exp(x)
	mu = x[:,0]; rho = x[:,1]; dp0 = x[:,2]; eta = x[:,3]; B0 = x[:,4]

	dBdmu = dp0*l*mu0/(2*B0)*(-(B0*l)**-1*np.sqrt(eta/mu)*np.tanh(B0*l/(2*np.sqrt(eta*mu))) +\
				eta*np.sqrt(eta*mu)*sech(B0*l/(2*np.sqrt(eta*mu)))**2/(2*(eta*mu)**(3./2)))
	dBdrho = 0.0*np.empty(x.shape[0])
	dBddp0 = l*mu0/(2*B0)*(1 - 2*np.sqrt(eta*mu)/(B0*l)*np.tanh(B0*l/(2*np.sqrt(eta*mu))))
	dBdeta = dp0*l*mu0/(2*B0)*(-(B0*l)**-1*np.sqrt(mu/eta)*np.tanh(B0*l/(2*np.sqrt(eta*mu)))+\
				mu*np.sqrt(eta*mu)*sech(B0*l/(2*np.sqrt(eta*mu)))**2/(2*(eta*mu)**(3./2)))
	dBdB0 = -dp0*l*mu0/(2*B0**2)*(1 - 2*np.sqrt(eta*mu)/(B0*l)*np.tanh(B0*l/\
		(2*np.sqrt(eta*mu)))) + dp0*l*mu0/(2*B0)*(2*np.sqrt(eta*mu)/(B0**2*l)*np.tanh(\
		B0*l/(2*np.sqrt(eta*mu))) - B0**-1*sech(B0*l/(2*np.sqrt(eta*mu)))**2)

	dBdmu = dBdmu[:,None]; dBdrho = dBdrho[:,None]; dBddp0 = dBddp0[:,None]
	dBdeta = dBdeta[:,None]; dBdB0 = dBdB0[:,None]
	return np.hstack((dBdmu, dBdrho, dBddp0, dBdeta, dBdB0))*x*(ub - lb).reshape((1, 5))/2.


########### Run program on uniform mesh ################

if __name__ == '__main__':

	outdir = os.path.abspath(os.path.join(os.getcwd(), 'ICML_results'))
	if not os.path.isdir(outdir):
	    os.makedirs(outdir)

	#The Hartmann problem
	meshy = ff.make_mesh(5,0.2)

	lb = np.log(np.array([.05, 1, .5, .5, .1])); ub = np.log(np.array([.2, 5, 3, 3, 1]))

	xu = lb + 2./(ub-lb)*(np.array(meshy) + 1)

	u = uavg(xu)
	B = Bind(xu)
	dB = preprocessing.normalize(Bind_grad(xu))
	du = preprocessing.normalize(uavg_grad(xu))


	mesh_kdtree = spatial.KDTree(meshy)

	np.random.seed(46)
	stepsize = .02
	seedPoint = np.ravel(2*np.random.rand(5, 1) - 1)
	am, fVals  = ff.build_AM_from_data(
	    seedPoint = seedPoint, mesh_kdtree = mesh_kdtree,
	    fSamples= u, gradPaths=du, stepsize=stepsize)

	fig = splinefitHu(am,fVals)
	fig.savefig(os.path.join(outdir,'Hartmann_uFitSpline.pdf'),bbox_inches = 'tight')

	numpts = len(am)
	sValues = np.linspace(0., numpts, numpts) / (numpts)
	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Values of $\gamma_{Hu}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Value: $x_i\,(\gamma_u(t))$', fontsize = 13)
	plt.plot(sValues, am[:,0], c='#66c2a5', label = r'$\log(\mu)$', markersize = 2, linewidth=2)
	plt.plot(sValues, am[:,1], c='#fc8d62', label = r'$\log(\rho)$', markersize = 2,linewidth=2)
	plt.plot(sValues, am[:,2], c='#8da0cb',  label = r'$\log(\frac{dp_0}{dt})$', markersize = 2,linewidth=2)
	plt.plot(sValues, am[:,3], c='#e78ac3', label = r'$\log(\eta)$', markersize = 2,linewidth=2)
	plt.plot(sValues, am[:,4], c='#a6d854', label = r'$\log(B_0)$', markersize = 2,linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	#plt.grid(True)
	# plt.savefig('Hartmann_uCoords.pdf',bbox_inches = 'tight')

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Derivatives of $\gamma_{Hu}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Rate of Change: $|\,\frac{dx_i\,(\gamma_u(t))}{dt}\,|$', fontsize = 13)
	plt.plot(sValues,np.abs(np.gradient(am[:,0])), c='#66c2a5', label = r'$|\log(\mu)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,1])), c='#fc8d62', label = r'$|\log(\rho)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,2])), c='#8da0cb',  label = r'$|\log(\frac{dp_0}{dt})^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,3])), c='#e78ac3', label = r'$|\log(\eta)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,4])), c='#a6d854', label = r'$|\log(B_0)^\prime|$', markersize = 3, linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	fig.savefig(os.path.join(outdir,'Hartmann_uDerivs.pdf'),bbox_inches = 'tight')

	am2, fVals2  = ff.build_AM_from_data(
	    seedPoint = seedPoint, mesh_kdtree = mesh_kdtree,
	    fSamples= B, gradPaths=dB, stepsize=.02)

	fig = splinefitHB(am2,fVals2)
	fig.savefig(os.path.join(outdir,'Hartmann_BFitSpline.pdf'),bbox_inches = 'tight')

	numpts2 = len(am2)
	sValues2 = np.linspace(0., numpts2, numpts2) / (numpts2)
	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Values of $\gamma_{HB}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Value: $x_i\,(\gamma_B(t))$', fontsize = 13)
	plt.plot(sValues2, am2[:,0], c='#66c2a5', label = r'$\log(\mu)$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,1], c='#fc8d62', label = r'$\log(\rho)$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,2], c='#8da0cb',  label = r'$\log(\frac{dp_0}{dt})$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,3], c='#e78ac3', label = r'$\log(\eta)$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,4], c='#a6d854', label = r'$\log(B_0)$', markersize = 3, linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	#plt.savefig('Hartmann_BCoords.pdf',bbox_inches='tight')

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Derivatives of $\gamma_{HB}(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Rate of Change: $|\,\frac{dx_i\,(\gamma_B(t))}{dt}\,|$', fontsize = 13)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,0])), c='#66c2a5', label = r'$|\log(\mu)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,1])), c='#fc8d62', label = r'$|\log(\rho)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,2])), c='#8da0cb',  label = r'$|\log(\frac{dp_0}{dt})^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,3])), c='#e78ac3', label = r'$|\log(\eta)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,4])), c='#a6d854', label = r'$|\log(B_0)^\prime|$', markersize = 3, linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	fig.savefig(os.path.join(outdir,'Hartmann_BDerivs.pdf'),bbox_inches = 'tight')

	#Idealized MHD Generator

	#load data from data files
	data = pn.DataFrame.from_csv('data_MHD/MHD_Generator_Data_Uavg.txt').as_matrix()
	Xu = data[:,:5]; u = data[:,5]; du = data[:,6:]

	data = pn.DataFrame.from_csv('data_MHD/MHD_Generator_Data_Bind.txt').as_matrix()
	XB = data[:,:5]; B = data[:,5]; dB = data[:,6:]

	#new upper/lower bounds
	lb = np.log(np.array([.001, .1, .1, .1, .1])); ub = np.log(np.array([.01, 10, .5, 10, 1]))

	#scale gradients according to the chain rule, get normalized inputs
	du = .5*(ub - lb)*Xu*du; XXu = 2*(np.log(Xu) - lb)/(ub - lb) - 1
	dB = .5*(ub - lb)*XB*dB; XXB = 2*(np.log(XB) - lb)/(ub - lb) - 1

	normdu = preprocessing.normalize(du)
	normdB = preprocessing.normalize(dB)


	mesh_kdtree_2 = spatial.KDTree(XXu)
	am, fVals  = ff.build_AM_from_data(
	    seedPoint = seedPoint, mesh_kdtree = mesh_kdtree_2,
	    fSamples=u, gradPaths=normdu, stepsize=.002)

	fig = splinefitu(am,fVals)
	fig.savefig(os.path.join(outdir,'MHD_uFitSpline.pdf'),bbox_inches = 'tight')

	numpts = len(am)
	sValues = np.linspace(0., numpts, numpts) / (numpts)
	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Values of $\gamma_u(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Value: $x_i\,(\gamma_u(t))$', fontsize = 13)
	plt.plot(sValues, am[:,0], c='#66c2a5', label = r'$\log(\mu)$', markersize = 2, linewidth=2)
	plt.plot(sValues, am[:,1], c='#fc8d62', label = r'$\log(\rho)$', markersize = 2,linewidth=2)
	plt.plot(sValues, am[:,2], c='#8da0cb',  label = r'$\log(\frac{dp_0}{dt})$', markersize = 2,linewidth=2)
	plt.plot(sValues, am[:,3], c='#e78ac3', label = r'$\log(\eta)$', markersize = 2,linewidth=2)
	plt.plot(sValues, am[:,4], c='#a6d854', label = r'$\log(B_0)$', markersize = 2,linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	# plt.savefig('MHD_uCoords.pdf', bbox_inches='tight')

	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Derivatives of $\gamma_u(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Rate of Change: $|\,\frac{dx_i\,(\gamma_u(t))}{dt}\,|$', fontsize = 13)
	plt.plot(sValues,np.abs(np.gradient(am[:,0])), c='#66c2a5', label = r'$|\log(\mu)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,1])), c='#fc8d62', label = r'$|\log(\rho)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,2])), c='#8da0cb',  label = r'$|\log(\frac{dp_0}{dt})^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,3])), c='#e78ac3', label = r'$|\log(\eta)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues,np.abs(np.gradient(am[:,4])), c='#a6d854', label = r'$|\log(B_0)^\prime|$', markersize = 3, linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	plt.ylim(-0.0001,0.0018)
	#plt.grid(True)
	fig.savefig(os.path.join(outdir,'MHD_uDerivs.pdf'),bbox_inches = 'tight')

	am2, fVals2  = ff.build_AM_from_data(
	    seedPoint = seedPoint, mesh_kdtree = mesh_kdtree_2,
	    fSamples= B, gradPaths=normdB, stepsize=.002)

	fig = splinefitB(am2,fVals2)
	fig.savefig(os.path.join(outdir,'MHD_BFitSpline.pdf'),bbox_inches = 'tight')


	numpts2 = len(am2)
	sValues2 = np.linspace(0., numpts2, numpts2) / (numpts2)
	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Values of $\gamma_B(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Value: $x_i\,(\gamma_B(t))$', fontsize = 13)
	plt.plot(sValues2, am2[:,0], c='#66c2a5', label = r'$\log(\mu)$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,1], c='#fc8d62', label = r'$\log(\rho)$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,2], c='#8da0cb',  label = r'$\log(\frac{dp_0}{dt})$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,3], c='#e78ac3', label = r'$\log(\eta)$', markersize = 3, linewidth=2)
	plt.plot(sValues2, am2[:,4], c='#a6d854', label = r'$\log(B_0)$', markersize = 3, linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	# plt.savefig('MHD_BCoords.pdf', bbox_inches='tight')


	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(r'Coordinate Derivatives of $\gamma_B(t)$', fontsize = 13)
	ax.set_xlabel(r'Curve Parameter: $t$', fontsize = 13)
	#ax.set_ylabel(r'Rate of Change: $|\,\frac{dx_i\,(\gamma_B(t))}{dt}\,|$', fontsize = 13)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,0])), c='#66c2a5', label = r'$|\log(\mu)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,1])), c='#fc8d62', label = r'$|\log(\rho)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,2])), c='#8da0cb',  label = r'$|\log(\frac{dp_0}{dt})^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,3])), c='#e78ac3', label = r'$|\log(\eta)^\prime|$', markersize = 3, linewidth=2)
	plt.plot(sValues2,np.abs(np.gradient(am2[:,4])), c='#a6d854', label = r'$|\log(B_0)^\prime|$', markersize = 3, linewidth=2)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlim(0,1)
	plt.ylim(0,0.00175)
	fig.savefig(os.path.join(outdir,'MHD_BDerivs.pdf'),bbox_inches = 'tight')
