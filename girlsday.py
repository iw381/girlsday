import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck15
import matplotlib.colors as colors
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import numpy.random
from numpy.linalg import det
from mpl_toolkits import mplot3d
import matplotlib.ticker as ticker

plt.rc('font', family='serif')
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['axes.labelsize'] = 34
mpl.rcParams['legend.fontsize'] = 24
def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${}\!\times 10^{{{}}}$'.format(a, b)

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import healpy as hp

import sys, platform, os
import camb
from camb import model, initialpower


import warnings
warnings.filterwarnings('ignore')


########################## PART 1 #############################################################

#define plotting routines
def plotEarth(earthMap):
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)

    #plotting map
    im = ax.imshow(earthMap,vmin=-8000, vmax=6705)
    
    #setting position and size of colorbar (xPos,yPos,width,height)
    cax = fig.add_axes([0.32, 0.25, 0.4, 0.018]) 
    cbar = fig.colorbar(im,orientation="horizontal",cax=cax)
    cbar.set_label('Höhe über N.N. [m]', size=24, labelpad=20)
    ax.axis('off')

def plotPowerspectrum(clCur,LMAX,isCMB=False):    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)

    #split the ell and cl arrays it two halfes 
    # s.t. they can be plotted in two different plot styles 
    ellLeq = ell[ell<=LMAX]
    clDimlessLeq = ellLeq* (ellLeq + 1) * clCur[ell<=LMAX]
    ellGeq = ell[ell>=LMAX]
    clDimlessGeq = ellGeq * (ellGeq + 1) * clCur[ell>=LMAX]
    
    #plot the C_l power spectrum
    ax.plot(ellLeq, clDimlessLeq, c="#222222", lw=1.5)
    ax.plot(ellGeq, clDimlessGeq, c="#666666", lw=1.5)
    
    # add the vertical line and shaded region
    ax.axvline(x=LMAX,lw=2)
    xLim = np.array(ax.get_xlim())    
    xLim[1] = 210
    if (isCMB):
        xLim[1] = 1024

    ax.axvspan(LMAX, xLim[1], facecolor='0.2', alpha=0.2)

    # plot cosmetics
    if (not isCMB):
        ax.set_xscale("log")
    ax.set_xlim(xLim)
    ax.set_xlabel(r"$\ell$", size=24)
    if (not isCMB):
        ax.set_yscale("log")
    ax.set_ylabel(r"$\ell(\ell+1)C_{\ell}$", size=24)

    plt.title("Leistungsspektrum $C_{\ell}$",fontsize=24)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(True)


earthMap = hp.read_map("etopo5_hp.fits")
nside = hp.pixelfunc.get_nside(earthMap)

mollview = hp.mollview(earthMap,return_projected_map=True)[::-1];
hp.graticule()
plt.close("all")



########################## PART 2 #############################################################


LMAX = 1024
cl = hp.anafast(earthMap, lmax=LMAX)
ell = np.arange(len(cl))

def plotAlmMap (LMAX):
    alm = hp.map2alm(earthMap,lmax=LMAX)
    mapReconstr = hp.alm2map(alm, nside,lmax=LMAX)
    mollviewMapReconstr = hp.mollview(mapReconstr,return_projected_map=True)[::-1];
    plt.close("all")

    plotEarth(mollviewMapReconstr)
    plotPowerspectrum(cl,LMAX)


int_wdgt = widgets.IntSlider(
    description=r'$\ell_\text{max}$:',
    fontsize=34,
    value=1,
    min=1, max=200, step=15,
    layout=widgets.Layout(width='90%'))


########################## PART 3 #############################################################


#number of $\ell$-modes that are to be plotted in the triangular plot
#CAUTION: plot below has been formatted according to maxL = 8; 
#         maxL can be changed but then plot has to be reformatted 
maxL = 8

#calculate decomposition
alm = hp.map2alm(earthMap,lmax=LMAX)

#extract the modes for the triangular plot and safe them in the 
# 4d array 'mapArray' with structure [l,m,xCoord,yCoord]  
mapArray = np.zeros([maxL+1,maxL+1,400,800])
for curL in range(maxL+1):
    offset = 0
    for i in range(curL+1):

        indCur = offset+curL
        offset = offset+LMAX-i

        almNew = alm.copy()
        almNew[:indCur] = 0
        almNew[indCur+1:] = 0

        mapReconstr = hp.alm2map(almNew, nside,lmax=LMAX)

        mapArray[curL,i] = hp.mollview(mapReconstr,return_projected_map=True)[::-1];
        plt.close("all")

#formatting paramters that need to be adjusted if maxL is changed
leftMarginEll       = -700 
leftMarginM         = 150
topMargMfirst       = -140
topMargMfollow      = 160

#triangular plot
def triangularPlot ():
    fig, axes = plt.subplots(figsize=(13, 13), sharex=True, sharey=True, ncols=maxL+1, nrows=maxL)
    fig.subplots_adjust(hspace = -0.8)
    for i in range(maxL):
        for j in range(maxL):
            if i<j:
                if(i==0):
                    axes[i, j].text(leftMarginM,topMargMfollow,"$m=%d$"%j,size=18)
                axes[i, j].axis('off')
            else:
                if(i==0):
                    axes[i, j].text(leftMarginM,topMargMfirst,"$m=%d$"%j,size=18)
                axes[i, j].imshow(mapArray[i,j])
                axes[i, j].axis('off')
                if(j==0):
                    axes[i, j].text(leftMarginEll,230,"$\ell=%d$:"%i,size=18)


    #last column that gives the cumulative sum of all modes with l<=lCur
    axes[0, maxL].text(leftMarginM,topMargMfirst,r"$\Sigma_{\ell\leq\ell_{cur},m(\ell)}$",size=18)
    for curL in range(maxL):
        curSummedMap = np.zeros_like(mapArray[0,0])
        for j in range(curL):
            curSummedMap += np.sum(mapArray[j],axis=0)
        
        axes[curL, maxL].imshow(curSummedMap)
        axes[curL, maxL].axis('off')



########################## PART 4 #############################################################


#define plotting routines for CMB

def plotCMB(cmbMap):
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)

    #plotting map
    im = ax.imshow(cmbMap,vmin=-0.0005, vmax=0.0005)
    
    #setting position and size of colorbar (xPos,yPos,width,height)
    cax = fig.add_axes([0.32, 0.25, 0.4, 0.018]) 
    cbar = fig.colorbar(im,orientation="horizontal",cax=cax)
    cbar.set_label('$\Delta T$', size=24, labelpad=20)
    ax.axis('off')


cmbMap = hp.read_map("planckSmica512.fits")
nside = hp.pixelfunc.get_nside(cmbMap)

if (nside!=512):
    raise NameError('Code only implemented for nsideCMB = nsideEart = 512')

mollviewCMB = hp.mollview(cmbMap,return_projected_map=True)[::-1];
hp.graticule()
plt.close("all")


LMAX = 1024
clCMB = hp.anafast(cmbMap, lmax=LMAX)
ellCMB = np.arange(len(clCMB))[2:]


def plotPowerspectrumCMB(clCur,LMAX,isCMB=False):    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)

    #split the ell and cl arrays it two halfes 
    # s.t. they can be plotted in two different plot styles 
    ellLeq = ellCMB[ellCMB<=LMAX]
    clDimlessLeq = ellLeq* (ellLeq + 1) * clCur[ellCMB<=LMAX]
    ellGeq = ellCMB[ellCMB>=LMAX]
    clDimlessGeq = ellGeq * (ellGeq + 1) * clCur[ellCMB>=LMAX]
    
    #plot the C_l power spectrum
    ax.plot(ellLeq, clDimlessLeq, c="#222222", lw=1.5)
    ax.plot(ellGeq, clDimlessGeq, c="#666666", lw=1.5)
    
    # add the vertical line and shaded region
    ax.axvline(x=LMAX,lw=2)
    xLim = np.array(ax.get_xlim())    
    xLim[1] = 210
    if (isCMB):
        xLim[1] = 1024

    ax.axvspan(LMAX, xLim[1], facecolor='0.2', alpha=0.2)

    # plot cosmetics
    if (not isCMB):
        ax.set_xscale("log")
    ax.set_xlim(xLim)
    ax.set_xlabel(r"$\ell$", size=24)
    if (not isCMB):
        ax.set_yscale("log")
    ax.set_ylabel(r"$\ell(\ell+1)C_{\ell}$", size=24)

    plt.title("Leistungsspektrum $C_{\ell}$",fontsize=24)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(True)

def plotAlmMapCMB (LMAX):
    alm = hp.map2alm(cmbMap,lmax=LMAX)
    mapReconstr = hp.alm2map(alm, nside,lmax=LMAX)
    print(min(mapReconstr),max(mapReconstr))
    mollviewMapReconstr = hp.mollview(mapReconstr,return_projected_map=True)[::-1];
    plt.close("all")

    plotCMB(mollviewMapReconstr)
    plotPowerspectrumCMB(clCMB[2:],LMAX,True)

int_wdgtCMB = widgets.IntSlider(
    description=r'$\ell_\text{max}$:',
    fontsize=34,
    value=116,
    min=1, max=200, step=15,
    layout=widgets.Layout(width='90%'))


########################## PART 4 #############################################################

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL_fiducial=powers['total']
ls = np.arange(totCL_fiducial.shape[0])

def createCldata (oBar):
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=oBar, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL_new=powers['total']
    ls = np.arange(totCL.shape[0])    

    np.savetxt("ClSpectra/Cl_%.3f.dat"%oBar,(ls,totCL_new[:,0]))

def plotCMBps (oBarPercent):
    
    
    oBar = oBarPercent/100.
    ls,clCur = np.loadtxt("ClSpectra/Cl_%.3f.dat"%oBar)

    #print(clCur)
    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)

    startI = 10
    ax.plot(ls[startI:],totCL_fiducial[startI:], color='k')
    ax.plot(ls[startI:],clCur[startI:], color='#2966a3')
    ax.set_xlim([0,2500])
    ax.set_ylim([0,15000])
    barPercent = oBar*100
    ax.text(1750,13000,"Atome: %.1f%%"%barPercent,size=28,color='#2966a3')
    ax.text(1750,11800,"Atome: 2.2%",size=28,color='k')

    
def checkData(minObar,maxObar,stepSize):
    oBar = minObar
    while oBar < maxObar:
        fileExists = os.path.isfile("ClSpectra/Cl_%.3f.dat"%oBar)
        if(not fileExists):
            createCldata(oBar)
        oBar += stepSize

# set parameters fro slider widget        
minObar = 0.002
maxObar = 0.2
stepSize = 0.01

#check if all necessary files are available
#create them if necessary
checkData(minObar,maxObar,stepSize)

#loadFiducial data
ls,totCL_fiducial = np.loadtxt("ClSpectra/Cl_0.022.dat")

int_wdgtPowerSpec = widgets.FloatSlider(
    description=r'Atome [in %]:',
    fontsize=34,
    value=minObar,
    min=minObar*100, max=maxObar*100, step=stepSize*100,
    layout=widgets.Layout(width='90%'))


