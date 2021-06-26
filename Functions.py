from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from ipywidgets import *
import pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)   
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def plot_X(X,ax):    
    ax.pcolormesh(M, S, X,vmin = 0, vmax=1,rasterized=True) 
    return
def plot_contour(X,X_1,ax,cut=0.01, Levels = 3,Contour="Xbase"):
    if Contour == "Xbase":
        Xcontour = Xbase.copy()
    elif Contour == "Xbasenet":
        Xcontour = Xbasenet.copy()
    else:
        Xcontour = (M<Contour)*(S<Contour)
    if Levels == 2:
        levels = [cut]
    else:
        levels = [cut,(1-cut)]    
    ax.contour(M, S, Xcontour, levels = levels,colors="darkorange",linewidths = 0.5)
    ax.contour(M, S, X,        levels = levels,colors="k",linewidths = 0.5)
    diff=abs(X-X_1)
    diff[-1][-1]=1
    ax.contour(M, S, diff, levels = [0.01],colors="w",linewidths = 0.5)
    return

def contour_count(cut, X,X_1,Contour="Xbase"):
    if Contour == "Xbase":
        Xlist = Xbase > cut
        Xsize = np.sum(Xlist)
    elif Contour == "Xbasenet":
        Xlist = Xbasenet > cut
        Xsize = np.sum(Xlist)
    else:
        Xlist = (M<Contour)*(S<Contour)
        Xsize = np.sum(Xlist)
    Y = X + X_1    
    Z = np.array([np.sum((y<2*cut)*Xlist)/Xsize for y in Y])
    O  = np.array([np.sum((y>2*(1-cut))*Xlist)/Xsize for y in Y])
    MIX  =1-Z-O
    F = np.array([np.sum((abs(diff)<0.02)*Xlist)/Xsize for diff in X-X_1])
    Z0= np.sum(Xbase[Xlist]<cut)/Xsize
    O0 = np.sum(Xbase[Xlist]>1-cut)/Xsize
    M0 = 1-Z0-O0
    return Z,O,MIX,F,Z0,O0,M0

def plot_stats(ax, X, X_1, cut=0.2, Contour="Xbase", xlim = 1, label = False):
    Z,O,M,F,Z0,O0,M0 = contour_count(cut, X,X_1,Contour)
    ax.plot(shifts, Z, label = label*"zero")
    ax.plot(shifts, O, label = label*"one")
    ax.plot(shifts, M, label = label*"mix")
    ax.plot(shifts, [1-i for i  in F], label = label*"fluc")
    ax.plot(shifts, F, label = label*"equi")
    ax.hlines(Z0,0,xlim, linestyle = "dotted", color = "blue",label =  label*"Z0")
    ax.hlines(O0,0,xlim, linestyle = "dashed", color = "orange", label = label*"O0")
    ax.hlines(M0,0,xlim, linestyle = "dashdot", color = "green", label = label*"M0")
    ax.grid()
    ax.set_xlim((0,xlim))
    return

def double_Integral(X, Y, Z):
    """
    numerically integrates over X and Y weighted by Z
    X= x-matrix
    Y= y-matrix
    Z= z-matrix
    """
    xmin = np.min(X)
    xmax = np.max(X)
    ymin = np.min(Y)
    ymax = np.max(Y)
    nx, ny = Z.shape
    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))
    
    #internal
    Z_Internal = Z[1:-1, 1:-1]

    # sides: up, down, left, right
    (Z_u, Z_d, Z_l, Z_r) = (Z[0, 1:-1], Z[-1, 1:-1], Z[1:-1, 0], Z[1:-1, -1])

    # corners
    (Z_ul, Z_ur, Z_dl, Z_dr) = (Z[0, 0], Z[0, -1], Z[-1, 0], Z[-1, -1])

    return dS * (np.sum(Z_Internal)\
                + 0.5 * (np.sum(Z_u) + np.sum(Z_d) + np.sum(Z_l) + np.sum(Z_r))\
                + 0.25 * (Z_ul + Z_ur + Z_dl + Z_dr))
