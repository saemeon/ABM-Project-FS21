from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from ipywidgets import *
import pickle


class ABM(object):
    def __init__(self,net, phi1, phi2, par, alpha = 1):
        mu1, sig1, mu2, sig2 = par
        self.N = net.number_of_nodes()
        self.alpha = alpha
        if phi1 == "in":
            self.phi1 = self.phi1_in
        if phi1 == "out":
            self.phi1 = self.phi1_out
        if phi1 == "red":
            self.phi1 = self.phi1_red
        if phi2 == "in":
            self.phi2 = self.phi2_in
        if phi2 == "out":
            self.phi2 = self.phi2_out
        if phi2 == "red":
            self.phi2 = self.phi2_red
        self.theta1  = np.random.normal(mu1,sig1,self.N)
        self.theta2 = np.random.normal(mu2,sig2,self.N)
        self.s = np.array(self.theta1 < 0)
        self.s_old = np.array(self.N*[-1])
        self.net = net
        self.neighbors =  [list(self.net.neighbors(node)) for node in net ]
        self.X = [0]
    def step(self, run_num):
        for run in range(run_num):
            s_new = self.s.copy()
            failed = list(np.where(self.s == 1)[0])
            healthy = list(np.where(self.s == 0)[0])
            s_new[healthy] = [self.phi1()[healthy] > self.theta1[healthy]]
            s_new[failed]  = [self.phi2()[failed]  < self.theta2[failed]]
            
            if np.array_equal(s_new, self.s) or np.array_equal(s_new, self.s_old): #converged or cyclic
                self.s_old = self.s
                self.s = s_new
                self.X.append(np.mean(self.s))
                break
                
            self.s_old = self.s
            self.s = s_new
            self.X.append(np.mean(self.s))
    def phi1_in(self):
        loc  = np.array([  np.mean(self.s[neigh]) for neigh  in self.neighbors]) 
        glob =   np.mean(self.s)
        return self.alpha*loc + (1-self.alpha)*glob
    def phi2_in(self):
        loc  = np.array([1-np.mean(self.s[neigh]) for neigh  in self.neighbors]) 
        glob = 1-np.mean(self.s)
        return self.alpha*loc + (1-self.alpha)*glob
    def phi1_out(self):
        load = np.array([    self.s[i]/len(neigh) for i,neigh in enumerate(self.neighbors)])
        loc  = np.array([np.sum(load[neigh]) for neigh in self.neighbors])
        glob =   np.mean(self.s)
        return self.alpha*loc + (1-self.alpha)*glob
    def phi2_out(self):
        load = np.array([(1-self.s[i])/len(neigh) for i,neigh in enumerate(self.neighbors)])
        loc  = np.array([np.sum(load[neigh]) for neigh in self.neighbors])
        glob = 1-np.mean(self.s)
        return self.alpha*loc + (1-self.alpha)*glob
    def phi1_sup(self):
        seek = np.array([(self.s[i])/(len(neigh)-np.sum(self.s[neigh])) for i,neigh in enumerate(self.neighbors)])
        loc  = np.array([np.sum(seek[neigh]) for neigh in self.neighbors])
        glob = np.sum(self.s)/(self.N-sum(self.s))
        return self.alpha*loc + (1-self.alpha)*glob
    def phi2_sup(self):
        supp = np.array([(1-self.s[i])/np.sum(self.s[neigh]) for i,neigh in enumerate(self.neighbors)])
        loc  = np.array([np.sum(supp[neigh]) for neigh in self.neighbors])
        glob = (self.N-sum(self.s))/ np.sum(self.s)
        return self.alpha*loc + (1-self.alpha)*glob
    
class ABM_MF(object):
    def __init__(self,N, par,  phi1 = "in", phi2 = "in",alpha = 1):
        mu1, sig1, mu2, sig2 = par
        self.N = N
        if phi1 == "in":
            self.phi1 = self.phi1_in
        if phi1 == "out":
            self.phi1 = self.phi1_out
        if phi1 == "red":
            self.phi1 = self.phi1_red
        if phi2 == "in":
            self.phi2 = self.phi2_in
        if phi2 == "out":
            self.phi2 = self.phi2_out
        if phi2 == "red":
            self.phi2 = self.phi2_red
        self.theta1 = np.random.normal(mu1,sig1,self.N)
        self.theta2 = np.random.normal(mu2,sig2,self.N)
        self.s = np.array(self.theta1<0)
        self.s_old = np.array(self.N*[-1])
        self.X = [0]
    def step(self, run_num):
        for run in range(run_num):
            s_new = self.s.copy()
            failed = list(np.where(self.s == 1)[0])
            healthy = list(np.where(self.s == 0)[0])
            s_new[healthy] = [self.phi1()[healthy] > self.theta1[healthy]]
            s_new[failed]  = [self.phi2()[failed]  < self.theta2[failed]]
            
            if np.array_equal(s_new, self.s) or np.array_equal(s_new, self.s_old): #converged or cyclic
                if run%2 ==1:
                    pass
                else:
                    self.s_old = self.s
                    self.s = s_new
                    self.X.append(np.mean(self.s))
                    break
                
            self.s_old = self.s
            self.s = s_new
            self.X.append(np.mean(self.s))
    def phi1_in(self):
        return np.array(self.N*[  np.mean(self.s)]) 
    def phi2_in(self):
        return np.array(self.N*[1-np.mean(self.s)]) 
    def phi1_out(self):
        return np.array(self.N*[  np.mean(self.s)])
    def phi2_out(self):
        return np.array(self.N*[1-np.mean(self.s)]) 
    def phi1_red(self):
        return np.array(self.N*[np.sum(self.s)/(self.N-sum(self.s))])
    def phi2_red(self):
        return np.array(self.N*[(self.N-sum(self.s))/ np.sum(self.s)])