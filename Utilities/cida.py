import numpy as np
from Utilities.Dynamics_Constraints_Controller import *
from scipy.linalg import sqrtm
from Utilities.ParticleFilter import ParticleFilter
import random
from collections import deque
configuration = yaml.load(open('./configuration.yml', 'r'), Loader=yaml.Loader)
rx = configuration['rx']
T = configuration['T']
num_particles = configuration['num_particles']
Pred_Horizon_N = configuration['Pred_Horizon_N']
number_of_simulations = configuration['number_of_simulations']
LangrangeMultp = configuration['LangrangeMultp']
ru = configuration['ru']
ry = configuration['ry']
V = configuration['V']
tau = configuration['tau']
alpha = configuration['alpha']
class CIDA(ParticleFilter):
    def __init__(self, x0, Cov0, num_particles, stateDynamics, measurementDynamics, Q, R,
                 Pred_Horizon_N, Controller, M, CostAndConstraints, LangrangeMultp):
        super().__init__(x0, Cov0, num_particles, stateDynamics, measurementDynamics, Q, R)
        self.Pred_Horizon_N = Pred_Horizon_N
        self.Controller = Controller
        self.LangrangeMultp = LangrangeMultp
        self.M = M
        self.CostAndConstraints = CostAndConstraints


    def sample_xk_prime(self, x0prime): #Generating state sequence x_k' for k=0,...,N-1
        xkprime=np.full((rx, self.Pred_Horizon_N), np.nan)
        Control_seq = np.full((ru, self.Pred_Horizon_N), np.nan)
        Wprime=sqrtm(self.Q).real @ np.random.randn(rx, self.Pred_Horizon_N)
        xkprime[:,0]=x0prime.squeeze()
        for k in range(self.Pred_Horizon_N-1):
            Control_seq[:,k] = self.Controller(xkprime[:,k]).squeeze()
            xkprime[:,k+1]=stateDynamics(xkprime[:,k], Control_seq[:,k], Wprime[:,k]).squeeze()
        Control_seq[:,k+1] = self.Controller(xkprime[:,k+1]).squeeze()
        return xkprime, Control_seq

    def sample_xk_dblPrime(self, Control_seq): #Generating state sequence x_k'' for k=0,...,N-1
        xk2prime = np.full((rx, self.Pred_Horizon_N+1), np.nan)
        W2prime = sqrtm(self.Q).real @ np.random.randn(rx, self.Pred_Horizon_N)
        x02prime = self.particles[:,random.sample(range(0, self.num_particles), 1)]
        xk2prime[:,0]=x02prime.reshape(rx,)
        for k in range(self.Pred_Horizon_N):
            u = Control_seq[:,k]
            xk2prime[:,k+1]=stateDynamics(xk2prime[:,k],u,W2prime[:,k]).squeeze()  
        return xk2prime

    # Random Search of CIDA
    def RandomSearch(self):
        ControlSeqCost = np.zeros((self.M,))
        ControlSeqRec = deque([])
        for i in range(self.M):
            x0prime=self.particles[:,random.sample(range(0, self.num_particles), 1)]
            _, Control_seq = self.sample_xk_prime(x0prime)
            ControlSeqRec.append(Control_seq)
            ratio_of_violations = np.zeros((self.Pred_Horizon_N+1,))
            for q in range(self.M):
                xk2prime = self.sample_xk_dblPrime(Control_seq)
                cost, violation_flag = CostAndConstraints(Control_seq, xk2prime)
                ratio_of_violations += violation_flag / self.M
                ControlSeqCost[i] += cost / self.Pred_Horizon_N
            if all(ratio_of_violations<=alpha):    
                ControlSeqCost[i] = ControlSeqCost[i] / self.M
            else:
                ControlSeqCost[i] = 10**6 + 10*ratio_of_violations.max()
                print('soft constraints were used.')
        minCost_index = ControlSeqCost.argmin()
        BestControlSequence = ControlSeqRec[minCost_index]
        return BestControlSequence[:,0]

    def ViolationProb(self): #calculates violation rates
        _, number_of_violations = CostAndConstraints(0.0, self.particles)
        ViolationRate = number_of_violations.sum() / self.num_particles
        return ViolationRate


