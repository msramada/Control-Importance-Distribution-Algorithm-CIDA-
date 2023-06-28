import numpy as np
from Utilities.Dynamics_Constraints_Controller import *
from scipy.linalg import sqrtm
from Utilities.ParticleFilter import ParticleFilter
import random
from collections import deque

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
        xk2prime = np.full((2, self.Pred_Horizon_N+1), np.nan)
        W2prime = sqrtm(self.Q).real @ np.random.randn(rx, self.Pred_Horizon_N)
        x02prime = self.particles[:,random.sample(range(0, self.num_particles), 1)]
        xk2prime[:,0]=x02prime.reshape(rx,)
        for k in range(self.Pred_Horizon_N):
            u = Control_seq[:,k]
            xk2prime[:,k+1]=stateDynamics(xk2prime[:,k],u,W2prime[:,k]).squeeze()  
        return xk2prime

    # Random Search of CIDA
    def RandomSearch(self):
        ControlSeqCost = np.zeros((self.num_particles,))
        ControlSeqRec = deque([])
        for i in range(self.num_particles):
            x0prime=self.particles[:,random.sample(range(0, self.num_particles), 1)]
            _, Control_seq = self.sample_xk_prime(x0prime)
            ControlSeqRec.append(Control_seq)
            for q in range(self.M):
                xk2prime = self.sample_xk_dblPrime(Control_seq)
                cost, number_of_violations = CostAndConstraints(Control_seq, xk2prime)
                ControlSeqCost[i] += (cost + 
                    self.LangrangeMultp * number_of_violations) / self.Pred_Horizon_N
                
            ControlSeqCost[i] = ControlSeqCost[i] / self.M
        minCost_index = ControlSeqCost.argmin()
        BestControlSequence = ControlSeqRec[minCost_index]
        return BestControlSequence

    def ViolationProb(self): #calculates violation rates
        _, number_of_violations = CostAndConstraints(0, self.particles)
        ViolationRate = number_of_violations / self.num_particles
        return ViolationRate


