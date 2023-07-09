import numpy as np
import control 
import cvxpy as cp
from scipy import interpolate
import yaml

configuration = yaml.load(open('../configuration.yml', 'r'), Loader=yaml.Loader)

Q = 0.2 * np.diag(np.ones(configuration['rx'],))
Q[2,2] = 0.1
R = 0.1 * np.diag(np.ones(configuration['ry'],))
R[2,2] = 0.1

def stateDynamics(x, u, w):
    x = x.squeeze()
    w = w.squeeze()
    u = u.squeeze()
    f = np.zeros((rx,))
    #configuration['tau'] = 0.2
    f[0] = x[0] + configuration['tau'] * configuration['V'] * np.sinc(u * configuration['tau'] / 2) * np.cos(x[2] + u * configuration['tau'] / 2)
    f[1] = x[1] + configuration['tau'] * configuration['V'] * np.sinc(u * configuration['tau'] / 2) * np.sin(x[2] + u * configuration['tau'] / 2)
    f[2] = x[2] + configuration['tau'] * u
    f = f + w
    f[2] = np.arctan2(np.sin(f[2]), np.cos(f[2]))
    return np.atleast_2d(f.squeeze()).T

def measurementDynamics(x, u):
    x = x.squeeze()
    u = u.squeeze()
    gx = np.zeros((ry, 1))
    gx = x[0:ry]
    return np.atleast_2d(gx.squeeze()).T

r=10.0
# Obstacles params
xs = np.array([8, -3, -10])
ys = np.array([-5, -9, 10])
rs = np.array([2, 2, 3])

# Load control data for interpolation
# Run DP_ValueIteration_MCGrid.py to generate the lookup table x,y,u
xx= np.load('Utilities/ControllerData/KlookuptableX.npy')
yy= np.load('Utilities/ControllerData/KlookuptableY.npy')
UU= np.load('Utilities/ControllerData/KlookuptableU.npy')
K_LT = interpolate.interp2d(xx, yy, UU, kind='linear') 

def controller(state): #Here you define your controller, whether an MPC, SMPC, CBF, PID, whatever...
    theta_star=K_LT(state[0].reshape(1,),state[1].reshape(1,))
    theta = state[2]
    theta_error = theta_star-theta
    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    angularVel = np.clip(5 * (theta_error), -np.pi, np.pi)
    return angularVel

def thetaError(state): #Here you define your controller, whether an MPC, SMPC, CBF, PID, whatever...
    theta_star=K_LT(state[0].reshape(1,),state[1].reshape(1,))
    theta = state[2]
    theta_error = theta_star-theta
    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    return theta_error


def Controller(state): #Here you define your controller, whether an MPC, SMPC, CBF, PID, whatever...
    x = state[0]
    y = state[1]
    theta = state[2]
    d = np.sqrt(x**2 + y**2)
    gamma = np.arctan2(y, x)
    theta_d = gamma - np.pi/2 - np.arctan(0.3 * (d-r))
    Fx = np.cos(theta_d) 
    Fy = np.sin(theta_d) 
    u = cp.Variable((2,))
    objective = cp.Minimize(cp.quad_form(u - np.array([Fx, Fy]).squeeze(), np.diag(np.ones(2,))))
    Del_h_mat = np.full((3,2), 0.0)
    alpha_h_vec = np.full((3,), 0.0)
    for nn in range(3):
        alpha_h_vec[nn] = (x-xs[nn]) ** 2 + (y-ys[nn]) ** 2 - rs[nn] ** 2
        Del_h_mat[nn,:] = 2 * np.array([x-xs[nn], y-ys[nn]]).squeeze()
    constraints = [Del_h_mat @ u >= -0.05*alpha_h_vec]
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    uStar = u.value
    theta_star = np.arctan2(uStar[1],uStar[0])
    theta_error = theta_star-theta
    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    angularVel = np.clip(20 * (theta_error), -np.pi, np.pi)
    return angularVel


def CostAndConstraints(Control_seq, xk2prime):
        d = np.sqrt( xk2prime[0,:] ** 2 + xk2prime[1,:] ** 2)
        gamma = np.arctan2(xk2prime[1,:], xk2prime[0,:])
        cost_d = ( (d-r) ** 2 ).sum()
        theta_d = gamma - np.pi/2 - np.arctan(0.3 * (d-r))
        error = xk2prime[2,:] * 0
        for j in range(len(xk2prime[2,:])):
             error[j] = thetaError(xk2prime[:,j])
        cost_theta = (error ** 2).sum()
        cost = cost_d * 0.0 + cost_theta * 1 
        # Bounds on the control
        #Control_violations = abs(Control_seq) > 5
        #Control_violations = np.append(Control_violations, False)
        # Linear state constraints violation: Hx > b
        xx = xk2prime[0,:].squeeze()
        yy = xk2prime[1,:].squeeze()
        State_violations = 0
        for nn in range(3):
            h_value = (xx-xs[nn]) ** 2 + (yy-ys[nn]) ** 2 - rs[nn] ** 2
            State_violations += (h_value < 0).sum()
        #number_of_violations = (State_violations.squeeze() | Control_violations.squeeze()).sum()
        number_of_violations = State_violations
        return cost, number_of_violations

