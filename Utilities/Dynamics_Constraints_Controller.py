import numpy as np
import control 
import cvxpy as cp

rx = 3
ru = 1
ry = 2
Q = 0.1 * np.diag(np.ones(rx,))
R = 0.2 * np.diag(np.ones(ry,))
V = 10
def stateDynamics(x, u, w):
    x = x.squeeze()
    w = w.squeeze()
    u = u.squeeze()
    f = np.zeros((rx,))
    tau = 0.1
    f[0] = x[0] + tau * V * np.sin(u * tau / 2) / (u * tau / 2) * np.cos(x[2] + u * tau / 2)
    f[1] = x[1] + tau * V * np.sin(u * tau / 2) / (u * tau / 2) * np.sin(x[2] + u * tau / 2)
    f[2] = x[2] + tau * u
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
xs = np.array([5, -3, -9])
ys = np.array([5, -9, 9])
rs = np.array([3, 2, 3])

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
    angularVel = np.clip(10 * (theta_error), -np.pi*2, np.pi*2)
    return angularVel


def CostAndConstraints(Control_seq,xk2prime):
        d = np.sqrt( xk2prime[0,:] ** 2 + xk2prime[1,:] ** 2)
        gamma = np.arctan2(xk2prime[1,:], xk2prime[0,:])
        cost_d = ( (d-r) ** 2).sum()
        theta_d = gamma - np.pi/2 - np.arctan(0.3 * (d-r))
        error = theta_d - xk2prime[2,:]
        error = np.arctan2(np.sin(error), np.cos(error))
        cost_theta = (error ** 2).sum()
        cost = cost_d * 0 + cost_theta * 1
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

