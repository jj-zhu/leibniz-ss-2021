'''
code by: j.zhu
minmax problem. but with obj estimated by KME
'''
import sys

# %% imports
# from gurobipy import * # boot gurobi, use casadi
import matplotlib.pyplot as plt
from casadi import *

# use sklearn for kernel/gram matrix computation
sys.path.append("../..")

# my python tools
from src.socp import scenario_mpc
import argparse

import pickle


# %% functions
def cost_expected_simple(X, n_sample):
    sum_i = 0
    for i in range(n_sample):
        try:
            sum_i += np.sum((X[i, :, 0] - 3) ** 2)
        except:
            assert False
    expected_cost = sum_i / n_sample
    return expected_cost


# %% args
parser = argparse.ArgumentParser()
parser.add_argument('--reg_coef', default=0.005, help='coefficient for lasso regularization strength')
parser.add_argument('--n_sample', default=20, help='number of scenarios')
parser.add_argument('--n_test', default=20, help='number of scenarios')
parser.add_argument('--n_discard', default=3, help='num samples to discard')
parser.add_argument('--deg_pk', default=4, help='degree of the polynomial kernel')
parser.add_argument('--opt_type', default='lasso', help='use lasso or mio. or none?')
parser.add_argument('--kernel_type', default='rbf', help='choose from p2, rbf, laplace')
parser.add_argument('--kernel_width', default=10.0)
parser.add_argument('--fixed_seed', default=True, help='is random seed for data fixed?')
parser.add_argument('--n_horizon', default=10, help='num control intervals. aka horizon in dt')
parser.add_argument('--t_horizon', default=1.0, help='time horizon')
parser.add_argument('--temperature_lasso', default=0.1)
parser.add_argument('--save_dir', default='fig/')

# %% run
if __name__ == '__main__':
    try:
        plt.style.use('jz')  # use my special style
    except:
        pass

    args = parser.parse_args()

    reg_coef = float(args.reg_coef)  # sparse coeff
    opt_type = args.opt_type
    n_discard = int(args.n_discard)  # number of samples to discard
    deg_pol = int(args.deg_pk)
    kernel_type = args.kernel_type
    kernel_width = float(args.kernel_width)
    fixed_seed = args.fixed_seed
    n_sample = int(args.n_sample)  # how many samples do we generate
    n_test = int(args.n_test)  # how many samples do we generate
    temperature_lasso = float(args.temperature_lasso)
    T = float(args.t_horizon)  # control horizon [s]
    N = int(args.n_horizon)  # Number of control intervals
    save_dir = args.save_dir

    # reduced scenarios
    n_reduced = n_sample - n_discard

    # for saving data
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # %% constants for mpc
    dt = T / N  # length of 1 control interval [s]
    tgrid = np.linspace(0, T, N + 1)
    nx = 2

    if fixed_seed:
        np.random.seed(0)

    # %% solve full opt problem
    # %% draw inital scenarios
    cov0 = diag([0.01 ** 2, 0.1 ** 2])
    mean0 = [0.5, 0]
    # random ranly draw scenarios
    x_sample_tmp = np.random.multivariate_normal(mean0, cov0, n_sample).T
    x0_sample = vertcat(x_sample_tmp)

    # %% smpc
    X, U, sol, _ = scenario_mpc(
        x0_sample, n_sample, N, dt, nx, tgrid,
        goal_stabilize=3.0,
        is_plot=True, is_return_all=True,
        name_savefig=save_dir + 'original'
    )

    # %% we now create a second bag of trajectories, stabilizing at different level

    # cov0_2 = diag([0.02 ** 2, 0.2 ** 2])
    # mean0_2 = [0.5, 0]  # we match the mean but not var
    x_sample_tmp_2 = np.random.multivariate_normal(mean0, cov0, n_sample).T
    x0_sample_2 = vertcat(x_sample_tmp)

    # %% smpc;
    X_2, U_2, sol_2, _ = scenario_mpc(
        x0_sample_2, n_sample, N, dt, nx, tgrid,
        goal_stabilize=1.8,
        is_plot=True, is_return_all=True,
        name_savefig=save_dir + 'original'
    )

    # %% save data
    with open('data1.pk', 'wb') as f:
        pickle.dump(X, f)
    with open('data2.pk', 'wb') as f:
        pickle.dump(X_2, f)
