'''
code by: j.zhu
# solve stochastic OCP
'''
import argparse
import os
from pylab import *
from casadi import *

# func def
def scenario_mpc(x0_sample, n_sample, N, dt, nx, tgrid, goal_stabilize=3.0, is_plot=True, name_savefig=None, is_return_all=True, is_test=False, u_input=None):
    ##
    # ----------------------------------
    #    continuous system dot(x)=f(x,u)
    # ----------------------------------
    # Construct a CasADi function for the ODE right-hand side
    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    if is_test:
        U = u_input
    u = MX.sym('u')  # control
    rhs = vertcat(x2, -0.1 * (1 - x1 ** 2) * x2 - x1 + u)
    x = vertcat(x1, x2)

    x1_bound = lambda t: 2 + 0.1 * cos(10 * t)
    # variance evolution integrator
    dfdx = jacobian(rhs, x)

    ##
    # -----------------------------------
    #    Discrete system x_next = F(x,u)
    # -----------------------------------
    opts = dict()
    opts["tf"] = dt
    intg = integrator('intg', 'cvodes', {'x': x, 'p': u, 'ode': rhs}, opts)

    ##
    # -----------------------------------------------
    #    Optimal control problem, multiple shooting
    # -----------------------------------------------

    opti = casadi.Opti()

    # Decision variable for states
    x = opti.variable(nx, n_sample)  # jz second indexes the scenarios; hence every state xt is now a matrix
    # P = opti.variable(nx, nx)
    # opti.set_initial(P, eye(2))

    # Initial constraints
    # for i in range(n_sample):
    opti.subject_to(x == x0_sample)
    # opti.subject_to(P == s0)

    # collection of all dec var
    if not is_test:
        U = []
    X = [x]
    # Ps = [P]
    # Gap-closing shooting constraints
    margin_all = [] # jz: constr violation margin
    for k in range(N):
        if not is_test:
            u = opti.variable()
            U.append(u)
        else:
            u = U[k]

        # joris: create var on the go, so no need to fudge with ordering
        x_next = opti.variable(nx, n_sample)  # next state, matrix to account for all scenarios
        # P_next = opti.variable(nx, nx)
        # opti.set_initial(P_next, eye(2))

        # iterate thru scenarios
        res = []
        xf = []
        for i in range(n_sample):
            res_i = intg(x0=x[:, i], p=u)  # integrator needs to integrate all samples
            res.append(res_i)

            # unpack the integ sol to state, cov
            xf_i = res_i['xf'][:nx]
            xf.append(xf_i)
            # Pf = reshape(res['xf'][nx:], nx, nx)

            opti.subject_to(x_next[:, i] == xf_i)
        # opti.subject_to(P_next == Pf)

        # shared control var
        if not is_test:
            opti.subject_to(opti.bounded(-40, u, 40))

        # %% design back off. 1 std away
        # var_t = horzcat(1, 0) @ P @ vertcat(1, 0)
        # backoff_t = sqrt(var_t)

        # state constr
        # no need for back off for scenario approach
        margin = []
        for i in range(n_sample):
            if not is_test:
                opti.subject_to(opti.bounded(-0.25, x[0, i],
                                         x1_bound(tgrid[k])))

            margin = vertcat(margin, x1_bound(tgrid[k]) - x[0, i])

        x = x_next
        # P = P_next

        X.append(x)
        margin_all.append(margin)
        # Ps.append(P)

    # term cons
    # var_t = horzcat(1, 0) @ P @ vertcat(1, 0)
    # backoff_t = sqrt(var_t)
    margin = []
    for i in range(n_sample):
        if not is_test:
            opti.subject_to(opti.bounded(-0.25, x_next[0, i], x1_bound(tgrid[N])))
        margin = vertcat(margin, x1_bound(tgrid[N]) - x_next[0, i])
    margin_all.append(margin)

    U = hcat(U)

    #%% prep for expected cost
    sum_i = 0
    for i in range(n_sample):
        try:
            sum_i += sumsqr(vertcat(*[X[j][0, i] for j in range(N + 1)]) - goal_stabilize)
        except:
            assert False
    expected_cost = sum_i / n_sample

    # cost_function = Function('cost_function', [X], [expected_cost])

    opti.minimize(expected_cost)

    opti.solver('ipopt')

    sol = opti.solve()

    if is_plot:
        figure()
        # Simulate forward in time using an initial state and control vector
        usol = sol.value(U)
        xsol = []
        for i in range(n_sample):
            xsol_i = np.squeeze([sol.value(X[j][:, i]) for j in range(N + 1)])
            xsol.append(xsol_i)

            # print(xsol.shape)

            plot(tgrid, xsol_i[:, 0].T, 'b.-', alpha=0.3)
        plot(tgrid, x1_bound(tgrid), 'r--')

        # for k in range(N + 1):
        #     # var = sol.value(horzcat(1, 0) @ Ps[k] @ vertcat(1, 0))
        #     sigma = sqrt(var)
        #     t = tgrid[k]
        #     plot([t, t], [xsol[0, k] - sigma, xsol[0, k] + sigma], 'k', 'linewidth', 2)

        # legend()
        xlabel('Time [s]')
        ylabel('x1')
        tight_layout()
        if name_savefig is not None:
            savefig(name_savefig+'_states.pdf')

        figure()
        print(usol.shape)
        step(tgrid, vertcat(usol, usol[-1]))
        title('applied control signal')
        ylabel('Force [N]')
        xlabel('Time [s]')
        tight_layout()
        if name_savefig is not None:
            savefig(name_savefig+'_ctrl.pdf')
        else:
            show()

    if is_return_all:
        x_return = []
        for i in range(n_sample):
            x_return.append(sol.value(horzcat(*[X[j][:, i] for j in range(N + 1)]).T)
                            )

        return np.asarray(x_return), sol.value(U), sol, sol.value(horzcat(*margin_all))

    return  sol # only return sol

if __name__ == '__main__':
    # %% stoprog param
    n_sample = 3  # how many samples do we generate

    #%% SMPC part
    #%% draw inital scenarios
    cov0 = diag([0.01 ** 2, 0.1 ** 2])
    mean0 = [0.5, 0]
    # random ranly draw scenarios
    x_sample_tmp = np.random.multivariate_normal(mean0, cov0, n_sample).T
    x0_sample = vertcat(x_sample_tmp)

    #%% constants for mpc
    T = 1.0  # control horizon [s]
    N = 40  # Number of control intervals

    dt = T / N  # length of 1 control interval [s]


    tgrid = np.linspace(0, T, N + 1)
    nx = 2
    #%% smpc


    sol = scenario_mpc(x0_sample, n_sample, T, N, dt, nx, is_plot=True)
    # sparsity plot
    # figure()
    # spy(sol.value(jacobian(opti.g, opti.x)))