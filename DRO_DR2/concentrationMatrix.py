
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy import integrate
import matplotlib.pyplot as plt


def concentration_matrix(tissue, CBF, Ktrans, aiffull):
    # --------------Default Parameters--------------------#
    sus_factor = 2.7 * 10 ** -8
    rp = 50
    re = 90
    fac1 = 1
    Hct_small = 0

    dose1 = 0  # when pp =1
    dose2 = 1
    # ------------------------------------------------#

    vi = tissue.get_ratio(1)
    vp = tissue.get_ratio(2)
    CBV = vp * 100
    ve = np.array([1 - vi - vp])
    ps = np.array([100 * Ktrans/ fac1])

    def pre(Npts,CB,CV,dose):
        cs0 = np.zeros(2)
        dt = 1
        t = np.arange(0, Npts + 1)
        size = t.size
        tspan = np.arange(0, size)

        to = 60
        toS = int(to / dt)

        cp3 = np.zeros(size)
        cp4 = np.zeros(size)

        r = 2
        b = 2
        co = 1

        for i in range(toS, size - 1):
            cp3[i] = co * ((tspan[i] - to) ** r) * np.exp(-(tspan[i] - to) / b)

        tor = 2
        rc = np.zeros(size)
        taur = 625

        for i in range(0, size - 1):
            rc[i] = (1 / taur) * np.exp(-(tspan[i] - tor) / taur)

        cpR = dt * np.convolve(cp3, rc)

        cp4[toS - 1:size - 1] = cp3[toS - 1:size - 1] + 8 * cpR[toS - 1 - tor:size - 1 - tor]

        aiffull_pre = dose * aiffull[0:size]

        CBF_unit = CBF / 60
        CBV_unit = CBV
        vp = CBV / 100 * (1 - Hct_small) * fac1
        ftm = CBF_unit * (1 - Hct_small) * fac1
        vptm = vp
        vetm = ve

        CBVtm = CBV_unit

        A = np.zeros((ps.size, t.size))
        B = np.zeros((ps.size, t.size))
        for ps_index in range(0, ps.size):
            pstm = ps[ps_index] / 60

            Cs = two_model_ODE(tspan, cs0, aiffull_pre, tspan, ftm, pstm, CBVtm, vetm, vptm)
            Cp = Cs[0]
            VpCp = Cp*vp

            # print(Cp)
            Ce = Cs[1]
            VeCe = Ce*ve
            A[ps_index] = VeCe
            B[ps_index] = VpCp
        return A,B


        # return cpR

    Nspt = 180

    (A,C) = pre(540, 0, 0, dose1)
    A = A[:,360:]
    C = C[:,360:]
    (D,E) = pre(180, 0, 0, dose2)

    A_tot = A+D
    C_tot = C+E
    G = abs(C_tot/vp - A_tot/vp)
    G1 = (C_tot/vp-A_tot/ve)

    return np.array([C_tot, C_tot/vp,C_tot/vp*sus_factor, A_tot,A_tot/ve,A_tot/ve*sus_factor])


def two_model_ODE(t_arr, cs0, aiffull, tspan, ftm, pstm, cbvtm, vetm, vptm):
    arterial_Input_Function = interp1d(tspan, aiffull)(t_arr)
    dCs = lambda t, cs: [(ftm / cbvtm) * (arterial_Input_Function[int(t)] - cs[0]) - (pstm / cbvtm) * (cs[0] - cs[1]),
                         (pstm / cbvtm) * (vptm / vetm) * (cs[0] - cs[1])]

    sol = solve_ivp(dCs, [t_arr[0], t_arr[- 1]], cs0, t_eval=t_arr)

    return sol.y

