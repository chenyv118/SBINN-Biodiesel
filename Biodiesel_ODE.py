from scipy.integrate import odeint
from tmp2.config.data_config import *
from tmp2.config.k_config import *

K = [Vreactf, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, Fa, CoAlco, af, Ae]


def biodiesel_model(t, k=None):
    if k is None:
        k = K
    CoAlco = k[22]
    af = k[23]
    Ae = k[24]
    Vreactf = k[0]

    def ode_func(x, t):
        T, D, M, BD, FFA, G, W, CH, E, EX, ET, ED, EM, ECH, Ef, Vp, V = x
        if t <= 120:
            Fa = k[21][0]
        else:
            Fa = k[21][1]
        if x[16] >= Vreactf:
            Fa = 0

        aT = af * (Vp / V)
        Af = (aT - Ae * (E + EX + ET + ED + EM + ECH)) / Ae

        r1 = k[1] * Ef * Af - k[11] * E
        r2 = k[2] * T * E - k[12] * ET
        r3 = k[3] * ET - k[13] * EX * D
        r4 = k[4] * D * E - k[14] * ED
        r5 = k[5] * ED - k[15] * EX * M
        r6 = k[6] * M * E - k[16] * EM
        r7 = k[7] * EM - k[17] * EX * G
        r8 = k[8] * EX * W - k[18] * FFA * E
        r9 = k[9] * EX * CH - k[19] * BD * E
        r10 = k[10] * CH * E - k[20] * ECH

        rg = (r7 * V * 92 / 1261)
        rw = (-r8 * V * 18 / 1000)

        return [
            -V * r2,
            V * (r3 - r4),
            V * (r5 - r6),
            V * r9,
            V * r8,
            V * r7,
            -V * r8,
            -V * (r9 + r10),
            V * (r1 - r2 - r4 - r6 + r8 + r9 - r10),
            V * (r3 + r5 + r7 - r8 - r9),
            V * (r2 - r3),
            V * (r4 - r5),
            V * (r6 - r7),
            V * r10,
            -V * r1,
            rg + rw,
            Fa
        ]

    # Experimental data in [mole/L]
    st = 0
    tmp = data[32][st:]
    tmp[0] = 1e-4

    FAME = tmp
    FFA = data[33][st:]
    TAG = data[34][st:]
    DAG = data[35][st:]
    MAG = data[36][st:]

    # Initial Condition
    T = TAG[0]
    D = DAG[0]
    M = MAG[0]
    B = FAME[0]
    FA = FFA[0]
    G = 1e-6
    W = mH2O
    CH = Alco
    E = 0.0
    EX = 0.0
    ET = 0.0
    ED = 0.0
    EM = 0.0
    ECH = 0.0
    Ef = Enzyme
    Vp = Vpo
    V = Vo
    x0 = [T, D, M, B, FA, G, W, CH, E, EX, ET, ED, EM, ECH, Ef, Vp, V]

    return odeint(ode_func, x0, t)


def main():
    t = np.arange(0, 1500, 1)[:, None]

    # y, info = biodiesel_model(np.ravel(t))
    y = biodiesel_model(np.ravel(t))
    np.savetxt("biodiesel.dat", np.hstack((t, y)))


if __name__ == "__main__":
    main()
