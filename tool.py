import numpy as np


# 获取数量级Orders of magnitude
def get_oofm(y):
    return np.floor(np.log10(y))


def weight_ode():
    loss = np.loadtxt("loss.dat")[1, 0:17]
    loss_oofm = get_oofm(loss)
    loss_weight = np.divide(1.0, 10.0 ** loss_oofm)
    return loss_weight.tolist()


def weight_bc(y):
    bc = np.amax(y, axis=0)
    bc_oofm = get_oofm(bc)
    loss_weight = np.divide(1.0, 10.0 ** bc_oofm)
    return loss_weight.tolist()


def weight_data():
    loss = np.loadtxt("data/loss.dat")[1, 34:]
    loss_oofm = get_oofm(loss)
    loss_weight = np.divide(1.0, 10.0 ** loss_oofm)
    return loss_weight.tolist()


def weight_k():
    k = np.loadtxt("config/k.dat")
    k_oofm = get_oofm(k)
    k_weight = np.power(10.0, k_oofm)
    return k_weight


def get_ci():
    ci = np.loadtxt("config/confidence_interval.dat")
    return ci

