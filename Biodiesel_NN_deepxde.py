import torch
from torch.nn.functional import tanh
import deepxde as dde
# from Biodiesel_ODE import biodiesel_model
from config.data_config import *
from tool import *

data_set = np.loadtxt("./data/biodiesel.dat")
t = data_set[:, 0].reshape(1500, -1)
y = data_set[:, 1:]
data_t = t
data_y = y
noise = 0.1
# Add noise
if noise > 0:
    std = noise * y.std(0)
    y[1:-1, :] += np.random.normal(0, std, (y.shape[0] - 2, y.shape[1]))
    np.savetxt("./data/biodiesel_noise.dat", np.hstack((t, y)))

# Init Parameter
k1_ = dde.Variable(0.01)
k2_ = dde.Variable(0.01)
k3_ = dde.Variable(0.01)
k4_ = dde.Variable(0.01)
k5_ = dde.Variable(0.01)
k6_ = dde.Variable(0.01)
k7_ = dde.Variable(0.01)
k8_ = dde.Variable(0.01)
k9_ = dde.Variable(0.01)
k10_ = dde.Variable(0.01)
k_1_ = dde.Variable(0.01)
k_2_ = dde.Variable(0.01)
k_3_ = dde.Variable(0.01)
k_4_ = dde.Variable(0.01)
k_5_ = dde.Variable(0.01)
k_6_ = dde.Variable(0.01)
k_7_ = dde.Variable(0.01)
k_8_ = dde.Variable(0.01)
k_9_ = dde.Variable(0.01)
k_10_ = dde.Variable(0.01)
var_list = [k1_, k2_, k3_, k4_, k5_, k6_, k7_, k8_, k9_, k10_, k_1_, k_2_, k_3_, k_4_, k_5_, k_6_, k_7_, k_8_, k_9_,
            k_10_]
k_weight = weight_k()


def ODE(t, y):
    row, col = y.shape
    T, D, M, BD, FFA, G, W, CH, E, EX, ET, ED, EM, ECH, Ef, Vp, V = [y[:, i:i + 1] for i in range(col)]

    Fa_ = torch.tensor(Fa)[(t > 120).type(torch.long)]
    Fa_ = Fa_ * ((V < Vreactf).type(torch.long).reshape(len(V), -1))
    # Confidence Interval
    ci = get_ci()

    def get_variable(variable, idx):
        l, r = ci[idx]
        m = (l + r) / 2
        res = m + tanh(variable) * (r - l) / 2
        return res

    k1 = get_variable(k1_, 0) * k_weight[0]
    k2 = get_variable(k2_, 1) * k_weight[1]
    k3 = get_variable(k3_, 2) * k_weight[2]
    k4 = get_variable(k4_, 3) * k_weight[3]
    k5 = get_variable(k5_, 4) * k_weight[4]
    k6 = get_variable(k6_, 5) * k_weight[5]
    k7 = get_variable(k7_, 6) * k_weight[6]
    k8 = get_variable(k8_, 7) * k_weight[7]
    k9 = get_variable(k9_, 8) * k_weight[8]
    k10 = get_variable(k10_, 9) * k_weight[9]
    k_1 = get_variable(k_1_, 10) * k_weight[10]
    k_2 = get_variable(k_2_, 11) * k_weight[11]
    k_3 = get_variable(k_3_, 12) * k_weight[12]
    k_4 = get_variable(k_4_, 13) * k_weight[13]
    k_5 = get_variable(k_5_, 14) * k_weight[14]
    k_6 = get_variable(k_6_, 15) * k_weight[15]
    k_7 = get_variable(k_7_, 16) * k_weight[16]
    k_8 = get_variable(k_8_, 17) * k_weight[17]
    k_9 = get_variable(k_9_, 18) * k_weight[18]
    k_10 = get_variable(k_10_, 19) * k_weight[19]

    aT = af * (Vp / torch.maximum(V, torch.as_tensor(1e-3)))
    Af = (aT - Ae * (E + EX + ET + ED + EM + ECH)) / Ae

    r1 = k1 * Ef * Af - k_1 * E
    r2 = k2 * T * E - k_2 * ET
    r3 = k3 * ET - k_3 * EX * D
    r4 = k4 * D * E - k_4 * ED
    r5 = k5 * ED - k_5 * EX * M
    r6 = k6 * M * E - k_6 * EM
    r7 = k7 * EM - k_7 * EX * G
    r8 = k8 * EX * W - k_8 * FFA * E
    r9 = k9 * EX * CH - k_9 * BD * E
    r10 = k10 * CH * E - k_10 * ECH

    rg = (r7 * V * 92 / 1261)
    rw = (-r8 * V * 18 / 1000)

    return [
        dde.grad.jacobian(y, t, i=0, j=0) - (-V * r2),
        dde.grad.jacobian(y, t, i=1, j=0) - (V * (r3 - r4)),
        dde.grad.jacobian(y, t, i=2, j=0) - (V * (r5 - r6)),
        dde.grad.jacobian(y, t, i=3, j=0) - (V * r9),
        dde.grad.jacobian(y, t, i=4, j=0) - (V * r8),
        dde.grad.jacobian(y, t, i=5, j=0) - (V * r7),
        dde.grad.jacobian(y, t, i=6, j=0) - (-V * r8),
        dde.grad.jacobian(y, t, i=7, j=0) - (-V * (r9 + r10)),
        dde.grad.jacobian(y, t, i=8, j=0) - (V * (r1 - r2 - r4 - r6 + r8 + r9 - r10)),
        dde.grad.jacobian(y, t, i=9, j=0) - (V * (r3 + r5 + r7 - r8 - r9)),
        dde.grad.jacobian(y, t, i=10, j=0) - (V * (r2 - r3)),
        dde.grad.jacobian(y, t, i=11, j=0) - (V * (r4 - r5)),
        dde.grad.jacobian(y, t, i=12, j=0) - (V * (r6 - r7)),
        dde.grad.jacobian(y, t, i=13, j=0) - (V * r10),
        dde.grad.jacobian(y, t, i=14, j=0) - (-V * r1),
        dde.grad.jacobian(y, t, i=15, j=0) - (rg + rw),
        dde.grad.jacobian(y, t, i=16, j=0) - Fa_
    ]


# 展开时域
geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])


def boundary(x, _):  # 边界条件
    return np.isclose(x[0], data_t[-1, 0])  # 比较两个array是不是每一个元素都相等，默认在1e-05的范围误差内 ，返回的是布尔值


y1 = data_y[-1]
# ODE在边界处的解
# bc = [dde.DirichletBC(geom, lambda X: y1[i], boundary, component=i) for i in range(0, 17)]
bc0 = dde.DirichletBC(geom, lambda X: y1[0], boundary, component=0)
bc1 = dde.DirichletBC(geom, lambda X: y1[1], boundary, component=1)
bc2 = dde.DirichletBC(geom, lambda X: y1[2], boundary, component=2)
bc3 = dde.DirichletBC(geom, lambda X: y1[3], boundary, component=3)
bc4 = dde.DirichletBC(geom, lambda X: y1[4], boundary, component=4)
bc5 = dde.DirichletBC(geom, lambda X: y1[5], boundary, component=5)
bc6 = dde.DirichletBC(geom, lambda X: y1[6], boundary, component=6)
bc7 = dde.DirichletBC(geom, lambda X: y1[7], boundary, component=7)
bc8 = dde.DirichletBC(geom, lambda X: y1[8], boundary, component=8)
bc9 = dde.DirichletBC(geom, lambda X: y1[9], boundary, component=9)
bc10 = dde.DirichletBC(geom, lambda X: y1[10], boundary, component=10)
bc11 = dde.DirichletBC(geom, lambda X: y1[11], boundary, component=11)
bc12 = dde.DirichletBC(geom, lambda X: y1[12], boundary, component=12)
bc13 = dde.DirichletBC(geom, lambda X: y1[13], boundary, component=13)
bc14 = dde.DirichletBC(geom, lambda X: y1[14], boundary, component=14)
bc15 = dde.DirichletBC(geom, lambda X: y1[15], boundary, component=15)
bc16 = dde.DirichletBC(geom, lambda X: y1[16], boundary, component=16)

# 随机取1/4的数据
n = len(data_t)
idx = np.append(
    np.random.choice(np.arange(1, n - 1), size=n // 4, replace=False), [0, n - 1]
)
# 可观察的状态变量
# ic = [dde.PointSetBC(data_t[idx], data_y[idx, i:i + 1], component=0) for i in range(5)]
ic0 = dde.PointSetBC(data_t[idx], data_y[idx, 0:1], component=0)
ic1 = dde.PointSetBC(data_t[idx], data_y[idx, 1:2], component=1)
ic2 = dde.PointSetBC(data_t[idx], data_y[idx, 2:3], component=2)
ic3 = dde.PointSetBC(data_t[idx], data_y[idx, 3:4], component=3)
ic4 = dde.PointSetBC(data_t[idx], data_y[idx, 4:5], component=4)
np.savetxt("./data/biodiesel_input.dat", np.hstack(
    (data_t[idx], data_y[idx, 0:1], data_y[idx, 1:2], data_y[idx, 2:3], data_y[idx, 3:4], data_y[idx, 4:5])))

data = dde.data.PDE(geom, ODE,
                    [bc0, bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8, bc9, bc10, bc11, bc12, bc13, bc14, bc15, bc16,
                     ic0, ic1, ic2, ic3, ic4], anchors=data_t)

net = dde.maps.FNN([17] + [128] * 3 + [17], "relu", "Glorot normal")


# 特征层
def feature_transform(t):
    t = t / 750
    return torch.cat(
        (
            t,
            torch.pow(t, 2),
            torch.pow(t, 3),
            torch.pow(t / 2, 2),
            torch.pow(t / 2, 3),
            torch.exp(-1 * t),
            torch.exp(-2 * t),
            torch.exp(1 * t),
            torch.exp(2 * t),
            torch.log2(t + 1),
            torch.log2(2 * t + 1),
            torch.log2(3 * t + 1),
            torch.log2(4 * t + 1),
            torch.sin(t),
            torch.sin(2 * t),
            torch.sin(3 * t),
            torch.sin(4 * t),
        ),
        dim=1
    )


net.apply_feature_transform(feature_transform)


def output_transform(t, y):
    return (
            torch.as_tensor(data_y[0]) + torch.tanh(t) * torch.tensor(np.divide(1, np.array(bc_weights))) * y
    )


net.apply_output_transform(output_transform)

model = dde.Model(data, net)

bc_weights = weight_bc(y)
ode_weights = [1] * 17
data_weights = [1] * 5
model.compile("adam", lr=1e-3, loss_weights=[1] * 39, external_trainable_variables=var_list)
# model.train(iterations=1000, display_every=1000)
# 预训练，优先获得数据锚点
# data_weights = weight_data()
# data_weights = weightBD_data()
# ode_weights = [0] * 17
# model.compile("adam", lr=1e-3, loss_weights=ode_weights + bc_weights + data_weights,
#               external_trainable_variables=var_list)
# model.train(iterations=1000, display_every=1000)
# ode_weights = weight_ode()

# if noise >= 0.1:
#     bc_weights = [w * 10 for w in bc_weights]
# # Large noise requires small data_weights
# if noise >= 0.1:
#     data_weights = [w / 10 for w in data_weights]

# Large noise requires large ode_weights
# if noise > 0:
#     ode_weights = [10 * w for w in ode_weights]
# model.compile("adam", lr=1e-3, loss_weights=ode_weights + bc_weights + data_weights,
#               external_trainable_variables=var_list)
variable = dde.callbacks.VariableValue(
    var_list, period=1000, filename="./data/variables.dat", precision=4,
)
checker = dde.callbacks.ModelCheckpoint(
    "model/model.ckpt", save_better_only=True, period=1000
)
losshistory, train_state = model.train(iterations=10000, display_every=1000, callbacks=[variable, checker])
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
