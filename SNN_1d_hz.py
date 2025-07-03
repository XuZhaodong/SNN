# SNN method
# Subspace Method Based on Neural Networks for Solving The Partial Differential Equation
# Author: Xu zhaodong;Sheng zhiqiang
# Case : Numerical example with 1D helmholtz equation

import time
import torch
import numpy as np
from scipy.linalg import lstsq
import torch.nn as nn

# =============================================================================
# # Parameter settings for the SNN method
# # epochs: the number of training times for the neural network subspace
# # N_text: the number of calculation points for the test error
# # N_in: the number of internal calculation points
# # N_b: the number of boundary calculation points
# # lamb: the parameters of the helmholtz equation
# # X_min: the lower bound of the calculation domain
# # X_max: the upper bound of the calculation domain
# # subspace_dim: the dimension of the subspace
# # net_size: The size of the MLP network structure
# # eL: the stopping criterion for the training of the neural network subspace
# =============================================================================
epochs = 5000
N_text = 2000
N_in = 1000
N_b = 1
lamb = 10
X_min = 0.0
X_max = 2.0
subspace_dim = 300
eL = 1e-3
net_size = [1, 100, 100, 100, 100, subspace_dim, 1]
k = 2 * (len(net_size) - 2)

print(f"epochs = {epochs}\nN_text = {N_text}\nN_in = {N_in}\nN_b = {N_b}\nlamb = {lamb}\nX_min = {X_min}\nX_max = {X_max}\nsubspace_dim = {subspace_dim}\neL = {eL}\nnet_size = {net_size}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type(torch.DoubleTensor)


setup_seed(1)


def exact(x):
    return torch.sin(3 * torch.pi * x + 3 * torch.pi / 20) * torch.cos(2 * torch.pi * x + torch.pi / 10) + 2


def ff(x):
    f = torch.sin(3 * torch.pi * x + 3 * torch.pi / 20)
    g = torch.cos(2 * torch.pi * x + torch.pi / 10)
    f_prime = 3 * torch.pi * torch.cos(3 * torch.pi * x + 3 * torch.pi / 20)
    g_prime = -2 * torch.pi * torch.sin(2 * torch.pi * x + torch.pi / 10)
    f_double_prime = -(3 * torch.pi) ** 2 * torch.sin(3 * torch.pi * x + 3 * torch.pi / 20)
    g_double_prime = -(2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x + torch.pi / 10)
    u_double_prime = f_double_prime * g + 2 * f_prime * g_prime + f * g_double_prime
    u = f * g + 2
    result = u_double_prime - lamb * u
    return result


def interior(n=N_in):
    x = torch.linspace(X_min, X_max, n).unsqueeze(1).double()
    cond = ff(x)
    return x.requires_grad_(True), cond


def left(n=N_b):
    x = X_min * torch.ones(n, 1).double()
    cond = exact(x)
    return x.requires_grad_(True), cond


def right(n=N_b):
    x = X_max * torch.ones(n, 1).double()
    cond = exact(x)
    return x.requires_grad_(True), cond


class NormalizeLayer(nn.Module):
    def __init__(self, a, b):
        super(NormalizeLayer, self).__init__()
        self.a = a
        self.b = b
        self.a = torch.tensor(a, dtype=torch.double)
        self.b = torch.tensor(b, dtype=torch.double)

    def forward(self, x):
        return self.a * x + self.b


class Net(nn.Module):
    def __init__(self, size=net_size, a=2 / (X_max - X_min), b=-1 - 2 * X_min / (X_max - X_min)):
        super(Net, self).__init__()
        self.normalize = NormalizeLayer(a, b)
        layers = []
        for i in range(len(size) - 1):
            if i < len(size) - 2:
                layers.append(nn.Linear(size[i], size[i + 1]).double())
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(size[i], size[i + 1], bias=False).double())
        self.net = nn.Sequential(*layers)

    def forward(self, x, return_hidden_layer=False):
        x_normalized = self.normalize(x)
        last_activation_output = None

        for layer in self.net:
            x_normalized = layer(x_normalized)
            if isinstance(layer, nn.Tanh):
                last_activation_output = x_normalized

        if return_hidden_layer:
            return last_activation_output

        return x_normalized

    def get_subspace_output(self, x):
        return self.forward(x, return_hidden_layer=True)


loss = nn.MSELoss()


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def compute_gradients(out, point):
    grad1 = []
    grad2 = []

    for i in range(subspace_dim):
        g1 = torch.autograd.grad(outputs=out[:, i], inputs=point,
                                 grad_outputs=torch.ones_like(out[:, i]),
                                 create_graph=True, retain_graph=True)[0]
        grad1.append(g1.squeeze().detach().numpy())

        g2 = torch.autograd.grad(outputs=g1, inputs=point,
                                 grad_outputs=torch.ones_like(g1),
                                 create_graph=False, retain_graph=True if i < subspace_dim - 1 else False)[0]
        grad2.append(g2.squeeze().detach().numpy())

    grad1_np = np.array(grad1).T
    grad2_np = np.array(grad2).T

    return grad1_np, grad2_np


def loss_interior(u):
    x, f = interior()
    u_in = u(x)
    return loss(gradients(u_in, x, 2) - lamb * u_in, f)


def compute_error(net, exact_solution, x):
    u_pred = net(x).detach().numpy()
    u_exact = exact_solution(x).detach().numpy()
    relative_l2_error = np.sqrt(np.sum((u_pred - u_exact) ** 2) / np.sum(u_exact ** 2))
    max_error = np.max(np.abs(u_pred - u_exact))
    return max_error, relative_l2_error


def assemble_matrix(net, points_interior, points_BC, subspace_dim, N_in, lamb):
    A_I = np.zeros([N_in, subspace_dim])
    A_B = np.zeros([2, subspace_dim])
    F = np.zeros([N_in + 2, 1])

    out = net.get_subspace_output(points_interior)
    values = out.detach().numpy()
    grad1, grad2 = compute_gradients(out, points_interior)

    Lu = grad2 - lamb * values
    A_I = Lu
    F_I = ff(points_interior).detach().numpy()

    point_BC = points_BC.clone().detach().requires_grad_(True)
    out_BC = net.get_subspace_output(point_BC)
    values_BC = out_BC.detach().numpy()
    A_B = values_BC
    F_B = exact(point_BC).detach().numpy()

    A = np.vstack((A_I, A_B))
    F = np.vstack((F_I, F_B))

    return (A, F)


def solve_lstsq(A, f):
    scale = 1.0
    for i in range(len(A)):
        ratio = scale / A[i, :].max()
        A[i, :] *= ratio
        f[i] *= ratio
    w = lstsq(A, f)[0]
    return w

def main():
    start_time = time.time()

    x_eval = torch.linspace(X_min, X_max, N_text).unsqueeze(1).double()

    points_interior, _ = interior()
    left_points, _ = left()
    right_points, _ = right()
    points_BC = torch.cat([left_points, right_points], dim=0)

    u = Net()
    u = u.double()

    params_to_optimize = [
        param for name, param in u.named_parameters() if f"net.{k}" not in name
    ]
    opt = torch.optim.Adam(params=params_to_optimize)
    u.net[k].weight = torch.nn.Parameter(torch.ones_like(u.net[k].weight))

    for i in range(epochs):
        opt.zero_grad()
        l = loss_interior(u)
        if i == 0:
            flag = l * eL
        if l <= flag:
            print(f"Training of the neural network subpace has stopped.")
            L_infty_error, l2_error = compute_error(u, exact, x_eval)
            print(f"Current Epoch: {i + 1}, Loss: {l.item()}")
            break

        l.backward()

        if i + 1 == 1:
            L_infty_error, l2_error = compute_error(u, exact, x_eval)
            print(f"Epoch: {i + 1}, Loss: {l.item()}")

        if (i + 1) % (epochs / 100) == 0 and i < epochs:
            L_infty_error, l2_error = compute_error(u, exact, x_eval)
            print(f"Epoch: {i + 1}, Loss: {l.item()}")

        opt.step()

    print('Calculating the coefficients of the subspace basis functions:')
    A, f = assemble_matrix(u, points_interior, points_BC, subspace_dim, N_in, lamb)
    print('Matrix shape: N=%s,M=%s' % (A.shape))
    w = solve_lstsq(A, f)

    w_tensor = torch.from_numpy(w).double()
    w_tensor = w_tensor.view(1, -1)
    u.net[k].weight = nn.Parameter(w_tensor)

    L_infty_error, l2_error = compute_error(u, exact, x_eval)
    l = loss_interior(u)
    end_time = time.time()

    print(f"The coefficients of the subspace basis functions have been updated. ")
    print(f"Final loss: {l.item()}\nFinal l_infty_error: {L_infty_error.item()}\nFinal relative_l2_error: {l2_error.item()}")
    print(f"Running time:{end_time - start_time}s.")

if __name__ == "__main__":
    main()
