import numpy as np
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *
from hyperparameters_estimations import grad_gamma

sigma = 1
gamma_true = 5

N_sims = 50
N_data = 2000

gamma_list_list = []
for i in tqdm.tqdm(range(N_sims)):
    n = np.random.randn(N_data)*sigma
    s = np.random.laplace(scale=gamma_true, size=(N_data))
    x = n + s

    gamma_init = 7
    step_size = 0.001
    iterations = 200

    gamma_list = []
    gamma_curr = gamma_init
    for j in range(iterations):
        grad = grad_gamma([gamma_curr, sigma], x)
        gamma_curr = gamma_curr + step_size*grad
        gamma_list.append(gamma_curr)
    gamma_list_list.append(gamma_list)

arr = np.array(gamma_list_list)
arr = np.mean(arr, axis=0)
figure = plt.figure(figsize=(7,5))
for gl in gamma_list_list:
    plt.plot(gl, color="gray", alpha=0.3)
plt.plot(arr, "r", linestyle="dashed", label="mean")
plt.hlines(gamma_true, 0, 200, label="true $\gamma$")
plt.legend()
plt.ylabel("$\gamma$")
plt.xlabel("# iteration")
plt.title("Estimation of $\gamma$ while keeping $\sigma$ constant, 50 realizations")
figure.savefig("estimate_gamma.png")