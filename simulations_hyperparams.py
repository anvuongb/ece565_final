import numpy as np
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *
from hyperparameters_estimations import gradient_descent_line_search

# similar to simulations.py
# but this file also estimates gamma, sigma using gradient ascent

# Monte Carlo sims - Gamma
# params
sigma = 1
gamma_list = np.arange(0.1, 10.1, 0.1) # gamma from 0.1 to 10 with 0.1 spacing
N_sims = 200
N_data = 2000

# runs
crlb_ = crlb(sigma, N_data)
mse_mle_ret = []
mse_map_ret = []
mse_mmse_ret = []
print("Running Monte Carlo - Gamma")

for gamma in tqdm.tqdm(gamma_list):
    mse_mle = []
    mse_map = []
    mse_mmse = []

    for i in range(N_sims):
        n = np.random.randn(N_data)*sigma
        s = np.random.laplace(scale=gamma, size=N_data)
        x = n + s

        gamma_est, sigma_est = -1, -1
        while gamma_est is None or gamma_est == np.nan or gamma_est < 0 or sigma_est is None or sigma_est == np.nan or sigma_est < 0:
            # estimate hyper parameters
            gamma_init = gamma + np.random.randn()
            while gamma_init < 0:
                gamma_init = gamma + np.random.randn()
                
            sigma_init = sigma + np.random.randn()
            while sigma_init < 0:
                sigma_init = sigma + np.random.randn()

            gamma_est, sigma_est = gradient_descent_line_search(gamma_init, sigma_init, x, steps=50, default_alpha=0.001)

        s_mle = mle_estimate(x)
        s_map = map_estimate(x, sigma_est, gamma_est)
        s_mmse = mmse_estimate(x, sigma_est, gamma_est)

        mse_mle.append(mse(s_mle, s))
        mse_map.append(mse(s_map, s))
        mse_mmse.append(mse(s_mmse, s))
    
    mse_mle_ret.append(np.mean(mse_mle))
    mse_map_ret.append(np.mean(mse_map))
    mse_mmse_ret.append(np.mean(mse_mmse))

print("Generating plot")
## Plots
gamma_arr = np.array(gamma_list)**2/(sigma**2)

fig = plt.figure(figsize=(8,6))
plt.plot(gamma_arr, np.repeat(crlb_, len(gamma_arr)), label="CRLB", linestyle=(0, (3, 10, 1, 10)))
plt.plot(gamma_arr, mse_map_ret, label="MAP estimates", linestyle="dotted")
plt.plot(gamma_arr, mse_mle_ret, label="MLE estimates", linestyle=(0, (3, 10, 1, 10, 1, 10)))
plt.plot(gamma_arr, mse_mmse_ret, label="MMSE estimates", linestyle="dashdot")
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("SNR ($\gamma^{2}/\sigma^{2}$)")
plt.ylabel("MSE")
plt.grid(True)
plt.title("MSE vs $\gamma^{2}/\sigma^{2}$")
plt.tight_layout()
# plt.show()
plt.savefig("sim_1_hyperparam.png")

# Monte Carlo sims - N_data
# params
sigma = 1
N_data_list = np.arange(1, 2000, 10) # gamma from 0.1 to 10 with 0.1 spacing
N_sims = 200
gamma = 1

# runs
mse_mle_ret = []
mse_map_ret = []
mse_mmse_ret = []
crlb_ret = []
print("Running Monte Carlo - N_data")
for N_data in tqdm.tqdm(N_data_list):
    crlb_ret.append(crlb(sigma, N_data))
    mse_mle = []
    mse_map = []
    mse_mmse = []

    for i in range(N_sims):
        n = np.random.randn(N_data)*sigma
        s = np.random.laplace(scale=gamma, size=N_data)
        x = n + s

        gamma_est, sigma_est = -1, -1
        while gamma_est is None or gamma_est == np.nan or gamma_est < 0 or sigma_est is None or sigma_est == np.nan or sigma_est < 0:
            # estimate hyper parameters
            gamma_init = gamma + np.random.randn()
            while gamma_init < 0:
                gamma_init = gamma + np.random.randn()
                
            sigma_init = sigma + np.random.randn()
            while sigma_init < 0:
                sigma_init = sigma + np.random.randn()

            gamma_est, sigma_est = gradient_descent_line_search(gamma_init, sigma_init, x, steps=50, default_alpha=0.001)

        s_mle = mle_estimate(x)
        s_map = map_estimate(x, sigma, gamma_est)
        s_mmse = mmse_estimate(x, sigma, gamma_est)

        mse_mle.append(mse(s_mle, s))
        mse_map.append(mse(s_map, s))
        mse_mmse.append(mse(s_mmse, s))
    
    mse_mle_ret.append(np.mean(mse_mle))
    mse_map_ret.append(np.mean(mse_map))
    mse_mmse_ret.append(np.mean(mse_mmse))

print("Generating plot")
## Plots

fig = plt.figure(figsize=(8,6))
plt.plot(N_data_list, crlb_ret, label="CRLB", linestyle=(0, (3, 10, 1, 10)))
plt.plot(N_data_list, mse_map_ret, label="MAP estimates", linestyle="dotted")
plt.plot(N_data_list, mse_mle_ret, label="MLE estimates", linestyle=(0, (3, 10, 1, 10, 1, 10)))
plt.plot(N_data_list, mse_mmse_ret, label="MMSE estimates", linestyle="dashdot")
plt.legend()
plt.yscale("log")
plt.xlabel("# data points")
plt.ylabel("MSE")
plt.grid(True)
plt.title(f"MSE vs #data points - $\gamma$={gamma}")
plt.tight_layout()
# plt.show()
plt.savefig("sim_2_hyperparam.png")