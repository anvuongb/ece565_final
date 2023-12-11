import numpy as np
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *
from hyperparameters_estimations import gradient_descent_line_search, ml_objective_gamma_sigma
from multiprocessing import Pool, cpu_count
import warnings

# Thread-based parallelized computation of Monte Carlo
# This also uses the MML to estimate sigma and gamma for each run

np.seterr(all="ignore") # MML can result in division by zero, so ignore warnings
warnings.filterwarnings('ignore') # ignore warnings altogether

sigma = 1
N_sims = 200
N_data = 2000
N_MML_estimates = 5

def compute_for_each_gamma(gamma):
    mse_mle = []
    mse_map = []
    mse_mmse = []

    for i in range(N_sims):
        n = np.random.randn(N_data)*sigma
        s = np.random.laplace(scale=gamma, size=N_data)
        x = n + s

        obj_min = np.inf
        gamma_est, sigma_est = -1, -1
        # run MML estimations a few times and pick the pair that maximize the likelihood
        for j in range(N_MML_estimates):
            # print(gamma, i, j)
            gamma_est_, sigma_est_ = -1, -1
            while gamma_est_ is None or gamma_est_ == np.nan or gamma_est_ < 0 or sigma_est_ is None or sigma_est_ == np.nan or sigma_est_ < 0:
                # estimate hyper parameters
                gamma_init = gamma + np.random.randn()
                while gamma_init <= 0:
                    gamma_init = gamma + np.random.randn()
                    
                sigma_init = sigma + np.random.randn()
                while sigma_init <= 0:
                    sigma_init = sigma + np.sqrt(0.1)*np.random.randn() # variance 0.1

                gamma_est_, sigma_est_ = gradient_descent_line_search(gamma_init, sigma_init, x, steps=50, default_alpha=0.001)
            obj = ml_objective_gamma_sigma([gamma_est_, sigma_est_], x)
            if obj < obj_min:
                obj_min = obj
                gamma_est, sigma_est = gamma_est_, sigma_est_

        s_mle = mle_estimate(x)
        s_map = map_estimate(x, sigma_est, gamma_est)
        s_mmse = mmse_estimate(x, sigma_est, gamma_est)

        mse_mle.append(mse(s_mle, s))
        mse_map.append(mse(s_map, s))
        mse_mmse.append(mse(s_mmse, s))

    mse_mle = np.array(mse_mle)
    mse_map = np.array(mse_map)
    mse_mmse = np.array(mse_mmse)
    return [gamma, np.mean(mse_mle[~np.isnan(mse_mle)]), np.mean(mse_map[~np.isnan(mse_map)]), np.mean(mse_mmse[~np.isnan(mse_mmse)])]

# Monte Carlo sims - Gamma
# params
gamma_list = np.arange(0.1, 10.1, 0.1) # gamma from 0.1 to 10 with 0.1 spacing.

# runs
crlb_ = crlb(sigma, N_data)
print(f"Detected {cpu_count()} threads")
print(f"Running Monte Carlo - Gamma on {cpu_count()} thread")

pool = Pool(12)
result = list(tqdm.tqdm(pool.imap(compute_for_each_gamma, gamma_list), total=len(gamma_list)))

# print(result)
print("Done.")
mse_mle_ret = []
mse_map_ret = []
mse_mmse_ret = []
gamma_arr = []

for r in result:
    gamma_arr.append(r[0])
    mse_mle_ret.append(r[1])
    mse_map_ret.append(r[2])
    mse_mmse_ret.append(r[3])

print("Generating plot")
## Plots
gamma_arr = np.array(gamma_arr)**2/(sigma**2)

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