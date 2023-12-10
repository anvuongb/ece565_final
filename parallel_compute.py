import numpy as np
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import tqdm
from helpers import *
from hyperparameters_estimations import gradient_descent_line_search
from multiprocessing import Pool, cpu_count

# Thread-based parallelized computation of Monte Carlo

sigma = 1
N_sims = 200
N_data = 2000

def compute_for_each_gamma(gamma):
    mse_mle = []
    mse_map = []
    mse_mmse = []

    n = np.random.randn(N_data, N_sims)*sigma
    s = np.random.laplace(scale=gamma, size=(N_data, N_sims))
    x = n + s

    s_mle = mle_estimate(x)
    s_map = map_estimate(x, sigma, gamma)
    s_mmse = mmse_estimate(x, sigma, gamma)

    mse_mle = mse(s_mle, s)
    mse_map = mse(s_map, s)
    mse_mmse = mse(s_mmse, s)

    return [gamma, np.mean(mse_mle), np.mean(mse_map), np.mean(mse_mmse)]

# Monte Carlo sims - Gamma
# params
gamma_list = np.arange(0.1, 10.1, 0.1) # gamma from 0.1 to 10 with 0.1 spacing.

# runs
crlb_ = crlb(sigma, N_data)
print(f"Detected {cpu_count()} threads")
print(f"Running Monte Carlo - Gamma on {cpu_count()} thread")

pool = Pool(cpu_count())
result = list(tqdm.tqdm(pool.imap(compute_for_each_gamma, gamma_list), total=len(gamma_list)))

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
plt.savefig("sim_1.png")