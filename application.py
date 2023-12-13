import numpy as np
import cv2
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
import pywt
import pywt.data
from helpers import mle_gamma, map_estimate, mmse_estimate, mse

# load image and add noise
original = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE )
original = cv2.resize(original, (0, 0), fx = 0.5, fy = 0.5)
figure = plt.figure(figsize=(7,5))
plt.imshow(original, cmap=plt.cm.gray)
plt.show()

# before adding noise
coeffs2 = pywt.dwt2(original, 'haar')
LL_true, (LH_true, HL_true, HH_true) = coeffs2

gamma_LH = mle_gamma(LH_true.ravel())
gamma_HL = mle_gamma(HL_true.ravel())
gamma_HH = mle_gamma(HH_true.ravel())

print(gamma_LH, gamma_HL, gamma_HH)

# add noise
s = original.shape
sigma = 20
X = original + np.random.normal(scale=sigma, size=s)
X = np.clip(X, 0, 255)
cv2.imwrite("images/noisy.png", X)

# Generate DWT of original image
# before adding noise
coeffs2 = pywt.dwt2(original, 'haar')
LL_true, (LH_true, HL_true, HH_true) = coeffs2
upper = np.concatenate([LL_true, HL_true], axis=1)
lower = np.concatenate([LH_true, HH_true], axis=1)
im = np.concatenate([upper, lower], axis=0)
# im = np.concatenate([X, im[0:323, :]], axis=1)
cv2.imwrite("images/original_transform.png", im)

# Generate DWT of noisy image
# before adding noise
coeffs2 = pywt.dwt2(X, 'haar')
LL, (LH, HL, HH) = coeffs2
upper = np.concatenate([LL, HL], axis=1)
lower = np.concatenate([LH, HH], axis=1)
im = np.concatenate([upper, lower], axis=0)
# im = np.concatenate([X, im[0:323, :]], axis=1)
cv2.imwrite("images/noisy_transform.png", im)

# Perform estimation 
LH_map = map_estimate(LH, sigma=sigma, gamma=gamma_LH)
LH_mmse = mmse_estimate(LH, sigma=sigma, gamma=gamma_LH)
print("MSE LH ML-MAP-MMSE", np.mean(np.square(LH-LH_true)), np.mean(np.square(LH_map-LH_true)), np.mean(np.square(LH_mmse-LH_true)))

HL_map = map_estimate(HL, sigma=sigma, gamma=gamma_HL)
HL_mmse = mmse_estimate(HL, sigma=sigma, gamma=gamma_HL)
print("MSE HL ML-MAP-MMSE", np.mean(np.square(HL-LH_true)), np.mean(np.square(HL_map-LH_true)), np.mean(np.square(HL_mmse-LH_true)))

HH_map = map_estimate(HH, sigma=sigma, gamma=gamma_HH)
HH_mmse = mmse_estimate(HH, sigma=sigma, gamma=gamma_HH)
print("MSE HH ML-MAP-MMSE", np.mean(np.square(HH-LH_true)), np.mean(np.square(HH_map-LH_true)), np.mean(np.square(HH_mmse-LH_true)))

# save a sample dist
fig = plt.figure(figsize=(7,5))
plt.hist(HL.ravel(), bins=100, density=True)
plt.xlabel("values of DWT coefficients")
plt.ylabel("density")
fig.savefig("images/lap_dist.png")

# save images
inv_map = pywt.idwt2((LL, (LH_map, HL_map, HH_map)), 'haar')
cv2.imwrite("images/map_recon.png", inv_map)
inv_mmse = pywt.idwt2((LL, (LH_mmse, HL_mmse, HH_mmse)), 'haar')
cv2.imwrite("images/mmse_recon.png", inv_mmse)