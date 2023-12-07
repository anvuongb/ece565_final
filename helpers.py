import numpy as np
from scipy.stats import norm
import tqdm

## Estimators for s
def crlb(sigma, N):
    return sigma**2/N

def mse(s_pred, s_true):
    return np.mean(np.square(s_pred-s_true))

def mle_estimate(X):
    return np.mean(X)

def map_estimate(X, sigma, gamma):
    thresh = sigma**2/gamma
    ind_pos = X > thresh
    ind_neg = X < thresh
    
    s = np.zeros(X.shape)
    s[ind_pos] = X[ind_pos] - thresh
    s[ind_neg] = X[ind_neg] + thresh
    return s

def mmse_estimate(X, sigma, gamma):
    const = np.exp(X/gamma + sigma**2/2/(gamma**2))*norm.cdf((-X-sigma**2/gamma)/sigma) +\
            np.exp(-X/gamma + sigma**2/2/(gamma**2))*norm.cdf((X-sigma**2/gamma)/sigma)
    
    expr1 = np.exp(X/gamma + sigma**2/2/(gamma**2))*((sigma/np.sqrt(2*np.pi))*\
            (0-np.exp(-1/2/(sigma**2)*(X+sigma**2/gamma)**2)) + (X + sigma**2/gamma) * norm.cdf((-X-sigma**2/gamma)/sigma))
    expr2 = -np.exp(-X/gamma + sigma**2/2/(gamma**2))*((sigma/np.sqrt(2*np.pi))*\
            (0-np.exp(-1/2/(sigma**2)*(X-sigma**2/gamma)**2)) - (X - sigma**2/gamma) * norm.cdf((X-sigma**2/gamma)/sigma))
    return (expr1 + expr2)/const


## Gradient ascent for ML estimation of gamma and sigma
def grad_gamma(X, gamma, sigma):
    N = len(X)
    
    num1 = -X/(gamma**2)*np.exp(X/gamma)*norm.cdf((-X-sigma**2/gamma)/sigma) +\
            sigma/(gamma**2)*np.exp(X/gamma)*1/np.sqrt(2*np.pi)*np.exp(-1/2*((-X-sigma**2/gamma)/sigma)**2)
    
    num2 = X/(gamma**2)*np.exp(-X/gamma)*norm.cdf((X-sigma**2/gamma)/sigma) +\
            sigma/(gamma**2)*np.exp(-X/gamma)*1/np.sqrt(2*np.pi)*np.exp(-1/2*((X-sigma**2/gamma)/sigma)**2)
    
    denom = np.exp(X/gamma)*norm.cdf((-X-sigma**2/gamma)/sigma) +\
            np.exp(-X/gamma)*norm.cdf((X-sigma**2/gamma)/sigma)
    
    grad = -N/gamma - N*(sigma**2)/(gamma**3) + np.sum((num1 + num2)/denom)
    
    return grad

def grad_sigma(X, gamma, sigma):
    N = len(X)
    
    num1 = 1/np.sqrt(2*np.pi)*(X/(sigma**2)-1/gamma)*np.exp(X/gamma)*np.exp(-1/2*((-X-sigma**2/gamma)/sigma)**2)
    
    num2 = -1/np.sqrt(2*np.pi)*(X/(sigma**2)+1/gamma)*np.exp(-X/gamma)*np.exp(-1/2*((X-sigma**2/gamma)/sigma)**2)
    
    denom = np.exp(X/gamma)*norm.cdf((-X-sigma**2/gamma)/sigma) +\
            np.exp(-X/gamma)*norm.cdf((X-sigma**2/gamma)/sigma)
    
    grad = N*sigma/(gamma**2) + np.sum((num1 + num2)/denom)
    
    return grad
    
def ml_estimate_sigma_gamma(X, gamma_init, sigma_init, 
                            max_steps=200, step_size=0.001, early_stopping_ratio=0.01):
    gamma_curr = gamma_init
    gamma_curr_ = gamma_init
    gamma_list = []
    gamma_list.append(gamma_curr)

    sigma_curr = sigma_init
    sigma_curr_ = sigma_init
    sigma_list = []
    sigma_list.append(sigma_curr)

    for i in tqdm.tqdm(np.arange(max_steps)):
        # alternating between gamma and sigma, this is much more stable
        for j in range(3):
            grad_gamma_ = grad_gamma(X, gamma_curr_, sigma_curr)
            # only update if gradients are not too small
            if np.abs(grad_gamma_) >= early_stopping_ratio*np.abs(gamma_curr_):
                gamma_curr_ = gamma_curr_ + step_size*grad_gamma_
            else:
                break
        gamma_curr = gamma_curr_
        gamma_list.append(gamma_curr)
        
        for j in range(3):        
            grad_sigma_ = grad_sigma(X, gamma_curr, sigma_curr_)
            # only update if gradients are not too small
            if np.abs(grad_sigma_) >= early_stopping_ratio*np.abs(sigma_curr_):
                sigma_curr_ = sigma_curr_ + step_size*grad_sigma_
            else:
                break
        sigma_curr = sigma_curr_
        sigma_list.append(sigma_curr)
    
    return gamma_curr, sigma_curr, gamma_list, sigma_list

# def ml_estimate_sigma_gamma(X, gamma_init, sigma_init, 
#                             max_steps=200, step_size=0.001, 
#                             early_stopping_ratio = 0.01):
#     gamma_curr = gamma_init
#     gamma_list = []
#     gamma_list.append(gamma_curr)

#     sigma_curr = sigma_init
#     sigma_list = []
#     sigma_list.append(sigma_curr)

#     # for _ in tqdm.tqdm(np.arange(max_steps)):
#     for _ in np.arange(max_steps):
#         grad_gamma_ = grad_gamma(X, gamma_curr, sigma_curr)
#         grad_sigma_ = grad_sigma(X, gamma_curr, sigma_curr)

#         # only update if gradients are not too small
#         if np.abs(grad_gamma_) >= early_stopping_ratio*np.abs(gamma_curr) and gamma_curr + step_size*grad_gamma_ > 0:
#             gamma_curr_ = gamma_curr + step_size*grad_gamma_
#         else:
#             gamma_curr_ = gamma_curr
#         if np.abs(grad_sigma_) >= early_stopping_ratio*np.abs(sigma_curr) and sigma_curr + step_size*grad_sigma_ > 0:
#             sigma_curr_ = sigma_curr + step_size*grad_sigma_
#         else:
#             sigma_curr_ = sigma_curr
#         gamma_list.append(gamma_curr_)
#         sigma_list.append(sigma_curr_)

#         # early stopping if gradients are small
#         if np.abs(grad_gamma_) < early_stopping_ratio*np.abs(gamma_curr) and np.abs(grad_sigma_) < early_stopping_ratio*np.abs(sigma_curr) or (gamma_curr + step_size*grad_gamma_ < 0 and sigma_curr + step_size*grad_sigma_ < 0):
#             break

#         gamma_curr, sigma_curr = gamma_curr_, sigma_curr_
    
#     return gamma_curr, sigma_curr, gamma_list, sigma_list
