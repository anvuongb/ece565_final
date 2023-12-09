from scipy.optimize import line_search
import numpy as np
from scipy.stats import norm

def grad_gamma(init_point, X):
    gamma, sigma = init_point[0], init_point[1]
    N = len(X)
    
    num1 = -X/(gamma**2)*np.exp(X/gamma)*norm.cdf((-X-sigma**2/gamma)/sigma) +\
            sigma/(gamma**2)*np.exp(X/gamma)*1/np.sqrt(2*np.pi)*np.exp(-1/2*((-X-sigma**2/gamma)/sigma)**2)
    
    num2 = X/(gamma**2)*np.exp(-X/gamma)*norm.cdf((X-sigma**2/gamma)/sigma) +\
            sigma/(gamma**2)*np.exp(-X/gamma)*1/np.sqrt(2*np.pi)*np.exp(-1/2*((X-sigma**2/gamma)/sigma)**2)
    
    denom = np.exp(X/gamma)*norm.cdf((-X-sigma**2/gamma)/sigma) +\
            np.exp(-X/gamma)*norm.cdf((X-sigma**2/gamma)/sigma)
    
    grad = -N/gamma - N*(sigma**2)/(gamma**3) + np.sum((num1 + num2)/denom)
    
    return grad

def grad_sigma(init_point, X):
    gamma, sigma = init_point[0], init_point[1]
    N = len(X)
    
    num1 = 1/np.sqrt(2*np.pi)*(X/(sigma**2)-1/gamma)*np.exp(X/gamma)*np.exp(-1/2*((-X-sigma**2/gamma)/sigma)**2)
    
    num2 = -1/np.sqrt(2*np.pi)*(X/(sigma**2)+1/gamma)*np.exp(-X/gamma)*np.exp(-1/2*((X-sigma**2/gamma)/sigma)**2)
    
    denom = np.exp(X/gamma)*norm.cdf((-X-sigma**2/gamma)/sigma) +\
            np.exp(-X/gamma)*norm.cdf((X-sigma**2/gamma)/sigma)
    
    grad = N*sigma/(gamma**2) + np.sum((num1 + num2)/denom)
    
    return grad

def grad_total(init_point, X):
    v = np.array([grad_gamma(init_point, X), grad_sigma(init_point, X)])
    return -v

def ml_objective_gamma_sigma(init_point, X):
    gamma, sigma = init_point[0], init_point[1]
    N = len(X)
    
    sum1 = N*np.log(1/2/gamma)
    sum2 = N*sigma**2//2/(gamma**2)
    sum3 = np.log(np.exp(X/gamma)*norm.cdf((-X-sigma**2/gamma)/sigma) + np.exp(-X/gamma)*norm.cdf((X-sigma**2/gamma)/sigma))
    sum3 = np.sum(sum3)
    
    return -(sum1 + sum2 + sum3)

def gradient_descent_line_search(gamma_init, sigma_init, X, steps=50, default_alpha=0.001):
    init_point = np.array([gamma_init, sigma_init])
    steps = 50
    default_alpha = 0.001
    
    try:
        # first run outside loop
        curr_point = init_point
        old_fval = ml_objective_gamma_sigma(curr_point, X)
        d = grad_total(curr_point, X)
        delta = -d/np.linalg.norm(d)
        alpha, _, _, _, _, _ = line_search(ml_objective_gamma_sigma, grad_total, curr_point, delta, args=([X]))
        new_point = init_point + alpha*delta
        new_fval = ml_objective_gamma_sigma(new_point, X)
        curr_point = new_point

        for i in range(steps-1):
            d = grad_total(curr_point, X)
            delta = -d/np.linalg.norm(d)
            new_alpha, _, _, new_fval, old_fval, _ = line_search(ml_objective_gamma_sigma, grad_total, curr_point, delta, args=([X]), old_fval=new_fval, old_old_fval=old_fval)
            if new_alpha is not None:
                new_point = curr_point + new_alpha*delta
            else:
                new_point = curr_point + default_alpha*delta
            curr_point = new_point
        if curr_point[0] == np.nan or curr_point[1] == np.nan:
            return None, None
        return curr_point[0], curr_point[1]
    except:
        return None, None