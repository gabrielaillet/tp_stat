from Tp_1 import n_bernoulli
import scipy.stats as stat
import numpy as np


n_bernou = n_bernoulli(10000,0.3)

def Confiance_bernoulli(n_ber,alpha):
    t_alpha = stat.norm.ppf(alpha)
    n = np.size(n_ber)
    p_n = np.sum(n_ber)/n
    alpha_n_alphna = (t_alpha**2 / n)
    delta = 4 * p_n * (1-p_n)* alpha_n_alphna + alpha_n_alphna**2
    p_n_minus = (p_n + (alpha_n_alphna/2) - np.sqrt(delta))/(1 + alpha_n_alphna)
    p_n_plus = (p_n + (alpha_n_alphna/2) + np.sqrt(delta))/(1 + alpha_n_alphna)

    return p_n_minus,p_n_plus

def n_echant(n,alpha,p):
    total = 0
    for i in range(n):
        n_bernou = n_bernoulli(1000, p)
        n_m,n_p = Confiance_bernoulli(n_bernou,alpha)
        if n_m < p < n_p:
            total += 1
    return total/n


print(n_echant(10000,0.2,0.3))



