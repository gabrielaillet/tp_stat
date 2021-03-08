from Tp_1 import n_bernoulli
import scipy.stats as stat
import numpy as np
from numpy.random import randn
import imageio
from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import loadtxt


def Confiance_bernoulli(n_ber, alpha):
    t_alpha = stat.norm.ppf(alpha)
    na = np.size(n_ber)
    n_np = np.array(n_ber)
    p_n = n_np.sum()
    p_n = float(p_n)
    p_n /= na
    alpha_n_alphna = (t_alpha ** 2 / na)
    delta = 4 * p_n * (1 - p_n) * alpha_n_alphna + alpha_n_alphna ** 2
    p_n_minus = (p_n + (alpha_n_alphna / 2) - float(np.sqrt(delta)) / 2) / (1 + alpha_n_alphna)
    p_n_plus = (p_n + (alpha_n_alphna / 2) + float(np.sqrt(delta)) / 2) / (1 + alpha_n_alphna)
    return p_n_minus, p_n_plus


def n_echant_bernou(n, alpha, p):
    total = 0
    for i in range(n):
        n_bernou = n_bernoulli(1000, p)
        n_m, n_p = Confiance_bernoulli(n_bernou, alpha)
        if n_m < p < n_p:
            total += 1
    return total / n


# exercice 4

def mean_borne(m, p, alpha):
    t_alpha = stat.norm.ppf(alpha)
    n_bernou = n_bernoulli(m, p)
    p_min, p_plus = Confiance_bernoulli(n_bernou, alpha)
    difference = p_plus - p_min

    if difference < (1 + m / t_alpha ** 2) ** (-1 / 2):
        print("success")
    else:
        print("failure")


def finding_n_echant(alpha, delta):
    return np.floor((stat.norm.ppf(alpha) / delta) ** 2)


def test_born():
    born = int(finding_n_echant(0.6, 0.01))
    b = np.zeros(100)
    for t in range(100):
        print (t)
        a = np.zeros(700)
        for i in range(1, 700):
            t_alpha = stat.norm.ppf(0.6)
            p = (1 + i / t_alpha ** 2) ** (-1 / 2)
            n_bernou = n_bernoulli(i, 0.7)
            p_min, p_plus = Confiance_bernoulli(n_bernou, 0.6)
            difference = p_plus - p_min
            a[i] = difference - p
        if a[born] <= 0.01:
            b[t] = 1
    print(b.sum())


# Part 2

# exercice5

def finding_I(a, b, delta, alpha_confiance):
    n = int(finding_n_echant(alpha_confiance, delta))
    print (n)
    n_ber = np.zeros(n)
    y = randn(n)
    for i in range(n):
        if a < y[i] < b:
            n_ber[i] = 1

    return Confiance_bernoulli(n_ber, alpha_confiance)


'''
a,b = finding_I(0,100000,0.001,0.95)
print ("the intervalle is",a,b)
iab = norm.cdf(b) - norm.cdf(a)
print ("the true value is",iab)
'''

# exercice 5
frontales = loadtxt('frontales.txt')
number_of_lines = frontales.size
bernou = np.zeros(number_of_lines)
for i in range(number_of_lines):
    if frontales[i] < 3:
        bernou[i] = 1

trust_interval = Confiance_bernoulli(bernou, 0.99)
print(trust_interval)

# part 2.3
