from numpy.random import rand as Uniforme
import matplotlib.pyplot as plt
import numpy as np
from math import factorial


######################################### 1 #################################
def Echantillon(n):
    return Uniforme(n)


# plt.hist(Echantillon(10000000),bins = 10,density = True)
# plt.show()

######################################## 2 ##################################

def Bernoulli(p):
    u = Uniforme(1)[0]
    if u < p:
        return 1
    return 0


def n_bernoulli(n, p):
    echan = []
    for i in range(n):
        echan += [Bernoulli(p)]
    return echan


def occurence(val, y):
    occ = np.zeros(val.size)
    for j in range(y.size):
        for k in range(val.size):
            if y[j] == val[k]:
                occ[k] += 1
                break
    return (occ)


def plot_n_bernoulli(n, p):
    val = np.zeros(2)
    val[0] = 1
    y = n_bernoulli(n, p)
    y = np.array(y)
    occ = occurence(val, y)
    for i in range(len(occ)):
        occ[i] = occ[i] / len(y)
    plt.bar(val, occ)
    plt.show()


# plot_n_bernoulli(10000,0.2)


def poivre_sel(p, Y):
    P = Y.copy()
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            x = Bernoulli(p)
            new_val = (1 - x) * Y[i][j] + x * 255
            P[i][j] = new_val
    return P


"""
Y = imageio.imread('cameraman.jpg')
plt.imshow(Y, cmap='gray')
plt.show()
Y = poivre_sel(0.1, Y)
plt.imshow(Y, cmap='gray')
plt.show()
"""


################################# 3 ####################################

def Exp(n, lamda):
    y = Uniforme(n)
    for i in range(len(y)):
        y[i] = -lamda * np.log(1 - y[i])
    return y


"""
plt.hist(Exp(1000000,3),bins = 1000,density = True,label='3')
plt.hist(Exp(1000000,2),bins = 1000,density = True,label='2')
plt.hist(Exp(1000000,1),bins = 1000,density = True,label='1')
plt.hist(Exp(1000000,0.5),bins = 1000,density = True,label='0.5')
plt.legend()
plt.show()
"""


def Rayleigh(n, lamda):
    y = Uniforme(n)
    for i in range(len(y)):
        y[i] = np.sqrt(lamda * lamda * np.log(1 / (1 - y[i])))
    return y


"""
plt.hist(Rayleight(100000,0.5),bins = 1000,density = True,label='0.5')
plt.hist(Rayleight(100000,1),bins = 1000,density = True,label='1')
plt.hist(Rayleight(100000,2),bins = 1000,density = True,label='2')
plt.hist(Rayleight(100000,3),bins = 1000,density = True,label='3')
plt.legend()
plt.show()
"""


def Gamme(n, alpha):
    y = np.power(Exp(n, 1), (alpha - 1))
    y = np.sum(y)
    y = y / n
    return y


def difference_gamma(n, alpha):
    y_exp = Gamme(n, alpha)
    y_real = factorial(alpha - 1)
    return abs(y_real - y_exp)/y_real


def plot_dif(n, alpha):
    y = np.zeros(n)
    truc = np.zeros(n)
    print(y[0])
    for i in range(n):
        truc[i] = i + 1
        y[i] = difference_gamma(i + 1, alpha)
    print(y[n - 1])
    plt.plot(truc, y)
    plt.show()


#print(difference_gamma(1000000,5))


def Beta(n, alpha, beta):
    return Gamme(n, alpha) * Gamme(n, beta) / Gamme(n, alpha + beta)

def Beta_bis(n,alpha,beta):
    y = Uniforme(n)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (y[i] ** (alpha - 1) ) * ((1-y[i])**(beta - 1))
    return  np.sum(z)

def resonnance(tau, n):
    y = np.ones((n, n))
    for i in range(n):
        for p in range(n):
            y[i][p] = Rayleigh(1, np.sqrt(2) * tau)
    plt.imshow(y, cmap='gray')
    plt.show()


#resonnance(0.1, 301)

def gaussien(m):
    y = Uniforme(m)
    y = y - 1/2
    Tm = np.sqrt(m) * (1/m * np.sum(y)) * np.sqrt(12)
    return Tm

def hist_gaussian(n,m,d):
    y = np.zeros(n)
    for i in range(n):
        y[i] = gaussien(m)
    print(y)
    plt.hist(y,bins=d,density = True)
    plt.show()

#hist_gaussian(1000000,1000,500)