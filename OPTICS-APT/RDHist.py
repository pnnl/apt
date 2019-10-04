"""
Ditting Histrogram of RD plot using Gaussian mixture models. 

Note that in future, a mixture of gamma functions may be better for \
this job. (need to figure out how to implement or which package to use)

author: Jing Wang
"""
import numpy as np
from sklearn import mixture
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

### Note that this is temporary solution, should solve all warnings later---
import warnings
warnings.filterwarnings("error", category=ConvergenceWarning)

__version__='0.2'

def CSR_kNND(k, rho):
    """
    The Complete spatial distribution of k-th nearest neighbor in 3-dimension
    can be treated as a special case of generalized gamma distribution
    (scaled in x and pdf space.)
    The implementation uses generalized gamma distribution from scipy and return
    a frozen distribution instance of object of the specific shape.

    k:
        k-th nearest neighbour (also called 'order' sometimes)
    rho:
        density of CSR, in #/nm3
    """
    d = 3 #dimension is 3, in default
    scale = np.power((4.0*np.pi/3.0*rho), -1.0/3.0) #scale that related to atomic density, rho
    CSRkNND = stats.gengamma(k, d, scale=scale)
    return CSRkNND

def est_random_knn(rho, det_eff, con, k):
    """
    Estimate k-the NN distance in a random background.

    rho:
        density
    det_eff:
        detection efficiency
    con:
        solute concentration
    k:
        k-th nearest neighbor
    """
    rho_solute = rho * det_eff * con
    CSR = CSR_kNND(k, rho_solute)
    CSR_stats = (CSR.mean(), CSR.std())

    print('The mean kNN distance of background is:', CSR_stats[0])
    print('The std kNN distance of background is:', CSR_stats[1])
    return CSR_stats

class BimodalCSRkNND(stats.rv_continuous):
    """
    NOT WORKING properly (utterly slow in fitting process). Something is wrong
    with rv_continuous class in scipy.
    """
    def _pdf(self, r, k, rho_1, rho_2, ratio):
        d = 3.0
        scale_1 = np.power((4.0*np.pi/3.0*rho_1), -1.0/3.0) #scale that related to atomic density, rho
        scale_2 = np.power((4.0*np.pi/3.0*rho_2), -1.0/3.0) #scale that related to atomic density, rho
        CSR_1 = stats.gengamma(k, d, scale=scale_1)
        CSR_2 = stats.gengamma(k, d, scale=scale_2)
        CSR_1_pdf = CSR_1.pdf(r)
        CSR_2_pdf = CSR_2.pdf(r)

        return ratio*CSR_1_pdf + (1.0-ratio)*CSR_2_pdf

def GMM_fitting_stats(RD):
    """
    Using two gaussian mixtures to fit a bimodal distribution, RD.
    However, there is no generally agreed summary of statistic to describe
    bimodal distribution.
    """
    try:
        gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', tol=1E-3, max_iter=1000)
        gmm.fit(RD)
        weights = gmm.weights_
        means = gmm.means_
        variances = gmm.covariances_

        return (weights[0], weights[1]), (means[0,0], means[1,0]), (np.sqrt(variances[0,0,0]), np.sqrt(variances[1,0,0]))
    except:
        gmm = mixture.GaussianMixture(n_components=1, covariance_type='full', tol=1E-3, max_iter=1000)
        gmm.fit(RD)
        weights = gmm.weights_
        means = gmm.means_
        variances = gmm.covariances_

        return weights, means, np.sqrt(variances[0,0,0])


    

def GMM_fitting_posterior_proba(RD, P=0.5, num=1000):
    """
    Fit RD with Gaussian Mixture model of 2 components. Return posterior probability
    of each component for each observation.

    Returen a maximum RD at which observation has P (default ~50%) chance of
    belonging to a clusters.
    """
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', tol=1E-3, max_iter=1000)
    gmm.fit(RD)

    means = gmm.means_

    if means[0, 0] < means[1, 0]:
        flag = 0
    else:
        flag = 1

    x=np.linspace(np.min(RD), np.max(RD), num)
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))

    x_idx = np.argwhere(responsibilities[:, flag] > P)

    if len(x_idx) == 0:
        return 0
    else:
        return x[np.max(x_idx)]


if __name__ == '__main__':
    # import APT_IOs as aptios
    import matplotlib.pyplot as plt

    RD_filename = 'new_RD.txt'

    temp = np.loadtxt(RD_filename, skiprows=1)
    temp = temp.reshape(-1,1)
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', tol=1E-5, max_iter=10000)
    gmm.fit(temp)

    weights = gmm.weights_
    means = gmm.means_
    variances = gmm.covariances_

    if means[0, 0] < means[1, 0]:
        flag = 0
    else:
        flag = 1

    x=np.linspace(np.min(temp), np.max(temp), num=100)
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    score_samples = gmm.score_samples(x.reshape(-1, 1))

    P=0.5
    x_idx = np.argwhere(responsibilities[:, flag] > P)

    # print(x_idx)
    respon = gmm.predict_proba(temp)
    respon_select = respon[:, flag]
    temp_select = temp[respon_select > P]
    # print(temp_select)
    # print(responsibilities)

#    print respon
#
#    label = gmm.predict(temp)

#    print label

    plt.figure()
    plt.plot(x, np.exp(score_samples), color='r')
    plt.plot(x[x_idx], responsibilities[x_idx, flag])
    plt.hist(temp, bins=100, density=True)
    plt.scatter(x[np.max(x_idx)], P)
    plt.show()

#    r = np.linspace(0.0, 2.5, num=1000)
#    CSR1 = CSR_kNND(10, 10.0)
#    CSR1_data = CSR1.pdf(r)
#    CSR2 = CSR_kNND(10, 33)
#    CSR2_data = CSR2.pdf(r)
#    CSR3 = CSR_kNND(10, 88)
#    CSR3_data = CSR3.pdf(r)
#    CSR4 = CSR_kNND(10, 1.0)
#    CSR4_data = CSR4.pdf(r)
#    plt.plot(r, CSR1_data)
#    plt.plot(r, CSR2_data)
#    plt.plot(r, CSR3_data)
#    plt.plot(r, CSR4_data)

# test random knn
    rho = 88.48 # per nm3
    det_eff = 0.37
    con = 0.12
    k = 10
    mean, std = est_random_knn(rho, det_eff, con, k)
    CSR = CSR_kNND(k, rho*det_eff*con)
    data = CSR.rvs(size=1000)
    plt.figure()
    plt.hist(data)
    plt.show()