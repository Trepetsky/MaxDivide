import numpy as np
import scipy.spatial
import scipy.linalg
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
from scipy.stats import moment
import scipy.stats as stats
from scipy.special import kl_div


class MultiDistMetric:
    def __init__(self, data_list, metrics_keys):
        self.data_list = data_list  # list of datasets, each a 2D numpy array
        self.all_metrics = {
            'mean_corr': (-1, mean_corr),
            'gen_var': (1, gen_variance),
            'mean_mahal': (1, mean_mahalanobis),
            'cov_trace': (1, cov_trace),
            'max_eigen': (1, max_eigenvalue),
            'tot_var_dist': (1, total_variation_distance),
            'energy_stat': (1, energy_statistic),
            'joint_ent': (-1, joint_entropy),
            'cond_ent': (-1, conditional_entropy),
            'mean_joint_mom': (-1, mean_joint_moment)
        }
        self.metrics = {key: self.all_metrics[key] for key in metrics_keys}

    def calc_metrics(self):
        dist_sum = 0
        n = len(self.data_list)
        for i in range(n):
            for j in range(i+1, n):
                dist_sum += self.calc_pairwise_metrics(self.data_list[i], self.data_list[j])
        return dist_sum

    def calc_pairwise_metrics(self, data1, data2):
        metric_sum = 0
        for multiplier, metric_func in self.metrics.values():
            metric_sum += multiplier * metric_func(data1, data2)
        return metric_sum


def mean_corr(data1, data2):
    corr1 = np.corrcoef(data1, rowvar=False).mean()
    corr2 = np.corrcoef(data2, rowvar=False).mean()
    return abs(corr1 - corr2)

def gen_variance(data1, data2):
    return abs(np.var(data1) - np.var(data2))

def mean_mahalanobis(data1, data2):
    cov = np.cov(data1, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    mean_vec = np.mean(data1, axis=0)
    distances = np.apply_along_axis(lambda x: scipy.spatial.distance.mahalanobis(x, mean_vec, inv_cov), 1, data2)
    return np.mean(distances)

def cov_trace(data1, data2):
    return abs(np.trace(np.cov(data1, rowvar=False)) - np.trace(np.cov(data2, rowvar=False)))

def max_eigenvalue(data1, data2):
    return abs(np.max(np.linalg.eigvals(np.cov(data1, rowvar=False))) - np.max(np.linalg.eigvals(np.cov(data2, rowvar=False))))

def joint_entropy(data1, data2):
    data_combined = np.vstack((data1, data2))
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_combined)
    kde = stats.gaussian_kde(data_pca.T)
    entropy = -kde.logpdf(data_pca.T).mean()
    return entropy

def conditional_entropy(data1, data2):
    data_combined = np.vstack((data1, data2))
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_combined)
    kde_joint = stats.gaussian_kde(data_pca.T)
    kde_marginal = stats.gaussian_kde(data_pca[:, 0])
    entropy = -kde_joint.logpdf(data_pca.T).mean() + kde_marginal.logpdf(data_pca[:, 0]).mean()
    return entropy

def total_variation_distance(data1, data2):
    p = np.histogramdd(data1)[0].ravel()
    q = np.histogramdd(data2)[0].ravel()
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(np.abs(p - q)) / 2

def energy_statistic(data1, data2):
    n = data1.shape[0]
    m = data2.shape[0]
    distance_1 = np.linalg.norm(data1[:, None] - data1[None, :], axis=2)
    distance_2 = np.linalg.norm(data2[:, None] - data2[None, :], axis=2)
    distance_12 = np.linalg.norm(data1[:, None] - data2[None, :], axis=2)
    return 2/n/m*np.sum(distance_12) - 1/n**2*np.sum(distance_1) - 1/m**2
