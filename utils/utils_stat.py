import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def kde_bayes_factor(samples, base=0.0, save_file=None, bandwidth=None, n_jobs=None, kernel='gaussian'):
    if samples.ndim == 1:
        # (samples, ), that is, only 1 feature
        samples = samples[:, None]

    if bandwidth is None:
        kde = KernelDensity(kernel=kernel)
        bandwidth = [0.01, 0.05, 0.1, 1.0]
        grid = GridSearchCV(kde, {'bandwidth': bandwidth}, n_jobs=n_jobs)
        grid.fit(samples)
        kde = grid.best_estimator_
    else:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(samples)

    if save_file is not None:
        joblib.dump(kde, save_file)

    log_prob = kde.score_samples(samples)
    log_prob_base = kde.score_samples([[base]])

    cnt_gt, cnt_eq, cnt_lt = 0, 0, 0
    for _ in log_prob:
        if _ > log_prob_base:
            cnt_gt += 1
        elif _ < log_prob_base:
            cnt_lt += 1
        else:
            cnt_eq += 1

    return 1.0 * cnt_gt / (cnt_gt + cnt_lt + cnt_eq), kde.bandwidth


def run_hypothesis_test(grads, opt, bandwidth=None):
    if opt.data in ['mnist', 'cifar10']:
        results = [[[
            kde_bayes_factor(grads[:, i, j, k], bandwidth=bandwidth, n_jobs=opt.n_jobs)
            for k in range(grads.shape[3])]
            for j in range(grads.shape[2])]
            for i in range(grads.shape[1])]
        p_s_results = [[[
            results[i][j][k][0]
            for k in range(grads.shape[3])]
            for j in range(grads.shape[2])]
            for i in range(grads.shape[1])]
        bandwidths = [[[
            results[i][j][k][1]
            for k in range(grads.shape[3])]
            for j in range(grads.shape[2])]
            for i in range(grads.shape[1])]
    else:
        results = [
            kde_bayes_factor(grads[:, i], bandwidth=bandwidth, n_jobs=opt.n_jobs)
            for i in range(opt.n_features)]
        p_s_results = [results[i][0] for i in range(opt.n_features)]
        bandwidths = [results[i][1] for i in range(opt.n_features)]

    return np.array(p_s_results), np.array(bandwidths)
