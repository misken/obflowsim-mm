import numpy as np
from scipy import optimize
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from scipy.stats import norm

from qng import qng


class ErlangcEstimator(BaseEstimator):
    """ Erlang-C formula for probability of wait in an M/M/c queue.

    No parameters are actually fit. This is just an analytical formula implemented
    as an sklearn Estimator so that it can be used in pipelines.

    Parameters
    ----------
    col_idx_arate : float
        Column number in X corresponding to arrival rate
    col_idx_meansvctime : float
        Column number in X corresponding to mean service time
    col_idx_numservers : int
        Column number in X corresponding to number of servers (c) in system

    """

    def __init__(self, col_idx_arate, col_idx_meansvctime, col_idx_numservers):
        self.col_idx_arate = col_idx_arate
        self.col_idx_meansvctime = col_idx_meansvctime
        self.col_idx_numservers = col_idx_numservers

    def fit(self, X, y=None):
        """Empty fit method since no parameters to be fit

        Checks shapes of X, y and sets is_fitted_ to True.
        Use ``predict`` to get predicted y values.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False)
        else:
            X = check_array(X, accept_sparse=False)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ Compute Erlang-C using qng library
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        X_df = pd.DataFrame(X)

        y = X_df.apply(
            lambda x: qng.erlangc(x[self.col_idx_arate] * x[self.col_idx_meansvctime], int(x[self.col_idx_numservers])),
            axis=1)

        return np.array(y)


class LoadEstimator(BaseEstimator):
    """ Load as approximation for mean occupancy

    No parameters are actually fit. This is just an analytical formula implemented
    as an sklearn Estimator so that it can be used in pipelines.

    Parameters
    ----------
    col_idx_arate : float
        Column number in X corresponding to arrival rate
    col_idx_meansvctime : float
        Column number in X corresponding to mean service time

    """

    def __init__(self, col_idx_arate, col_idx_meansvctime):
        self.col_idx_arate = col_idx_arate
        self.col_idx_meansvctime = col_idx_meansvctime

    def fit(self, X, y=None):
        """Empty fit method since no parameters to be fit

        Checks shapes of X, y and sets is_fitted_ to True.
        Use ``predict`` to get predicted y values.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False)
        else:
            X = check_array(X, accept_sparse=False)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ Compute load as arrival_rate * avg_svc_time
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        X_df = pd.DataFrame(X)

        y = X_df.apply(lambda x: x[self.col_idx_arate] * x[self.col_idx_meansvctime], axis=1)

        return np.array(y)


class SqrtLoadEstimator(BaseEstimator):
    """ Load based normal approximation for occupancy percentiles

    No parameters are actually fit. This is just an analytical formula implemented
    as an sklearn Estimator so that it can be used in pipelines.

    Parameters
    ----------
    col_idx_arate : float
        Column number in X corresponding to arrival rate
    col_idx_meansvctime : float
        Column number in X corresponding to mean service time
    pctile: float
        Percentile of interest

    """

    def __init__(self, col_idx_arate, col_idx_meansvctime, pctile):
        self.col_idx_arate = col_idx_arate
        self.col_idx_meansvctime = col_idx_meansvctime
        self.pctile = pctile

    def fit(self, X, y=None):
        """Empty fit method since no parameters to be fit

        Checks shapes of X, y and sets is_fitted_ to True.
        Use ``predict`` to get predicted y values.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False)
        else:
            X = check_array(X, accept_sparse=False)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ Compute load as arrival_rate * avg_svc_time
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        zscore = norm().ppf(self.pctile)

        X_df = pd.DataFrame(X)

        y = X_df.apply(lambda x: x[self.col_idx_arate] * x[self.col_idx_meansvctime] +
                                 zscore * np.sqrt(x[self.col_idx_arate] * x[self.col_idx_meansvctime]), axis=1)

        return np.array(y)


class CondMeanWaitLDREstimator(BaseEstimator):
    """ Approximation for conditional mean wait time in LDR due to blocking by PP

    No parameters are actually fit. This is just an analytical formula implemented
    as an sklearn Estimator so that it can be used in pipelines.

    Parameters
    ----------
    col_idx_arate : float
        Column number in X corresponding to arrival rate
    col_idx_meansvctime : float
        Column number in X corresponding to mean service time
    col_idx_numservers : int
        Column number in X corresponding to number of servers (c) in system
    col_idx_cv2svctime : float
        Column number in X corresponding to mean service time

    """

    def __init__(self, col_idx_arate, col_idx_meansvctime, col_idx_numservers, col_idx_cv2svctime):
        self.col_idx_arate = col_idx_arate
        self.col_idx_meansvctime = col_idx_meansvctime
        self.col_idx_numservers = col_idx_numservers
        self.col_idx_cv2svctime = col_idx_cv2svctime

    def fit(self, X, y=None):
        """Empty fit method since no parameters to be fit

        Checks shapes of X, y and sets is_fitted_ to True.
        Use ``predict`` to get predicted y values.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False)
        else:
            X = check_array(X, accept_sparse=False)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ Compute conditional mean wait using mgc approximation
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        X_df = pd.DataFrame(X)

        y = X_df.apply(
            lambda x: condmeantime_blockedby_pp_hat(x[self.col_idx_arate], x[self.col_idx_meansvctime],
                                  int(x[self.col_idx_numservers]), x[self.col_idx_cv2svctime]),
            axis=1)

        return np.array(y)


def prob_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime):
    """
    Approximate probability of being blocked in ldr or csect waiting for a pp bed.

    Modeling pp as an M/G/c queue and using erlang C approx
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    prob = qng.mgc_prob_wait_erlangc(arr_rate, pp_svcrate, int(pp_cap))

    return prob


def meantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime):
    """
    Approximate unconditional mean time blocked in ldr or csect waiting for a pp bed.

    Modeling pp as an M/G/c queue and using approximation by Kimura.
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    meantime = qng.mgc_mean_qwait_kimura(arr_rate, pp_svcrate, int(pp_cap), pp_cv2_svctime)

    return meantime


def ldr_vartime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime):
    """
    Approximate unconditional variance of time blocked in ldr waiting for a pp bed.

    Modeling pp as an M/G/c queue and using approximation by Whitt.
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    vartime = qng.ggm_qwait_whitt_varw(arr_rate, pp_svcrate, int(pp_cap), 1.0, pp_cv2_svctime)

    return vartime


def condmeantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime):
    """
    Approximate conditional mean time blocked in ldr or csect waiting for a pp bed.

    Modeling pp as an M/G/c queue
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    prob = qng.mgc_prob_wait_erlangc(arr_rate, pp_svcrate, int(pp_cap))
    meantime = qng.mgc_mean_qwait_kimura(arr_rate, pp_svcrate, int(pp_cap), pp_cv2_svctime)
    condmeantime = meantime / prob

    return condmeantime


def condpctiletime_blockedby_pp_hat(prob, arr_rate, pp_mean_svctime, pp_cv2_svctime, pp_cap):
    pass

    return -1


def _fixedpt_func_mgc_mean_qwait(x, arr_rate, effsvctime, cap, cv2):
    svcrate = 1.0 / (effsvctime - x)

    # Need to worry about rho >= 1
    # if svcrate <= arr_rate:
    #     # rho is >=1
    #     #raise ValueError("Traffic intensity >= 1.0")
    #     return np.infty
    # else:
    return qng.mgc_mean_qwait_kimura(arr_rate, svcrate, cap, cv2)


def obs_blockedby_ldr_hats(arr_rate, csect_rate, ldr_mean_svctime, ldr_cv2_svctime, ldr_cap,
                           pp_mean_svctime, pp_cv2_svctime, pp_cap):
    # Use MGc approximation
    ldr_meantime_blockedby_pp = meantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, int(pp_cap), pp_cv2_svctime)
    # ldr_vartime_blockedby_pp = ldr_vartime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime)

    # Start with no reduction by queueing time in obs. Only non c-sections can be blocked in LDR
    ldr_effmean_svctime_init = ldr_mean_svctime + (1 - csect_rate) * ldr_meantime_blockedby_pp
    # ldr_effvar_svctime_init = ldr_cv2_svctime * (ldr_mean_svctime ** 2) + ldr_vartime_blockedby_pp

    # The following two variables not used right now as we are assuming effective variance is constant
    # Review this next estimate for eff variance in svc time in LDR
    # ldr_effvar_svctime_init = ldr_cv2_svctime * (ldr_mean_svctime ** 2) + ldr_vartime_blockedby_pp
    # ldr_effcv2_svctime_init = ldr_effvar_svctime_init / (ldr_effmean_svctime_init ** 2)

    # Estimate mean time blocked in obs by solving a fixed point problem
    try:
        fixed_point = optimize.fixed_point(_fixedpt_func_mgc_mean_qwait, [0.0], xtol=1e-04,
                                           args=(arr_rate, ldr_effmean_svctime_init, int(ldr_cap), ldr_cv2_svctime))

        meantime_blockedbyldr_fixedpt = fixed_point[0]  # Since optimize.fixed_point returns an array

        # Now update the estimate of effective svc time in LDR
        # Oops! Following line double adds the PP part
        # ldr_effmean_svctime_final = ldr_effmean_svctime_init + (1 - csect_rate) * ldr_meantime_blockedby_pp - meantime_blockedbyldr_fixedpt
        ldr_effmean_svctime_final = ldr_effmean_svctime_init - meantime_blockedbyldr_fixedpt

        # Finally, compute estimate of prob blocked in obs and conditional meantime blocked
        prob_blockedby_ldr = qng.mgc_prob_wait_erlangc(arr_rate, 1.0 / ldr_effmean_svctime_final, int(ldr_cap))
        condmeantime_blockedbyldr = meantime_blockedbyldr_fixedpt / prob_blockedby_ldr

        return (meantime_blockedbyldr_fixedpt, ldr_effmean_svctime_final,
                prob_blockedby_ldr, condmeantime_blockedbyldr)

    except (RuntimeError, ValueError) as e:
        print(e)
        rho_init = arr_rate * ldr_effmean_svctime_init / ldr_cap
        print(f'arr_rate={arr_rate}, ldr_effmean_svctime_init={ldr_effmean_svctime_init}, '
              f'ldr_cap={int(ldr_cap)}, rho_init={rho_init}')

        meantime_blockedbyldr_init = qng.mgc_mean_qwait_kimura(arr_rate, 1.0 /ldr_effmean_svctime_init, int(ldr_cap), ldr_cv2_svctime)

        # Now update the estimate of effective svc time in LDR
        # Oops! Following line double adds the PP part
        # ldr_effmean_svctime_final = ldr_effmean_svctime_init + (1 - csect_rate) * ldr_meantime_blockedby_pp - meantime_blockedbyldr_fixedpt


        # Finally, compute estimate of prob blocked in obs and conditional meantime blocked
        prob_blockedby_ldr = qng.mgc_prob_wait_erlangc(arr_rate, 1.0 / ldr_effmean_svctime_init, int(ldr_cap))
        condmeantime_blockedbyldr = np.infty

        return (meantime_blockedbyldr_init, ldr_effmean_svctime_init,
                prob_blockedby_ldr, meantime_blockedbyldr_init / prob_blockedby_ldr)




if __name__ == '__main__':

    train_df = pd.read_csv('mmdata/train_exp9_tandem05_nodischadj.csv')
    # train_df.set_index('scenario', drop=False, inplace=True)

    results = []
    scenarios = range(1, 151)

    for scenario_ in scenarios:
        arr_rate_ = train_df.ix[scenario_ - 1, 'lam_obs']
        csect_rate_ = train_df.ix[scenario_ - 1, 'tot_c_rate']
        ldr_mean_svctime_ = train_df.ix[scenario_ - 1, 'alos_ldr']
        ldr_cv2_svctime_ = train_df.ix[scenario_ - 1, 'cv2_ldr']
        # ldr_cv2_svctime = train_df.ix[scenario - 1, 'actual_los_cv2_mean_ldr']
        ldr_cap_ = train_df.ix[scenario_ - 1, 'cap_ldr']
        pp_mean_svctime_ = train_df.ix[scenario_ - 1, 'alos_pp']
        pp_cv2_svctime_ = train_df.ix[scenario_ - 1, 'cv2_pp']
        pp_cap_ = train_df.ix[scenario_ - 1, 'cap_pp']
        sim_mean_waitq_ldr_mean_ = train_df.ix[scenario_ - 1, 'mean_waitq_ldr_mean']
        sim_mean_pct_waitq_ldr_ = train_df.ix[scenario_ - 1, 'mean_pct_waitq_ldr']
        sim_actual_los_mean_mean_ldr_ = train_df.ix[scenario_ - 1, 'actual_los_mean_mean_ldr']
        sim_mean_pct_blocked_by_pp_ = train_df.ix[scenario_ - 1, 'mean_pct_blocked_by_pp']
        sim_mean_blocked_by_pp_mean_ = train_df.ix[scenario_ - 1, 'mean_blocked_by_pp_mean']

        pct_blockedby_pp_ = prob_blockedby_pp_hat(arr_rate_, pp_mean_svctime_, pp_cap_, pp_cv2_svctime_)
        ldr_meantime_blockedby_pp_ = condmeantime_blockedby_pp_hat(arr_rate_, pp_mean_svctime_, pp_cap_,
                                                                   pp_cv2_svctime_)
        (obs_meantime_blockedbyldr_, ldr_effmean_svctime, obs_prob_blockedby_ldr, obs_condmeantime_blockedbyldr) = \
            obs_blockedby_ldr_hats(arr_rate_, csect_rate_, ldr_mean_svctime_, ldr_cv2_svctime_, ldr_cap_,
                                   pp_mean_svctime_, pp_cv2_svctime_, pp_cap_)

        scen_results = {'scenario': scenario_,
                        'arr_rate': arr_rate_,
                        'prob_blockedby_ldr_approx': obs_prob_blockedby_ldr,
                        'prob_blockedby_ldr_sim': sim_mean_pct_waitq_ldr_,

                        'condmeantime_blockedbyldr_approx': obs_condmeantime_blockedbyldr * 24.0,
                        'condmeantime_blockedbyldr_sim': sim_mean_waitq_ldr_mean_,
                        'ldr_effmean_svctime_approx': ldr_effmean_svctime * 24.0,
                        'ldr_effmean_svctime_sim': sim_actual_los_mean_mean_ldr_,
                        'prob_blockedby_pp_approx': pct_blockedby_pp_,
                        'prob_blockedby_pp_sim': sim_mean_pct_blocked_by_pp_,
                        'condmeantime_blockedbypp_approx': ldr_meantime_blockedby_pp_ * 24.0,
                        'condmeantime_blockedbypp_sim': sim_mean_blocked_by_pp_mean_}

        results.append(scen_results)

        # print("scenario {}\n".format(scenario))
        # print(results)

    results_df = pd.DataFrame(results)
    print(results_df)

    # results_df.to_csv("obnetwork_approx_vs_sim.csv")
    results_df.to_csv("obnetwork_approx_vs_sim_testing.csv")
