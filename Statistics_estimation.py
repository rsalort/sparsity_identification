#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:00:33 2026

@author: rodrigosalort
"""

#%% Global configuration

import numpy as np

d = 2 ** 18         
delta = 0.1         
sigma = 0.1

target_quantiles = [0.01, 0.05, 0.10, 0.25]
x_indices = np.arange(0, d)
x_max = d - 1


distributions = {}

# Measurements from single-run realization 

p_uniform = np.ones(d) / d
distributions['Dense Uniform distribution'] = {'p_true': p_uniform, 'M': 15000}

x_gauss = np.linspace(-3, 3, d)
p_gaussian = np.exp(-x_gauss ** 2 / (2 * sigma ** 2))
p_gaussian /= np.sum(p_gaussian)
distributions['Dense Gaussian distribution'] = {'p_true': p_gaussian, 'M': 15000}

p_flat = np.zeros(d)
p_flat[d // 2 - 50: d // 2 + 50] = 1.0 / 100
distributions['Uniform sparse (100 states)'] = {'p_true': p_flat, 'M': 12767}


p_gaussian_s = np.zeros(d)
x_block = np.linspace(-3, 3, 300)
p_block = np.exp(-x_block ** 2 / 2)
p_gaussian_s[d // 2 - 150: d // 2 + 150] = p_block / np.sum(p_block)
distributions['Gaussian Sparse (300 states)'] = {'p_true': p_gaussian_s, 'M': 13008}


# For moments 
def sample_and_count(p_true, M):
    """
    Return the raw counts per state
    """
    samples = np.random.choice(d, size=M, p=p_true)
    return np.bincount(samples, minlength=d)


def compute_moments(p, x=x_indices):
    """
    Return the first and second raw moments and the variance of a distribution p.
    """
    m1 = np.sum(x * p)
    m2 = np.sum((x ** 2) * p)
    var = m2 - m1 ** 2
    return m1, m2, var


def theoretical_moment_bounds(epsilon):
    """
    Return the theoretical error bounds on the 1st and 2nd raw moments 
    for a given epsilon.
    """
    error_m1_theo = x_max * epsilon
    error_m2_theo = (x_max ** 2) * epsilon
    return error_m1_theo, error_m2_theo


def print_moment_comparison(name, M, epsilon, true_moments, est_moments,
                             error_m1_theo, error_m2_theo, var_theo, extra=""):
    """Print results"""
    m1_true, m2_true, var_true = true_moments
    m1_est, m2_est, var_est = est_moments
    
    error_m1 = np.abs(m1_true - m1_est)/m1_true * 100
    error_m2 = np.abs(m2_true - m2_est)/m2_true * 100
    error_var = np.abs(var_true - var_est)/var_true * 100
    
    if epsilon != 0: 
        print(f"\n\U0001F539 {name.upper()} (M = {M})")
        print("-" * 105)
        print(f"     Raw Moment k=1 | True: {m1_true:14.2f} | Est.: {m1_est:14.2f} | Rel. Emp. Err: {error_m1:12.7f} | Theo. Bound: {error_m1_theo:12.2f}")
        print(f"     Raw Moment k=2 | True: {m2_true:14.4e} | Est.: {m2_est:14.4e} | Rel. Emp. Err: {error_m2:12.4e} | Theo. Bound: {error_m2_theo:12.4e}")
        print(f"     Variance       | True: {var_true:14.4e} | Est.: {var_est:14.4e} | Rel. Emp. Err: {error_var:12.4e} | Theo. Bound: {var_theo:12.4e}")
        print("-" * 105)
    else:
        print(f"\n\U0001F539 {name.upper()} (M = {M})")
        print("-" * 105)
        print(f"     Raw Moment k=1 | True: {m1_true:14.2f} | Est.: {m1_est:14.2f} | Rel. Emp. Err: {error_m1:12.7f}")
        print(f"     Raw Moment k=2 | True: {m2_true:14.4e} | Est.: {m2_est:14.4e} | Rel. Emp. Err: {error_m2:12.4e} ")
        print(f"     Variance       | True: {var_true:14.4e} | Est.: {var_est:14.4e} | Rel. Emp. Err: {error_var:12.4e} ")
        print("-" * 105)
        


# For quantiles
def get_quantile(cdf, q, domain):
    """
    Invert the CDF
    """
    idx = np.searchsorted(cdf, q)
    idx = np.clip(idx, 0, len(domain) - 1)
    return domain[idx]


def print_quantile_comparison(name, M, epsilon, cdf_true, cdf_est, extra="", q_max_cap=1.0):
    """
    Print results
    """
    print(f"\n\U0001F539 {name.upper()} (M = {M})")
    if epsilon != 0: 
        print(f"{'Quantile (q)':<15} | {'True Q(q)':<12} | {'Est. Q_hat(q)':<15} | {'Confidence Interval [Q_min, Q_max]':<40}")
    else:
        print(f"{'Quantile (q)':<15} | {'True Q(q)':<12} | {'Est. Q_hat(q)':<15} | {'Relative Error (%)':<40}")
    print("-" * 90)

    for q in target_quantiles:
        q_min = max(0.0, q - epsilon)
        q_max = min(q_max_cap, q + epsilon)

        q_true = get_quantile(cdf_true, q, x_indices)
        q_est = get_quantile(cdf_est, q, x_indices)
        
        if epsilon != 0: 
            lower_bound = get_quantile(cdf_est, q_min, x_indices)
            upper_bound = get_quantile(cdf_est, q_max, x_indices)
            print(f"q = {q:<10.2f} | {q_true:<12.0f} | {q_est:<15.0f} | [{lower_bound:.0f}, {upper_bound:.0f}]")
        else:
            rel_e = abs(q_est - q_true)/q_true *100
            print(f"q = {q:<10.2f} | {q_true:<12.0f} | {q_est:<15.0f} | {rel_e:.6e}")
    print("=" * 90)


#%% Means and quantiles via MLE

np.random.seed(42)

print("=" * 95)
print("MLE ESTIMATOR")
print("=" * 95)

for name, dist in distributions.items():
    p_true = dist['p_true']
    M = dist['M']

    counts = sample_and_count(p_true, M)
    mle_p = counts / M

    epsilon = np.sqrt(np.log(2 * M * (M + 1) / delta) / (2 * M))  # DKW-type bound

    true_moments = compute_moments(p_true)
    est_moments = compute_moments(mle_p)
    error_m1_theo, error_m2_theo = theoretical_moment_bounds(epsilon)
    var_theo = 3 * error_m2_theo

    print_moment_comparison(name, M, epsilon, true_moments, est_moments,
                             error_m1_theo, error_m2_theo, var_theo)

    cdf_true = np.cumsum(p_true)
    cdf_est = np.cumsum(mle_p)
    print_quantile_comparison(name, M, epsilon, cdf_true, cdf_est)

print("=" * 95)


#%% Means and quantiles via Raw GT

np.random.seed(42)

print("=" * 95)
print("RAW GT")
print("=" * 95)

epsilon = 0

for name, dist in distributions.items():
    p_true = dist['p_true']
    M = dist['M']

    counts = sample_and_count(p_true, M)
    mle_p = counts / M

    n1 = int(np.sum(counts == 1))
    S_size = int(np.sum(counts > 0))
    G0 = n1 / M  # empirical missing mass

    gt_p = (1.0 - G0) * mle_p  # raw rescaling; unobserved states remain at 0

    extra = f", |S| = {S_size}, n1 = {n1}, G0 = {G0:.4f}"

    true_moments = compute_moments(p_true)
    est_moments = compute_moments(gt_p)

    print_moment_comparison(name, M, epsilon, true_moments, est_moments,
                             error_m1_theo, error_m2_theo, var_theo, extra)

    cdf_true = np.cumsum(p_true)
    cdf_est = np.cumsum(gt_p)
    print_quantile_comparison(name, M, epsilon, cdf_true, cdf_est, extra, q_max_cap=1.0 - G0)

print("=" * 95)


#%% Means and quantiles via Uniform GT

np.random.seed(42)

print("=" * 95)
print("UNIFORM GT")
print("=" * 95)

epsilon = 0

for name, dist in distributions.items():
    p_true = dist['p_true']
    M = dist['M']

    counts = sample_and_count(p_true, M)

    observed_mask = counts > 0
    S_size = int(np.sum(observed_mask))
    n1 = int(np.sum(counts == 1))
    n0 = d - S_size
    G0 = n1 / M  

    gt_p = np.zeros(d, dtype=float)
    if n0 > 0:
        gt_p[observed_mask] = (1.0 - G0) * (counts[observed_mask] / M)
        gt_p[~observed_mask] = G0 / n0 
    else:
        gt_p[observed_mask] = counts[observed_mask] / M


    extra = f", |S| = {S_size}, n1 = {n1}, G0 = {G0:.4f}"

    true_moments = compute_moments(p_true)
    est_moments = compute_moments(gt_p)
    
    print_moment_comparison(name, M, epsilon, true_moments, est_moments,
                             error_m1_theo, error_m2_theo, var_theo, extra)

    cdf_true = np.cumsum(p_true)
    cdf_est = np.cumsum(gt_p)
    print_quantile_comparison(name, M, epsilon, cdf_true, cdf_est, extra)

print("=" * 95)
