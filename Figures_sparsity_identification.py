#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Sparsity Identification and Information Extraction for Quantum States

Simulation and figures

@author: rodrigosalort
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



#%% Fig: 5.1 - Distributions

np.random.seed(42)


d = 2**18
indices = np.arange(0, d)
sigma = 0.1


# 1. Uniform distribution
p_uniform = np.ones(d) / d

plt.figure(figsize=(6, 3.7))
plt.step(indices, p_uniform, where='mid', color='blue')
plt.fill_between(indices, 0, p_uniform, step='mid',
                 color='blue', alpha=.3)
plt.title('Uniform distribution')
plt.xlabel('Basis state index, $i$')
plt.ylabel('Probability, $p_i$')
plt.grid(axis='y', ls='--', alpha=.5)
plt.xlim(.5, d + .5)
plt.ylim(0, 1.5/d)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.show()


# 2. Gaussian distribution
x = np.linspace(-3, 3, d)
p_normal = np.exp(-x**2/(2*sigma**2))
p_normal /= p_normal.sum()

plt.figure(figsize=(6, 3.7))
plt.plot(indices, p_normal, color='orange', lw=1.5)
plt.fill_between(indices, p_normal, color='orange', alpha=.4)
plt.title('Gaussian distribution')
plt.xlabel('Basis state index, $i$')
plt.ylabel('Probability, $p_i$')
plt.grid(axis='y', ls='--', alpha=.5)
plt.xlim(1, d)
plt.ylim(0, 1.1*p_normal.max())
plt.tight_layout()
plt.show()


# 3. Sparse Gaussian distribution
n_sparse_var = 300
x_block = np.linspace(-3, 3, n_sparse_var)
p_block = np.exp(-x_block**2/2)
p_block /= p_block.sum()

centre = d // 2

p_sparse_var = np.zeros(d)
start = centre - n_sparse_var // 2
end_d = start + n_sparse_var
p_sparse_var[start:end_d] = p_block

fig, ax = plt.subplots(figsize=(6, 3.7))

zoom_xlim = (130900, 131250)
zoom_ylim = (0, 1.1*p_sparse_var.max())

ax.plot(indices, p_sparse_var, color='red', lw=1.2)
ax.fill_between(indices, p_sparse_var, color='red', alpha=.4)

ax.set_title('Sparse distribution (300 states)', fontsize=14)
ax.set_xlabel('Basis state index, $i$')
ax.set_ylabel('Probability, $p_i$')
ax.grid(axis='y', ls='--', alpha=.5)
ax.set_xlim(.5, .5+d)
ax.set_ylim(zoom_ylim)

ax.add_patch(Rectangle(
    (zoom_xlim[0], 0),
    zoom_xlim[1] - zoom_xlim[0],
    zoom_ylim[1],
    lw=1, edgecolor='black', facecolor='none', alpha=.5))

axins = ax.inset_axes([.65, .5, .25, .35])
axins.plot(indices, p_sparse_var, color='red', lw=1.2)
axins.fill_between(indices, p_sparse_var, color='red', alpha=.4)
axins.set_xlim(zoom_xlim)
axins.set_ylim(zoom_ylim)
axins.set_title('Zoom (130900 - 131250)', fontsize=10)
axins.set_xticks(zoom_xlim)
axins.grid(axis='y', ls='--', alpha=.5)

for y in zoom_ylim:
    fig.add_artist(ConnectionPatch(
        xyA=(zoom_xlim[1], y),
        xyB=(zoom_xlim[0], y),
        coordsA='data', coordsB='data',
        axesA=ax, axesB=axins,
        color='black', lw=1, alpha=.6))

plt.tight_layout()
plt.show()

# 4. Highly sparse uniform distribution
n_sparse_flat = 100
p_sparse_flat = np.zeros(d)

start = centre - n_sparse_flat // 2
end_d = start + n_sparse_flat

p_sparse_flat[start:end_d] = 1 / n_sparse_flat

fig, ax = plt.subplots(figsize=(6, 3.7))

zoom_xlim = (131000, 131150)
zoom_ylim = (0, 1.1*p_sparse_flat.max())

ax.plot(indices, p_sparse_flat, color='green', lw=1)
ax.fill_between(indices, p_sparse_flat, color='green', alpha=.4)

ax.set_title('Highly sparse uniform distribution (100 states)',
             fontsize=14)
ax.set_xlabel('Basis state index, $i$')
ax.set_ylabel('Probability, $p_i$')
ax.grid(axis='y', ls='--', alpha=.5)
ax.set_xlim(.5, .5+d)
ax.set_ylim(zoom_ylim)

ax.add_patch(Rectangle(
    (zoom_xlim[0], 0),
    zoom_xlim[1] - zoom_xlim[0],
    zoom_ylim[1],
    lw=1, edgecolor='black', facecolor='none', alpha=.5))

axins = ax.inset_axes([.65, .5, .25, .35])
axins.plot(indices, p_sparse_flat, color='green', lw=1)
axins.fill_between(indices, p_sparse_flat, color='green', alpha=.4)
axins.set_xlim(zoom_xlim)
axins.set_ylim(zoom_ylim)
axins.set_title('Zoom (131000 - 131150)', fontsize=10)
axins.set_xticks(zoom_xlim)
axins.grid(axis='y', ls='--', alpha=.5)

for y in zoom_ylim:
    fig.add_artist(ConnectionPatch(
        xyA=(zoom_xlim[1], y),
        xyB=(zoom_xlim[0], y),
        coordsA='data', coordsB='data',
        axesA=ax, axesB=axins,
        color='black', lw=1, alpha=.6))

plt.tight_layout()
plt.show()


#%% Fig: 5.2 - Evolution of the identified support size

np.random.seed(42)

M_max = 15000

def accumulation_sim(p, shots):
    samples = np.random.choice(d, size=shots, p=p)
    _, primeros = np.unique(samples, return_index=True)
    nuevos = np.zeros(shots)
    nuevos[primeros] = 1
    return np.cumsum(nuevos)

acum_uniform = accumulation_sim(p_uniform, M_max)
acum_normal = accumulation_sim(p_normal, M_max)
acum_sparse_var = accumulation_sim(p_sparse_var, M_max)
acum_sparse_flat = accumulation_sim(p_sparse_flat, M_max)

x_measurements = np.arange(1, M_max + 1)
x_percentage = x_measurements / d * 100

fig, ax = plt.subplots(figsize=(12, 7.4))

ax.plot(x_percentage, acum_uniform,
        label='Uniform distribution', color='blue', lw=2.5)
ax.plot(x_percentage, acum_normal,
        label='Gaussian distribution', color='orange', lw=2.5)
ax.plot(x_percentage, acum_sparse_var,
        label='Sparse Gaussian distribution (300 states)', color='red', lw=2.5)
ax.plot(x_percentage, acum_sparse_flat,
        label='Highly sparse uniform distribution ($100$ states)', color='green', lw=2.5)

ax.set_title('Evolution of the identified support size', fontsize=22)
ax.set_xlabel(r'$M/d \ (\%)$', fontsize=18)
ax.set_ylabel(r'Number of distinct basis states observed, $r_{obs}$', fontsize=18)
ax.grid(True, linestyle='--', alpha=.5)
ax.tick_params(axis='both', labelsize=16)
ax.set_xlim(0, M_max / d * 100)
ax.set_ylim(0, M_max)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))

# Zoom
ax_inset = inset_axes(
    ax, width="100%", height="100%",
    bbox_to_anchor=(.6, .1, .32, .32),
    bbox_transform=ax.transAxes,
    loc='lower left', borderpad=0
)

ax_inset.plot(x_percentage, acum_sparse_var, color='red', lw=2.5)
ax_inset.plot(x_percentage, acum_sparse_flat, color='green', lw=2.5)

ax.legend(
    loc='upper left',
    fontsize=14,
    frameon=False
)

ax_inset.set_title('Sparse distributions zoom', fontsize=14, pad=6)
ax_inset.set_xlim(0, M_max / d * 100)
ax_inset.set_ylim(0, 350)
ax_inset.grid(True, linestyle='--', alpha=.4)
ax_inset.tick_params(axis='both', labelsize=11)
ax_inset.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))

mark_inset(
    ax, ax_inset,
    loc1=2, loc2=4,
    fc="none", ec="0.4",
    linestyle="--", alpha=.7
)

plt.tight_layout()
plt.show()

#%% Fig: 5.3 - Evolution of states observed exactly once

def singletons_sim(p, shots):
    samples = np.random.choice(d, size=shots, p=p)
    counts = np.zeros(d, dtype=int)
    singletons = np.zeros(shots, dtype=int)

    n_singletons = 0

    for t, state in enumerate(samples):
        if counts[state] == 0:
            n_singletons += 1
        elif counts[state] == 1:
            n_singletons -= 1

        counts[state] += 1
        singletons[t] = n_singletons

    return singletons

np.random.seed(42)
singletons_uniform = singletons_sim(p_uniform, M_max)
singletons_normal = singletons_sim(p_normal, M_max)
singletons_sparse_flat = singletons_sim(p_sparse_flat, M_max)
singletons_sparse_var = singletons_sim(p_sparse_var, M_max)


# Broken axis
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(12, 7.4),
    gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.08}
)

ax_top.set_facecolor('#f7f9fb')   
ax_bot.set_facecolor('#fdfaf4')  

for ax in (ax_top, ax_bot):
    ax.plot(x_percentage, singletons_uniform,
            label='Uniform distribution', color='blue', lw=2.2)
    ax.plot(x_percentage, singletons_normal,
            label='Gaussian distribution', color='orange', lw=2.2)
    ax.plot(x_percentage, singletons_sparse_var,
            label='Sparse Gaussian distribution (300 states)',
            color='red', lw=2.2)
    ax.plot(x_percentage, singletons_sparse_flat,
            label='Highly sparse uniform distribution ($100$ states)',
            color='green', lw=2.2)

    ax.grid(True, linestyle='--', alpha=.5)
    ax.tick_params(axis='both', labelsize=14)

ax_top.set_ylim(1000, 12000)
ax_bot.set_ylim(0, 100)
ax_bot.set_xlim(0, M_max / d * 100)

ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)

ax_top.tick_params(labeltop=False, top=False)
ax_bot.xaxis.set_major_formatter(
    mtick.PercentFormatter(decimals=2)
)

# Diagonal marks
d_len = .012
kwargs = dict(
    transform=ax_top.transAxes,
    color='k', clip_on=False, lw=1.2
)

ax_top.plot((-d_len, d_len), (-d_len, d_len), **kwargs)
ax_top.plot((1-d_len, 1+d_len), (-d_len, d_len), **kwargs)

kwargs.update(transform=ax_bot.transAxes)

ax_bot.plot(
    (-d_len, d_len), (1-d_len, 1+d_len), **kwargs
)
ax_bot.plot(
    (1-d_len, 1+d_len), (1-d_len, 1+d_len), **kwargs
)

ax_top.set_title(
    'Evolution of states observed exactly once',
    fontsize=22, pad=12
)

ax_bot.set_xlabel(r'$M/d \ (\%)$', fontsize=18)

fig.text(
    0.02, 0.5,
    r'Number of states observed once, $n_{1}$',
    va='center', rotation='vertical', fontsize=18
)

ax_top.legend(
    fontsize=14, loc='upper left',
    frameon=False
)

plt.subplots_adjust(
    left=.12, right=.95, top=.92, bottom=.1
)
plt.show()


#%% Fig: 5.4 - Evolution of the empirical Good-Turing estimate

np.random.seed(42)

gt_uniform = singletons_uniform / x_measurements
gt_normal = singletons_normal / x_measurements
gt_sparse_flat = singletons_sparse_flat / x_measurements
gt_sparse_var = singletons_sparse_var / x_measurements



fig, ax = plt.subplots(figsize=(12, 7.4))

ax.plot(x_percentage, gt_uniform,
        label='Uniform distribution', color='blue', lw=2.5)
ax.plot(x_percentage, gt_normal,
        label='Gaussian distribution', color='orange', lw=2.5)
ax.plot(x_percentage, gt_sparse_var,
        label='Sparse Gaussian distribution (300 states)',
        color='red', lw=2.5)
ax.plot(x_percentage, gt_sparse_flat,
        label='Highly sparse uniform distribution ($100$ states)',
        color='green', lw=2.5)

ax.set_title('Evolution of the empirical Good-Turing estimate',
             fontsize=22)
ax.set_xlabel(r'$M/d \ (\%)$', fontsize=18)
ax.set_ylabel('Estimated missing mass, $G_{0}$', fontsize=18)

ax.grid(True, linestyle='--', alpha=.5)
ax.legend(fontsize=16, loc='lower right', frameon=False)
ax.tick_params(axis='both', labelsize=16)

ax.set_xlim(0, M_max / d * 100)
ax.set_ylim(0, 1.05)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))

plt.tight_layout()
plt.show()

#%% Fig: 5.5 - Missing mass upper bounds across distributions
np.random.seed(42)

epsilon = 0.1
delta = 0.1


def plot_bounds(ax, M, M_percentage, gt, delta, color, label):
    bound_BK = (gt + (1 + np.sqrt(2)) 
                * np.sqrt(np.log(2 * M * (M + 1) / delta) / M))

    bound_MS = (gt + (2 * np.sqrt(2) + np.sqrt(3))
                * np.sqrt(np.log(3 * M * (M + 1) / delta) / M))

    ax.plot( 
        M_percentage, bound_BK,
        color=color, lw=2.2,
        linestyle='-', alpha=.85,
        label=label
    )

    ax.plot(
        M_percentage, bound_MS,
        color=color, lw=2.2,
        linestyle='--', alpha=.25
    )


fig, ax = plt.subplots(figsize=(16, 7.4))

plot_bounds(
    ax, x_measurements, x_percentage,
    gt_uniform, delta, 'blue', 'Uniform'
)

plot_bounds(
    ax, x_measurements, x_percentage,
    gt_normal, delta, 'orange', 'Gaussian'
)

plot_bounds(
    ax, x_measurements, x_percentage,
    gt_sparse_var, delta, 'red',
    'Sparse Gaussian (300 states)'
)

plot_bounds(
    ax, x_measurements, x_percentage,
    gt_sparse_flat, delta, 'green',
    'Highly sparse uniform ($100$ states)'
)

ax.axhline(
    y=epsilon,
    color='black',
    linestyle='-',
    lw=2
)


# Intersactions

bound_BK_flat = (gt_sparse_flat + (1 + np.sqrt(2))
                 * np.sqrt(np.log(
                     2 * x_measurements * (x_measurements + 1) / delta)
                     / x_measurements)
                 )

bound_BK_var = (gt_sparse_var + (1 + np.sqrt(2))
                * np.sqrt(np.log(
                    2 * x_measurements * (x_measurements + 1) / delta)
                    / x_measurements)
                )

idx_flat = np.where(bound_BK_flat <= epsilon)[0]
idx_var = np.where(bound_BK_var <= epsilon)[0]


# Highly-sparse

if len(idx_flat) > 0:

    M_flat = idx_flat[0]
    pct_flat = x_percentage[idx_flat[0]]

    ax.plot(
        pct_flat, epsilon,
        'o',
        color='darkgreen',
        markersize=9,
        zorder=5
    )

    ax.annotate(
        f'M={M_flat} ({pct_flat:.2f}%)',
        xy=(pct_flat, epsilon),
        xytext=(pct_flat - .7, .25),
        arrowprops=dict(
            facecolor='black',
            shrink=.08,
            width=1,
            headwidth=6
        ),
        fontsize=16,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=.3',
            fc='white',
            ec='green',
            alpha=.8
        )
    )


# Gaussian Sparse

if len(idx_var) > 0:

    M_var = idx_var[0]
    pct_var = x_percentage[idx_var[0]]

    ax.plot(
        pct_var, epsilon,
        'o',
        color='darkred',
        markersize=9,
        zorder=5
    )

    ax.annotate(
        f'M={M_var} ({pct_var:.2f}%)',
        xy=(pct_var, epsilon),
        xytext=(pct_var + .6, -.1),
        arrowprops=dict(
            facecolor='black',
            shrink=.08,
            width=1,
            headwidth=6
        ),
        fontsize=16,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=.3',
            fc='white',
            ec='red',
            alpha=.8
        )
    )


# Plot

ax.set_title(
    r'Missing mass upper bounds across distributions'
    r'($\delta=0.1$)',
    fontsize=22
)

ax.set_xlabel(
    r'$M/d \ (\%)$',
    fontsize=18
)

ax.set_ylabel(
    'Missing mass upper bound',
    fontsize=18
)

ax.grid(
    True,
    linestyle='--',
    alpha=.5
)

ax.tick_params(
    axis='both',
    labelsize=16
)

ax.set_xlim(
    0,
    M_max / d * 100
)

ax.set_ylim(0, 1)

ax.xaxis.set_major_formatter(
    mtick.PercentFormatter(decimals=2)
)


handles, labels = ax.get_legend_handles_labels()

bk_line = Line2D(
    [0], [0],
    color='gray',
    lw=2.2,
    linestyle='-'
)

ms_line = Line2D(
    [0], [0],
    color='gray',
    lw=2.2,
    linestyle='--',
    alpha=.7
)

threshold_line = Line2D(
    [0], [0],
    color='black',
    lw=2
)

handles.extend(
    [bk_line, ms_line, threshold_line]
)

labels.extend([
    r'$\tilde{M}_{0}^{\mathrm{BK}}$ bound (solid lines)',
    r'$\tilde{M}_{0}^{\mathrm{MS}}$ bound (dashed lines)',
    r'Threshold ($\varepsilon=0.10$)'
])

ax.legend(
    handles,
    labels,
    fontsize=16,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    frameon=False
)


# Zoom

zoom_xlim = (
    12600 / d * 100,
    13200 / d * 100
)

zoom_ylim = (
    epsilon - .002,
    epsilon + .002
)

ax.add_patch(
    Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        linewidth=1,
        edgecolor='black',
        facecolor='gray',
        alpha=.2
    )
)

axins = ax.inset_axes(
    [1.1, .15, .35, .35]
)

plot_bounds(
    axins, x_measurements, x_percentage,
    gt_uniform, delta, 'blue', ''
)

plot_bounds(
    axins, x_measurements, x_percentage,
    gt_normal, delta, 'orange', ''
)

plot_bounds(
    axins, x_measurements, x_percentage,
    gt_sparse_var, delta, 'red', ''
)

plot_bounds(
    axins, x_measurements, x_percentage,
    gt_sparse_flat, delta, 'green', ''
)

axins.axhline(
    y=epsilon,
    color='black',
    linestyle='-',
    lw=2
)

if len(idx_flat) > 0:
    axins.plot(
        pct_flat, epsilon,
        'o',
        color='darkgreen',
        markersize=9,
        zorder=5
    )

if len(idx_var) > 0:
    axins.plot(
        pct_var, epsilon,
        'o',
        color='darkred',
        markersize=9,
        zorder=5
    )

axins.set_xlim(zoom_xlim)
axins.set_ylim(zoom_ylim)

axins.set_title(
    r'Zoom ($M \in [12600,13200]$)',
    fontsize=16
)

axins.grid(
    axis='both',
    linestyle='--',
    alpha=.5
)

axins.xaxis.set_major_formatter(
    mtick.PercentFormatter(decimals=2)
)

axins.tick_params(
    axis='both',
    labelsize=14
)

con1 = ConnectionPatch(
    xyA=(zoom_xlim[0], zoom_ylim[1]),
    xyB=(zoom_xlim[0], zoom_ylim[0]),
    coordsA='data',
    coordsB='data',
    axesA=ax,
    axesB=axins,
    color='black',
    lw=1,
    alpha=.6
)

con2 = ConnectionPatch(
    xyA=(zoom_xlim[1], zoom_ylim[1]),
    xyB=(zoom_xlim[1], zoom_ylim[0]),
    coordsA='data',
    coordsB='data',
    axesA=ax,
    axesB=axins,
    color='black',
    lw=1,
    alpha=.6
)

fig.add_artist(con1)
fig.add_artist(con2)

plt.tight_layout()
plt.show()


#%% Fig: 5.6 - Measurements required for ...

np.random.seed(42)

max_M = int(2 * d)
M_vals = np.arange(1, max_M + 1)
penalty_BK = (
    1 + np.sqrt(2)) * np.sqrt(
        np.log(2 * M_vals * (M_vals + 1) / delta) / M_vals
        )
  
sparsity_log = np.logspace(0, np.log10(d / 2), 150).astype(int)
 
r_min_zoom = int(0.08 * d)
r_max_zoom = int(0.095 * d)
sparsity_corte = np.linspace(r_min_zoom, r_max_zoom, 10).astype(int)
 

sparsity_vals = np.unique(np.concatenate([sparsity_log, 
                                          sparsity_corte, [int(d / 2)]]))
sparsity_vals = np.sort(sparsity_vals)
 
if d / 2 not in sparsity_vals:
    sparsity_vals = np.append(sparsity_vals, int(d / 2))  
    sparsity_vals = np.sort(sparsity_vals)
 

def good_turing_simulation(p_dist, shots, dimension):
    samples = np.random.choice(dimension, size=shots, p=p_dist)
    states_count = np.zeros(dimension, dtype=int)
    singletons_ev = np.zeros(shots, dtype=int)
 
    counter = 0
    for t in range(shots):
        state_meas = samples[t]
        if states_count[state_meas] == 0:
            counter += 1
        elif states_count[state_meas] == 1:
            counter -= 1
        states_count[state_meas] += 1
        singletons_ev[t] = counter
 
    ejex_M = np.arange(1, shots + 1)
    return singletons_ev / ejex_M
 
M_intersection_uniform = []
sparsity_uniform = []
 
M_inters_gaussian = []
sparsity_gaussian = []
 
 
for r in sparsity_vals:
    #  Uniform
    p_unif = np.zeros(d)
    p_unif[:r] = 1.0 / r
    
    M_estimate_unif = 0
    if r > 1:
        E_G0_unif = np.power(1.0 - 1.0/r, M_vals)
        c_teo_unif = E_G0_unif + penalty_BK
        idx_t = np.where(c_teo_unif <= epsilon)[0]
        if len(idx_t) > 0:
            M_estimate_unif = M_vals[idx_t[0]]
    
    # Expected M according r       
    M_simulate_unif = int(M_estimate_unif * 1.2)
    if M_simulate_unif < 1000: M_simulate_unif = 1000 
    if M_simulate_unif > max_M: M_simulate_unif = max_M
    if r == 1: M_simulate_unif = 5000 
    
    gt_empirical_unif = good_turing_simulation(p_unif, M_simulate_unif, d)
    M_eje_sim_unif = np.arange(1, M_simulate_unif + 1)
    bound_BK_unif = gt_empirical_unif + (
        (1 + np.sqrt(2)) 
        * np.sqrt(
            np.log(
                2 * M_eje_sim_unif * 
                (M_eje_sim_unif +1) / delta) / M_eje_sim_unif
            )
        )
    
    idx_int_unif = np.where(bound_BK_unif <= epsilon)[0]
    if len(idx_int_unif) > 0:
        M_intersection_uniform.append(M_eje_sim_unif[idx_int_unif[0]])
        sparsity_uniform.append(r)
        
    # Gaussian
    p_norm = np.zeros(d)
    if r == 1:
        p_norm[:r] = 1.0
    else:
        x_block = np.linspace(-3, 3, r)
        p_block = np.exp(-x_block**2 / 2)
        p_norm[:r] = p_block / np.sum(p_block)
        
    # Expected M according r 
    M_simulate_norm = int(M_estimate_unif * 1.8)
    if M_simulate_norm < 1500: M_simulate_norm = 1500
    if M_simulate_norm > max_M: M_simulate_norm = max_M
    if r == 1: M_simulate_norm = 5000 
    
    gt_empirical_norm = good_turing_simulation(p_norm, M_simulate_norm, d)
    M_eje_sim_norm = np.arange(1, M_simulate_norm + 1)
    bound_BK_norm = gt_empirical_norm + (
        (1 + np.sqrt(2)) 
        * np.sqrt(
            np.log(2 * M_eje_sim_norm 
                   * (M_eje_sim_norm +1) / delta) / M_eje_sim_norm
            )
        )
    
    idx_int_norm = np.where(bound_BK_norm <= epsilon)[0]
    if len(idx_int_norm) > 0:
        M_inters_gaussian.append(M_eje_sim_norm[idx_int_norm[0]])
        sparsity_gaussian.append(r)
 
# Results
sparsity_uniform = np.array(sparsity_uniform)
M_intersection_uniform = np.array(M_intersection_uniform)
M_pct_uniforme = (M_intersection_uniform / d) * 100
 
sparsity_gaussian = np.array(sparsity_gaussian)
M_inters_gaussian = np.array(M_inters_gaussian)
M_pct_normal = (M_inters_gaussian / d) * 100
 
# Plot

r_pct_uniforme = (sparsity_uniform / d) * 100
r_pct_normal = (sparsity_gaussian / d) * 100
 

fig, ax = plt.subplots(figsize=(12, 7.4))
 
ax.plot(r_pct_uniforme, M_pct_uniforme, color='green', lw=1.5, marker='o', markersize=3.0, label='r-sparse uniform distribution')
ax.plot(r_pct_normal, M_pct_normal, color='red', lw=1.5, marker='s', markersize=3.0, label=r'r-sparse Gaussian distribution')
 
ax.set_title(r'Measurements required for $\tilde{M}_{0}^{BK} \leq 0.1$ vs system sparsity', fontsize=22)
ax.set_xlabel('Non-zero states ratio, $r/d$ ($\%$)', fontsize=18) 
ax.set_ylabel('$M/d \ (\%)$', fontsize=18)
 
ax.tick_params(axis='both', which='major', labelsize=16)
 
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0)) 
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
 
ax.set_xlim(0, 50) 
ax.legend(fontsize=16, loc='upper left', frameon = False)
 
# Zoom arriba-izquierda
zoom1_x_lim_pct = (500 / d) * 100
zoom1_xlim = (0, zoom1_x_lim_pct)
zoom1_ylim = (4.8, 5.2)
 
rect1 = Rectangle((zoom1_xlim[0], zoom1_ylim[0]), 
                  zoom1_xlim[1] - zoom1_xlim[0], 
                  zoom1_ylim[1] - zoom1_ylim[0], 
                  linewidth=1.0, edgecolor='black', facecolor='gray', alpha=0.2)
ax.add_patch(rect1)
 
# Location
axins1 = ax.inset_axes([0.08, 0.45, 0.30, 0.32])
 
axins1.plot(r_pct_uniforme, M_pct_uniforme, color='green', lw=1.5, marker='o', markersize=4.0)
axins1.plot(r_pct_normal, M_pct_normal, color='red', lw=1.5, marker='s', markersize=4.0)
 
axins1.set_xlim(zoom1_xlim)
axins1.set_ylim(zoom1_ylim)
axins1.set_title(r"Zoom ($r \in [0, 500]$)", fontsize=13)
axins1.tick_params(axis='both', which='major', labelsize=12)
axins1.grid(axis='both', linestyle='--', alpha=0.5)
axins1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
axins1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
 
con1 = ConnectionPatch(xyA=(zoom1_xlim[0], zoom1_ylim[1]), xyB=(zoom1_xlim[0], zoom1_ylim[0]),
                       coordsA="data", coordsB="data", axesA=ax, axesB=axins1, color="black", lw=1.0, alpha=0.4)
con2 = ConnectionPatch(xyA=(zoom1_xlim[1], zoom1_ylim[1]), xyB=(zoom1_xlim[1], zoom1_ylim[0]),
                       coordsA="data", coordsB="data", axesA=ax, axesB=axins1, color="black", lw=1.0, alpha=0.4)
fig.add_artist(con1)
fig.add_artist(con2)
 
# Zoom 2 

zoom2_xlim = (8, 9.5)
 
idx_in_window = np.where((r_pct_uniforme >= zoom2_xlim[0]) & (r_pct_uniforme <= zoom2_xlim[1]))[0]
if len(idx_in_window) > 0:
    y_min = min(np.min(M_pct_uniforme[idx_in_window]), np.min(M_pct_normal[idx_in_window]))
    y_max = max(np.max(M_pct_uniforme[idx_in_window]), np.max(M_pct_normal[idx_in_window]))
    zoom2_ylim = (y_min * 0.95, y_max * 1.05)
else:
    zoom2_ylim = (0, 10)
 
rect2 = Rectangle((zoom2_xlim[0], zoom2_ylim[0]), 
                  zoom2_xlim[1] - zoom2_xlim[0], 
                  zoom2_ylim[1] - zoom2_ylim[0], 
                  linewidth=1.0, edgecolor='black', facecolor='gray', alpha=0.2)
ax.add_patch(rect2)
 
# Location
axins2 = ax.inset_axes([0.60, 0.12, 0.30, 0.32])
 
axins2.plot(r_pct_uniforme, M_pct_uniforme, color='green', lw=1.5, marker='o', markersize=4.0)
axins2.plot(r_pct_normal, M_pct_normal, color='red', lw=1.5, marker='s', markersize=4.0)
 
# First intersection
min_len = min(len(r_pct_uniforme), len(r_pct_normal))
diff = M_pct_uniforme[:min_len] - M_pct_normal[:min_len]
cambios_signo = np.where(np.diff(np.sign(diff)) != 0)[0]
 
for idx in cambios_signo:
    x1 = r_pct_uniforme[idx]
    x2 = r_pct_uniforme[idx+1]
    if zoom2_xlim[0] <= x1 <= zoom2_xlim[1]:
        y1_u, y2_u = M_pct_uniforme[idx], M_pct_uniforme[idx+1]
        y1_n, y2_n = M_pct_normal[idx], M_pct_normal[idx+1]
        m_u = (y2_u - y1_u) / (x2 - x1)
        m_n = (y2_n - y1_n) / (x2 - x1)
        if m_u != m_n:
            x_cruce = x1 + (y1_n - y1_u) / (m_u - m_n)
            y_cruce = y1_u + m_u * (x_cruce - x1)
            axins2.axvline(x=x_cruce, color='orange', linestyle='--', lw=1.5, zorder=3)
            axins2.plot(x_cruce, y_cruce, marker='o', color='orange', markersize=6, zorder=4)
        break
 
axins2.set_xlim(zoom2_xlim)
axins2.set_ylim(zoom2_ylim)
axins2.set_title("Intersection Zoom", fontsize=13) 
axins2.grid(axis='both', linestyle='--', alpha=0.5)
axins2.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
axins2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
axins2.tick_params(axis='both', which='major', labelsize=12)
 

con3 = ConnectionPatch(xyA=(zoom2_xlim[0], zoom2_ylim[1]), xyB=(zoom2_xlim[0], zoom2_ylim[1]),
                       coordsA="data", coordsB="data", axesA=ax, axesB=axins2, color="black", lw=1.0, alpha=0.4)
con4 = ConnectionPatch(xyA=(zoom2_xlim[0], zoom2_ylim[0]), xyB=(zoom2_xlim[0], zoom2_ylim[0]),
                       coordsA="data", coordsB="data", axesA=ax, axesB=axins2, color="black", lw=1.0, alpha=0.4)
fig.add_artist(con3)
fig.add_artist(con4)
 
plt.tight_layout()
plt.show()


#%% Measurements over 100 experiments

num_runs = 100

term_BK = (1 + np.sqrt(2)) * np.sqrt(
    np.log(2 * x_measurements * (x_measurements + 1) / delta) / x_measurements
)

arr_meas_green = np.zeros(num_runs)
arr_pct_green = np.zeros(num_runs)
arr_meas_red = np.zeros(num_runs)
arr_pct_red = np.zeros(num_runs)

for run in range(num_runs):
    np.random.seed(run)
    
    _ = good_turing_simulation(p_uniform, M_max, d)
    _ = good_turing_simulation(p_normal, M_max, d)
    
    gt_sparse_flat = good_turing_simulation(p_sparse_flat, M_max, d)
    gt_sparse_var = good_turing_simulation(p_sparse_var, M_max, d)
    
    BK_bound_green = gt_sparse_flat + term_BK
    BK_bound_red = gt_sparse_var + term_BK

    idx_green = np.where(BK_bound_green <= epsilon)[0]
    idx_red = np.where(BK_bound_red <= epsilon)[0]

    arr_meas_green[run] = x_measurements[idx_green[0]] if len(idx_green) > 0 else np.nan 
    arr_pct_green[run] = x_percentage[idx_green[0]] if len(idx_green) > 0 else np.nan
    
    arr_meas_red[run] = x_measurements[idx_red[0]] if len(idx_red) > 0 else np.nan
    arr_pct_red[run] = x_percentage[idx_red[0]] if len(idx_red) > 0 else np.nan 

median_green = np.nanmedian(arr_meas_green)
median_red =  np.nanmedian(arr_meas_red)


def iqr(arr):
    median = np.median(arr)
    q25 = np.percentile(arr, 25)
    q1 = np.percentile(arr, 0.1)
    q75 = np.percentile(arr, 75)
    iqr = q75 - q25
    return median, q25, q75, iqr, q1

# Sparse Uniform 
med_g, q25_g, q75_g, iqr_g, q1a = iqr(arr_meas_green)
pct_med_g, pct_q25_g, pct_q75_g, _, _= iqr(arr_pct_green)

# Sparse Gaussian 
med_r, q25_r, q75_r, iqr_r, q1b = iqr(arr_meas_red)
pct_med_r, pct_q25_r, pct_q75_r, _, _ = iqr(arr_pct_red)

print(
    f"Green (100 states): Median = {med_g:.0f} shots [IQR: {iqr_g:.0f} | Q25: {q25_g:.0f}, Q75: {q75_g:.0f}] ({pct_med_g:.2f}%)"
)
print(
    f"Red (300 states):   Median = {med_r:.0f} shots [IQR: {iqr_r:.0f} | Q25: {q25_r:.0f}, Q75: {q75_r:.0f}] ({pct_med_r:.2f}%)"
)








