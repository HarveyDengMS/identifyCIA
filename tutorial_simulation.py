# This code simulates the data experiment for causal inference with CIA identification
from warnings import filterwarnings

# filterwarnings('ignore')

import pandas as pd
import numpy as np
from PCDK import PCDK, estimate
from causalinference import CausalModel
from time import time
import os

# some distribution and math function
norm = lambda n: np.random.normal(size=n)  # generating normal distribution
logit = lambda n: 1 / (1 + np.exp(-n))
effect_sigma = 0.
unif = lambda n: np.random.uniform(size=n)
binary = lambda n: (unif(np.shape(n)) <= n) + 0

# name
T = 't'
Y = 'y'

# output path
outputpath = 'results/simulation/'
os.makedirs(outputpath, exist_ok=True)

def collect(r):
    """
    collects the results from OLS and PSM
    :param r: the dict of the results
    :return:
    """
    r2 = {}
    for i_r in r:
        i_y = r[i_r]
        if i_y.__class__ == pd.Series:
            i_y = {
                'ATE': i_y['Coef.'],
                'SE': i_y['Std.Err.'],
                'time': i_y['time']
            }
        else:
            i_y = {
                'ATE': i_y['ate'],
                'SE': i_y['ate_se'],
                'time': i_y['time']
            }
        r2[i_r] = i_y

    return pd.DataFrame(r2)


def data_study1(n=10000, e=0.05, binary_t=True):
    # data1
    x1 = norm(n)
    x2 = norm(n)
    x3 = norm(n)
    t = x1 * 0.3 + x2 * 0.2 + norm(n) * e
    if binary_t:
        t = t > 0
    y = t * 0.3 + x1 * 0.1 + x3 * 0.2 + norm(n) * e
    d = {
        'x1': x1,
        'x2': x2,
        'x3': x3,
        T: t.astype(np.int),
        Y: y
    }

    return pd.DataFrame(d)


def study1_confounding(n=40000, e=0.05):
    print('running study 1')
    d = data_study1(n, e)
    covariates = [i for i in d if i.startswith('x')]

    # by graph model
    dag = PCDK(T, Y)
    G = dag.fit(d)
    backdoor = dag.find_cia()

    # OLS
    r1 = estimate(d, T, Y, backdoor)  # valid
    r2 = estimate(d, T, Y)  # lacking
    r3 = estimate(d, T, Y, covariates)  # redundant

    # PSM
    print("estimating ATE in study1 by PSM, which might take some minutes")
    # valid PSM
    start_time = time()
    cm = CausalModel(Y=d[Y].values,
                     D=d[T].values,
                     X=d[backdoor].values)
    cm.est_via_matching()
    r4 = dict(cm.estimates['matching'])
    r4['time'] = time() - start_time

    # lacking PSM
    start_time = time()
    cm = CausalModel(Y=d[Y].values,
                     D=d[T].values,
                     X=d[['x2', 'x3']].values)
    cm.est_via_matching(bias_adj=True)
    r5 = dict(cm.estimates['matching'])
    r5['time'] = time() - start_time

    # redundant PSM
    start_time = time()
    cm = CausalModel(Y=d[Y].values,
                     D=d[T].values,
                     X=d[['x1', 'x2', 'x3']].values)
    cm.est_via_matching(bias_adj=True)
    r6 = dict(cm.estimates['matching'])
    r6['time'] = time() - start_time

    # collect results
    r = {
        'valid control': r1,
        'no control': r2,
        'all control': r3,
        'psm_valid': r4,
        'psm_lacking': r5,
        'psm_redundant': r6
    }

    r2 = collect(r)
    r2.to_excel(f'{outputpath}bias_endogeneity.xlsx', index=True, )
    return r2


def data_study2(n=10000, e=0.05, binary_t=True):
    # data 4
    t = norm(n)
    if binary_t:
        t = t > 0
    x1 = t * 0.6 + norm(n) * e
    y = x1 * 0.5 + norm(n) * e
    d4 = {
        T: t.astype(np.int),
        'x1': x1,
        Y: y,
    }

    # data 5
    t = norm(n)
    if binary_t:
        t = t > 0
    x1 = t * 0.4 + norm(n) * e
    y = x1 * 0.5 + t * 0.1 + norm(n) * e
    d5 = {
        T: t.astype(np.int),
        'x1': x1,
        Y: y
    }

    data = [pd.DataFrame(i) for i in [d4, d5]]
    return data


def study2_overcontrol(n=40000, e=0.05):
    print('running study 2')
    data = data_study2(n, e)

    for i, d in enumerate(data):
        covariates = [i for i in d if i.startswith('x')]

        # by graph model
        dag = PCDK(T, Y)
        dag.fit(d)
        backdoor = dag.find_cia()
        print(f'backdoor: {backdoor}')

        # OLS
        r1 = estimate(d, T, Y, backdoor)  # valid
        r2 = estimate(d, T, Y)  # lacking
        r3 = estimate(d, T, Y, covariates)  # redundant

        # collect results
        r = {
            'valid control': r1,
            'no control': r2,
            'all control': r3,
        }

        r2 = collect(r)
        r2.to_excel(f'{outputpath}bias_overcontrol_{i}.xlsx', index=True)
        return r2


def data_study3(n=10000, e=1, binary_t=True):
    # data1
    x2 = norm(n)
    x3 = norm(n)
    t = x2 * 0.2 + norm(n) * e
    if binary_t:
        t = t > 0
    y = t * 0.3 + x3 * 0.2 + norm(n) * e
    x1 = t * 0.2 + y * 0.3 + norm(n) * e
    d8 = {
        T: t.astype(np.int),
        Y: y,
        'x1': x1,
        'x2': x2,
        'x3': x3
    }
    return pd.DataFrame(d8)


def study3_endogenous_selection(n=40000, e=0.05):
    print('running study 3')
    d = data_study3(n, e)

    # by graph model
    dag = PCDK(T, Y)
    G = dag.fit(d)
    backdoor = dag.find_cia()
    print(f'backdoor: {backdoor}')

    # OLS
    r1 = estimate(d, T, Y, backdoor)  # valid
    r2 = estimate(d, T, Y)  # no lacking
    # over
    r3 = estimate(d, T, Y, ['x1'])
    r4 = estimate(d, T, Y, ['x2', 'x3'])
    r5 = estimate(d, T, Y, ['x1', 'x2', 'x3'])

    # collect results
    r = {
        'valid control': r1,
        'no control': r2,
        'control x1': r3,
        'control x23': r4,
        'control x123': r5,
    }
    r2 = collect(r)
    r2.to_excel(f'{outputpath}bias_selection.xlsx', index=True)


def data_studies456(n_comfounder=3, n_overcontrol=3, n_collider=3, n=40000, e=0.05, ate=0.3):
    d = {}
    d['x1'] = norm(n)  # only influence t
    d['x2'] = norm(n)  # only influence y
    x_index = 3  # the current index of covariates to generate

    # effects level
    p = 0.2

    t = d['x1'] * 0.2 + norm(n) * e
    y = d['x2'] * 0.2 + norm(n) * e
    # confounding
    for confounder in range(n_comfounder):
        i_x = d[f'x{x_index}'] = norm(n)
        t += i_x * p
        y += i_x * p
        x_index += 1
    t = (t > 0) + 0

    # mediators
    for over in range(n_overcontrol):
        i_x = d[f'x{x_index}'] = t * 0.2 + norm(n) * e
        y += i_x * 0.5
        x_index += 1
    # direct treatment level
    direct_treat = ate - 0.1 * n_overcontrol
    y += t * direct_treat

    # colliders
    for collider in range(n_collider):
        d[f'x{x_index}'] = t * p + y * p + norm(n) * e
        x_index += 1

    d[T] = t
    d[Y] = y

    return pd.DataFrame(d)


def study4_mixed(n=40000, e=0.05):
    print('running study 4')
    r = []
    for confounder in [1, 2, 3]:
        for collider in [1, 2, 3]:
            for mediator in [1, 2, 3]:
                # generate data
                d = data_studies456(confounder, mediator, collider, n, e)
                d = data_studies456(1, mediator, collider, n, e)

                # record start time
                t0 = time()

                # identify backdoor
                dag = PCDK(T, Y)
                G = dag.fit(d)
                backdoor = dag.find_cia()
                ate = estimate(d, T, Y, backdoor)

                # record info
                ate['confounder'] = confounder
                ate['collider'] = collider
                ate['mediator'] = mediator
                ate['time'] = time() - t0

                r.append(ate)

    r2 = pd.DataFrame(r)
    r2['unbias'] = (r2['[0.025'] <= 0.3) * (r2['0.975]'] >= 0.3) + 0
    r2.to_excel(f'{outputpath}robust.xlsx', index=False)
    print('finished!')


def study5_robustness2(n_repeat=30):
    print('running study 5')
    r = []
    confounder = 3
    mediator = 3
    collider = 3
    e = 0.05
    for sample_size in [200, 800, 3200, 12800]:
        for causal_strength in [0.1, 0.3, 0.5]:
            for i in range(n_repeat):
                d = data_studies456(confounder, mediator, collider, sample_size, e, causal_strength)
                # record start time
                t0 = time()
                # identify backdoor
                dag = PCDK(T, Y)

                G = dag.fit(d)
                print(G)
                backdoor = dag.find_cia()
                ate = estimate(d, T, Y, backdoor)

                # record info
                i_r = {
                    'n': sample_size,
                    'true_ate': causal_strength,
                    'time': time() - t0,
                    'epoch': i
                }
                i_r.update(ate.to_dict())

                r.append(i_r)

    r2 = pd.DataFrame(r)
    r2['consistent'] = (r2['[0.025'] <= r2['true_ate']) * (r2['0.975]'] >= r2['true_ate']) + 0
    out_path = f'{outputpath}robust2.xlsx'
    r2.to_excel(out_path, index=False)
    print('finished!')


def study6_knowledge(n_repeat=60):
    print('running study 6')
    r = []
    confounder = 3
    mediator = 3
    collider = 3
    e = 0.05
    true_ate = 0.5
    n = 200

    domain1 = []
    domain2 = ['x3', T, 'x6', Y, 'x9']
    domain3 = [['x3', 'x4'], T, ['x6', 'x7'], Y, ['x9', 'x10']]
    domain4 = [['x3', 'x4', 'x5'], T, ['x6', 'x7', 'x8'], Y, ['x9', 'x10', 'x11']]
    domain5 = [['x1', 'x2', 'x3', 'x4', 'x5'], T, ['x6','x7', 'x8'], Y, ['x9', 'x10', 'x11']]
    domains = [domain1, domain2, domain3, domain4, domain5]

    for i_d, domain in enumerate(domains):
        for epoch in range(n_repeat):
            d = data_studies456(confounder, mediator, collider, n, e, true_ate)
            # record start time
            t0 = time()
            # identify backdoor
            dag = PCDK(T, Y)
            if len(domain) > 0:
                dag.add_timeline(domain)

            G = dag.fit(d)
            print(G)
            backdoor = dag.find_cia()
            ate = estimate(d, T, Y, backdoor)

            # record info
            i_r = {
                'n': n,
                'domain': i_d + 1,
                'true_ate': true_ate,
                'time': time() - t0,
                'epoch': epoch
            }
            i_r.update(ate.to_dict())

            r.append(i_r)

    r2 = pd.DataFrame(r)
    r2['consistent'] = (r2['[0.025'] <= r2['true_ate']) * (r2['0.975]'] >= r2['true_ate']) + 0
    out_path = f'{outputpath}robust3.xlsx'
    r2.to_excel(out_path, index=False)
    print('finished!')
    return r2


if __name__ == '__main__':
    np.random.seed(1)
    study1_confounding()
    study2_overcontrol()
    study3_endogenous_selection()
    study4_mixed()
    study5_robustness2()
    study6_knowledge()
