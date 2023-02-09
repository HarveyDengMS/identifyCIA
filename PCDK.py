# PCDK: the causal graph algorithm PC integrated with domain knowledge
# the code is developed based on gCastle, which can be installed by "pip install gcastle" in cmd
# you can directly call pc_domain, find_cia, and estimate to obtain the estiamted DAG and CIA sufficiency
# or you can utilize PCDK class to achieve the same process
import pandas as pd
from copy import deepcopy
from itertools import combinations
import statsmodels.api as sm
import numpy as np
from warnings import filterwarnings
from time import time
from scipy.stats import norm
import math

filterwarnings('ignore')
# max iteration of orient identification
MAX_ITER = 3
NAME = 'name'


def pc_domain(data: pd.DataFrame, domain: dict = {}, alpha: float = 0.05, ccit=None, name=None):
    """
    Causal discovery algorithm based on PC and domain knowledge
    :param data: pd.DataFrame (n_samples, n_features), training data
    :param domain: dict {("from", "to"): edge}, in which edge = 1 indicates variable "from" causes "to" in domain,
        edge = 0 excludes that cause. e.g., domain = {("Y", "T"): 0} excludes Y -> T, domain = {("T", "Y"): 1} specifies
        T -> Y and excludes Y -> T in DAG
    :param alpha: significance level
    :param ccit: the conditional independence test function. default: gauss test
    :param name: notation for the results
    :return: predicted causal matrix
    """
    n_na = pd.DataFrame(data).isna().sum().sum()
    if n_na > 0:
        print(f"The data includes {n_na} missing values, which are not supported currently. We replace them by 0.")
        data = data.fillna(0)

    skeleton, sep_set = formation_by_CI_test(data, domain=domain, alpha=alpha, ccit=ccit)
    # Generating an causal matrix (DAG)
    m_pred = orient(skeleton, sep_set)
    m_est = pd.DataFrame(m_pred, index=data.keys(), columns=data.keys())
    if name is not None:
        m_est[NAME] = name
    return m_est


def ci_test(data: np.array, i: int, j: int, ctrl_var: list = [], fun_ccit=None):
    """conditional independence test
    :param data: np.array with shape n*m, n: sample size, m: feature size
    :param i: column index of i in data
    :param j: column index of j in data
    :param ctrl_var: column index list of control variable in data
    :param fun_ccit: the function for conditional independence test with a structure as Gauss test (Default)
    :return: p value of CI test, where p<significant level indicates i and j are significantly interdependent
    """
    if fun_ccit is None:
        # default is gauss test. You can replace the default function by yours
        fun_ccit = gauss_test
    p = fun_ccit(data, i, j, ctrl_var)
    return p


def gauss_test(data: np.array, i: int, j: int, ctrl_var: list = []):
    """Gauss test from gCastle
    :param data: np.array with shape n*m, n: sample size, m: feature size
    :param i: column index of i in data
    :param j: column index of j in data
    :param ctrl_var: column index list of control variable in data
    :return: p value of CI test, where p<significant level indicates i and j are significantly interdependent
    """

    n = data.shape[0]
    k = len(ctrl_var)
    if k == 0:
        r = np.corrcoef(data[:, [i, j]].T)[0][1]
    else:
        sub_index = [i, j]
        sub_index.extend(ctrl_var)
        sub_corr = np.corrcoef(data[:, sub_index].T)
        # inverse matrix
        PM = np.linalg.inv(sub_corr)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    cut_at = 0.99999

    r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1
    # Fisher’s z-transform
    res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
    p = 2 * (1 - norm.cdf(abs(res)))
    return p


def formation_by_CI_test(d, domain=None, alpha=0.05, ccit=None):
    """Origin PC-algorithm for learns a skeleton graph with domain knowledge"""
    # init skeleton of fully connected graph
    n_features = d.shape[1]
    features = np.array(d.keys())
    data = d.values
    le = {i: id for id, i in enumerate(features)}
    skeleton = np.ones((n_features, n_features)) - np.eye(n_features)
    domain = {(le[i[0]], le[i[1]]): domain[i] for i in domain}  # replace keys to index numbers

    # init separation sets and nodes
    sep_set = {}
    nodes = list(range(n_features))

    # constrain from domain knowledge
    for i, j in domain:
        skeleton[i, j] = domain[i, j]
        if domain[i, j] == 1:
            # direct
            skeleton[j, i] = 0
        if skeleton[i, j] + skeleton[j, i] == 0:
            # independent (separation)
            sep_set[(i, j)] = []

    # start iter
    k = 0
    while k <= n_features - 2:
        for i in nodes:
            for j in nodes:
                # omit edges that have been cut either by domain or by conditional independence test
                if skeleton[i, j] == 0:
                    continue

                # omit edges that are specified by domain and exist
                if (i, j) in domain:
                    if domain[i, j] == 1:
                        continue

                if k == 0:
                    p_value = ci_test(data, i, j, ctrl_var=[], fun_ccit=ccit)
                    if p_value >= alpha:
                        skeleton[i, j] = skeleton[j, i] = 0
                        print(f'{features[i]} to {features[j]} is pruned')
                        sep_set[(i, j)] = []
                    else:
                        pass
                else:
                    other_nodes = deepcopy(nodes)
                    other_nodes.remove(i)
                    other_nodes.remove(j)
                    # important modifier of PCD: only conditional on direct ascendants
                    for other in list(other_nodes):
                        if skeleton[other, i] + skeleton[other, j] == 0:
                            other_nodes.remove(other)

                    s = []
                    for ctrl_var in combinations(other_nodes, k):
                        ctrl_var = list(ctrl_var)
                        p_value = ci_test(data, i, j, ctrl_var, fun_ccit=ccit)
                        if p_value >= alpha:
                            s.extend(ctrl_var)
                        if s:
                            skeleton[i, j] = skeleton[j, i] = 0
                            print(f'{features[i]} to {features[j]} is pruned, conditioned on {features[s]}')
                            sep_set[(i, j)] = s
                            break
        k += 1

    return skeleton, sep_set


def orient(skeleton, sep_set):
    """Extending the Skeleton to the Equivalence Class

    it orients the undirected edges to form an equivalence class of DAGs.

    Parameters
    ----------
    skeleton : array
        The undirected graph
    sep_set : dict
        separation sets
        if key is (x, y), then value is a set of other variables
        not contains x and y

    Returns
    -------
    out : array
        An equivalence class of DAGs can be uniquely described
        by a completed partially directed acyclic graph (CPDAG)
        which includes both directed and undirected edges.
    """

    def _rule_1(cpdag):
        """Rule_1

        Orient i——j into i——>j whenever there is an arrow k——>i
        such that k and j are nonadjacent.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
                continue
            # search i——j
            else:
                all_k = [x for x in columns if x not in ij]
                for k in all_k:
                    if cpdag[k, i] == 1 and cpdag[i, k] == 0 \
                            and cpdag[k, j] + cpdag[j, k] == 0:
                        cpdag[j, i] = 0
        return cpdag

    def _rule_2(cpdag):
        """Rule_2

        Orient i——j into i——>j whenever there is a chain i——>k——>j.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
                continue
            # search i——j
            else:
                all_k = [x for x in columns if x not in ij]
                for k in all_k:
                    if cpdag[i, k] == 1 and cpdag[k, i] == 0 \
                            and cpdag[k, j] == 1 \
                            and cpdag[j, k] == 0:
                        cpdag[j, i] = 0
        return cpdag

    def _rule_3(cpdag, sep_set=None):
        """Rule_3

        Orient i——j into i——>j
        whenever there are two chains i——k——>j and i——l——>j
        such that k and l are non-adjacent.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
                continue
            # search i——j
            else:
                for kl in sep_set.keys():  # k and l are nonadjacent.
                    k, l = kl
                    # if i——k——>j and  i——l——>j
                    if cpdag[i, k] == 1 \
                            and cpdag[k, i] == 1 \
                            and cpdag[k, j] == 1 \
                            and cpdag[j, k] == 0 \
                            and cpdag[i, l] == 1 \
                            and cpdag[l, i] == 1 \
                            and cpdag[l, j] == 1 \
                            and cpdag[j, l] == 0:
                        cpdag[j, i] = 0
        return cpdag

    def _rule_4(cpdag, sep_set=None):
        """Rule_4

        Orient i——j into i——>j
        whenever there are two chains i——k——>l and k——>l——>j
        such that k and j are non-adjacent.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
                continue
            # search i——j
            else:
                for kj in sep_set.keys():  # k and j are nonadjacent.
                    if j not in kj:
                        continue
                    else:
                        kj = list(kj)
                        kj.remove(j)
                        k = kj[0]
                        ls = [x for x in columns if x not in [i, j, k]]
                        for l in ls:
                            if cpdag[k, l] == 1 \
                                    and cpdag[l, k] == 0 \
                                    and cpdag[i, k] == 1 \
                                    and cpdag[k, i] == 1 \
                                    and cpdag[l, j] == 1 \
                                    and cpdag[j, l] == 0:
                                cpdag[j, i] = 0
        return cpdag

    columns = list(range(skeleton.shape[1]))
    cpdag = deepcopy(skeleton)
    # pre-processing
    for ij in sep_set.keys():
        i, j = ij
        all_k = [x for x in columns if x not in ij and x not in sep_set[ij]]
        for k in all_k:
            if cpdag[i, k] + cpdag[k, i] != 0 \
                    and cpdag[k, j] + cpdag[j, k] != 0:
                if cpdag[i, k] + cpdag[k, i] == 2:
                    print(f'{k} -> {i} is excluded as {i} <> {j}')
                    cpdag[k, i] = 0
                if cpdag[j, k] + cpdag[k, j] == 2:
                    print(f'{k} -> {j} is excluded as {i} <> {j}')
                    cpdag[k, j] = 0

    cpdag_copy = deepcopy(cpdag)
    for i in range(MAX_ITER):
        cpdag = _rule_1(cpdag=cpdag)
        cpdag = _rule_2(cpdag=cpdag)
        cpdag = _rule_3(cpdag=cpdag, sep_set=sep_set)
        cpdag = _rule_4(cpdag=cpdag, sep_set=sep_set)
        if np.all(cpdag == cpdag_copy):
            break
        else:
            cpdag_copy = deepcopy(cpdag)
    return cpdag.astype(int)


def descendant(m, starts=['T'], blocks=[], max_depth=1000):
    """return the descendant nodes starting from 'starts' and blocked by 'blocks', by iterating the adjacent matrix"""
    starts = list(starts)
    blocks = list(blocks)

    # blocks
    m = pd.DataFrame(m)
    m2 = m.copy()
    if len(blocks) > 0:
        m2.loc[blocks, :] = 0
    m2 = np.array(m2)

    # reformat
    le = {i: id for id, i in enumerate(m.keys())}
    le = pd.Series(le)
    starts = le[starts].values

    # init descendant vector (including itself)
    n_features = m.shape[1]
    v = np.zeros([1, n_features])
    v[0, starts] = 1

    # iter
    v2 = 0
    for i in range(max_depth):
        v += np.matmul(v, m2)
        v = np.clip(v, 0, 1)
        if np.all(v == v2):
            break
        else:
            v2 = v.copy()

    # map index to label
    v = v.reshape([-1]) == 1
    v[starts] = False  # exclude start points
    des = np.array(m.keys())[v]
    return des


def non_des(m, starts=['T'], max_depth=1000):
    """return the non-descendant nodes"""
    m = pd.DataFrame(m)
    des = descendant(m, starts, max_depth=max_depth)
    nd = set(m.keys()) - set(des) - set(starts)
    return list(nd)


def find_cia(m, T='T', Y='Y', safe_mode=False):
    """

    :param m: dataframe of adjacent matrix
    :param T: name of treatment
    :param Y: name of outcome
    :return:
    """
    if safe_mode:
        try:
            return find_cia_core(m, T, Y)
        except:
            return ['error']
    else:
        return find_cia_core(m, T, Y)


def find_cia_core(m, T='T', Y='Y'):
    """
    identify CIA from adjacent matrix of DAG
    :param m: adjacent matrix of DAG
        pd.DataFrame in shape (n_features, n_features), where 1st dim for "from", 2nd for "to"
    :return: list of feature name, CIA list
    """
    if NAME in m:
        m = m.drop(NAME, axis=1)

    # condition1: non desecedents of T
    x1 = non_des(m, [T])

    # condition 2-1: on the path between T and Y with an arrow to T
    # undirected adjacent matrix
    x2 = []
    for i_x in x1:
        if Y in descendant(m, [i_x], [T]) and T in descendant(m, [i_x]):
            x2.append(i_x)

    backdoor = []
    for k in range(1, len(x2) + 1):
        for i_set in combinations(x2, k):
            # condition 2-2: blocks the path between T and Y with an arrow to T
            # the rest nodes on the path
            i_set = list(i_set)
            i_rest = set(x2) - set(i_set)
            if Y in descendant(m, i_rest, [T] + i_set) and T in descendant(m, i_rest, i_set):
                # exist other path between T and Y with an arrow to T
                continue
            else:
                # doesn't exist such a path after blocking i_set, such that i_set is the backdoor
                backdoor = i_set
                return backdoor

    return backdoor


def estimate(d, T, Y, covariates=[], interaction=False):
    """
    Fitting OLS with covariates.
    :param d: pd.DataFrame, including X, Y and specified covariates
    :param T: string, name of treatment variable in d
    :param Y: string, name of outcome variable in d
    :param covariates: list of string, feature name of covariates
    :param interaction: whether to including interaction terms of treatment and covariates
    :return: statsmodels result object, OLS estimation results
    """

    def get_treatment(result, T):
        return result.summary2().tables[1].loc[T, :]

    start_time = time()
    # get x and y
    y = d[Y]
    x_list = [T] + list(covariates)
    x = d[x_list]
    x = sm.add_constant(x)
    if interaction:
        for i in covariates:
            i_iter = f'T_{i}'
            x[i_iter] = x[T] * x[covariates]

    # estimate with OLS
    model = sm.OLS(y, x)
    result = model.fit()
    est = get_treatment(result, T)

    # time summary
    est['time'] = time() - start_time
    return est


def timeline_domain(x: list):
    """
    quickly generating domain knowledge based on timeline knowledge
    :param x: list of feature name ordered in time line, from left to right. e.g., [x1, x2, x3] indicates x1 -> x2 -> x3
    :return: dict of domain knowledge, where dict[(i,j)] = 0 indicates i -> j is excluded as j precedes i
    """
    # init output dict
    out = {}
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            x0 = np.array(x[i]).reshape([-1])
            x1 = np.array(x[j]).reshape([-1])
            for k0 in x0:
                for k1 in x1:
                    out[(k1, k0)] = 0
    return out


def antecedent_variables(x, exo):
    """
    quickly generating domain knowledge for exogeneous variable
    :param x: feature list
    :param exo: exogenous feature list
    :return: dict of domain knowledge that excluding the causal relationship directed to exogenous variables
    """
    out = {}
    x = x + exo
    x = np.unique(x)
    for i in x:
        for j in exo:
            if i != j:
                out[(i, j)] = 0

    return out


class PCDK:
    m_est = None
    domain = {}

    def __init__(self, T, Y, domain=None, alpha=0.01, name=None):
        """

        :param T: the label name of treatment variable in inputting dataframe
        :param Y: the label name of outcome variable in inputting dataframe
        :param domain: domain knowledge
        :param alpha: alpha level for conditional independence test
        :param name: name of the model
        """
        self.T = T
        self.Y = Y
        # default domain: exclude Y -> T
        if domain is None:
            self.domain = {}
            self.domain.update({(Y, T): 0})
        self.alpha = alpha
        self.name = name

    def fit(self, data: pd.DataFrame, ccit=None):
        """
        :param data: the dataframe containing treatment T, outcome Y, and control variables
        :return: the estimated causal graph
        """
        self.d = data
        self.m_est = pc_domain(data, domain=self.domain, alpha=self.alpha, ccit=ccit, name=self.name)
        return self.m_est

    def update_domain(self, domain:dict = {}):
        self.domain.update(domain)
        return self.domain

    def find_cia(self):
        self.cia = find_cia(self.m_est, self.T, self.Y)
        return self.cia

    def estimate_ate(self, interaction=False):
        self.treat_est = estimate(self.d, self.T, self.Y, self.cia, interaction)
        return self.treat_est

    def add_timeline(self, timeline, inplace=True):
        """
        quickly generating domain knowledge based on timeline knowledge
        :param timeline: list of feature name ordered in timeline, such as [t0, t1] in which t1 happens after t0
        :param inplace: whether update the time domain to the model
        :return: dict of time domain
        """
        time_domain = timeline_domain(timeline)
        if inplace:
            self.domain.update(time_domain)
        return time_domain

    def add_antecedents(self, end, exo, inplace=True):
        """
        quickly generating domain knowledge for exogeneous variable
        :param end: endogenous feature list
        :param exo: exdogenous feature list
        :param inplace: whether update the exogenous domain to the model
        :return: dict of exogenous domain
        """
        exo_domain = antecedent_variables(end, exo)
        if inplace:
            self.domain.update(exo_domain)
        return self.exo_domain
