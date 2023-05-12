from shared_pkgs.imports import *
from Helpers.helper_funcs import *

def psi_iptw(g, t, y, truncate_level=0.05):

    orig_g = g.copy()

    t = t.flatten()
    encoded_t = np.zeros((t.size, t.max() + 1))
    encoded_t[np.arange(t.size), t] = 1

    g = g.reshape(max(t) + 1, len(y)).T
    g_t = encoded_t * (1 / g)

    h1_0 = g_t[:, 1] - g_t[:, 0]
    h2_0 = g_t[:, 2] - g_t[:, 0]
    h3_0 = g_t[:, 3] - g_t[:, 0]
    h4_0 = g_t[:, 4] - g_t[:, 0]

    ite1_0 = np.multiply(h1_0, y.flatten()).reshape(-1, 1)
    ite2_0 = np.multiply(h2_0, y.flatten()).reshape(-1, 1)
    ite3_0 = np.multiply(h3_0, y.flatten()).reshape(-1, 1)
    ite4_0 = np.multiply(h4_0, y.flatten()).reshape(-1, 1)

    ite1_0 = np.mean(truncate_by_g(ite1_0, orig_g.reshape(-1, 1), level=truncate_level))
    ite2_0 = np.mean(truncate_by_g(ite2_0, orig_g.reshape(-1, 1), level=truncate_level))
    ite3_0 = np.mean(truncate_by_g(ite3_0, orig_g.reshape(-1, 1), level=truncate_level))
    ite4_0 = np.mean(truncate_by_g(ite4_0, orig_g.reshape(-1, 1), level=truncate_level))

    # ite=(t / g - (1-t) / (1-g))*y
    # return np.mean(truncate_by_g(ite, g, level=truncate_level))

    return [ite1_0, ite2_0, ite3_0, ite4_0]


def psi_aiptw(q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y, num_treats, truncate_level=0.05):

    q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y = truncate_all_by_g(q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y, num_treats, truncate_level)

    t = t.flatten()
    encoded_t = np.zeros((int(t.size), num_treats))
    encoded_t[np.arange(int(t.size)), t.astype(int)] = 1

    fullq = np.sum(np.concatenate([q_t0, q_t1, q_t2, q_t3, q_t4], axis=1) * encoded_t, axis=1)

    g = g.reshape(num_treats, len(y)).T
    g_t = encoded_t * (1 / g)

    h1_0 = g_t[:, 1] - g_t[:, 0]
    h2_0 = g_t[:, 2] - g_t[:, 0]
    h3_0 = g_t[:, 3] - g_t[:, 0]
    h4_0 = g_t[:, 4] - g_t[:, 0]

    ite1_0 = h1_0 * (y - fullq) + q_t1 - q_t0
    ite2_0 = h2_0 * (y - fullq) + q_t2 - q_t0
    ite3_0 = np.multiply(h3_0, y.flatten()).reshape(-1, 1)
    ite4_0 = np.multiply(h4_0, y.flatten()).reshape(-1, 1)

    # ite1_0 =  q_t1 - q_t0
    # ite2_0 =  q_t2 - q_t0
    # h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    # ite = h * (y - full_q) + q_t1 - q_t0

    return [np.mean(ite1_0), np.mean(ite2_0), np.mean(ite3_0), np.mean(ite4_0)]


def psi_naive(q_t0, q_t1, q_t2, q_t3, q_t4, g, truncate_level=0.):
    ite1_0 = (q_t1 - q_t0)
    ite2_0 = (q_t2 - q_t0)
    ite3_0 = (q_t3 - q_t0)
    ite4_0 = (q_t4 - q_t0)

    ite1_0 = np.mean(truncate_by_g(ite1_0, g, level=truncate_level))
    ite2_0 = np.mean(truncate_by_g(ite2_0, g, level=truncate_level))
    ite3_0 = np.mean(truncate_by_g(ite3_0, g, level=truncate_level))
    ite4_0 = np.mean(truncate_by_g(ite4_0, g, level=truncate_level))

    return [ite1_0, ite2_0, ite3_0, ite4_0]


def psi_very_naive(y, t):
    y1_0 = y[t == 1].mean() - y[t == 0].mean()
    y2_0 = y[t == 2].mean() - y[t == 0].mean()
    y3_0 = y[t == 3].mean() - y[t == 0].mean()
    y4_0 = y[t == 4].mean() - y[t == 0].mean()

    return [y1_0, y2_0, y3_0, y4_0]


def get_estimate(q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y_dragon, truncate_level=0.01):

    psi_vn = psi_very_naive(y_dragon, t)
    psi_n = psi_naive(q_t0, q_t1, q_t2, g, t, y_dragon, truncate_level=truncate_level)
    psi_iptw_ = psi_iptw(g, t, y, truncate_level=truncate_level)
    psi_aiptw_ = psi_aiptw(q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y, truncate_level=truncate_level)

    return psi_vn, psi_n


################ FOR DRAGONNET ################

def psi_naive_dr(q_t0, q_t1, g, truncate_level=0.05):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))

def psi_very_naive_dr(y, t):
    return y[t == 1].mean() - y[t == 0].mean()

def get_estimate_dr(q_t0, q_t1, g, t, y_dragon, truncate_level=0.01):
    """
    getting the back door adjustment & TMLE estimation
    """
    psi_vn = psi_very_naive_dr(y_dragon, t)
    psi_n = psi_naive_dr(q_t0, q_t1, g, truncate_level=truncate_level)

    return psi_vn, psi_n