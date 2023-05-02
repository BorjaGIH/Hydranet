from shared_pkgs.imports import *

def load_and_format_covariates(file_path, dataset):

    data = pandas.read_csv(file_path, delimiter=',')

    if dataset=='ihdp':
        binfeats = ["bw","b.head","preterm","birth.o","nnhealth","momage"]
        contfeats = ["sex","twin","b.marr","mom.lths","mom.hs",	"mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was",'momwhite','momblack','momhisp']
        perm = binfeats + contfeats
        x = data[perm]
    elif dataset=='synthetic':
        covars = ['x{}'.format(i) for i in range(30)]
        x = data[covars]

    return x


def load_other_vars(file_path):

    data = pandas.read_csv(file_path, delimiter=',')

    t, y, y0, y1, y2, y3, y4 = data['z'], data['y'], data['y_0'], data['y_1'],  data['y_2'], data['y_3'], data['y_4'],
    mu_0, mu_1, mu_2, mu_3, mu_4 =  data['mu_0'], data['mu_1'], data['mu_2'], data['mu_3'], data['mu_4']

    return t.values.reshape(-1, 1), y, y0, y1, y2, y3, y4, mu_0, mu_1, mu_2, mu_3, mu_4


def split_output(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    q_t2 = y_scaler.inverse_transform(yt_hat[:, 2].reshape(-1, 1).copy())
    q_t3 = y_scaler.inverse_transform(yt_hat[:, 3].reshape(-1, 1).copy())
    q_t4 = y_scaler.inverse_transform(yt_hat[:, 4].reshape(-1, 1).copy())
    g0 = yt_hat[:, 5].copy()
    g1 = yt_hat[:, 6].copy()
    g2 = yt_hat[:, 7].copy()
    g3 = yt_hat[:, 8].copy()
    g4 = yt_hat[:, 9].copy()
    g = np.concatenate([np.transpose(g0), np.transpose(g1), np.transpose(g2), np.transpose(g3), np.transpose(g4)],axis=0)

    if yt_hat.shape[1] == 14:
        eps = yt_hat[:, 10:14]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())

    return {'q_t0': q_t0, 'q_t1': q_t1, 'q_t2': q_t2, 'q_t3': q_t3, 'q_t4': q_t4, 'g': g, 't': t, 'y': y, 'x': x,'index': index, 'eps': eps}


def split_output_dr(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1,1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1,1).copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def split_output_t(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    q_t2 = y_scaler.inverse_transform(yt_hat[:, 2].reshape(-1, 1).copy())
    q_t3 = y_scaler.inverse_transform(yt_hat[:, 3].reshape(-1, 1).copy())
    q_t4 = y_scaler.inverse_transform(yt_hat[:, 4].reshape(-1, 1).copy())

    y = y_scaler.inverse_transform(y.copy())

    return {'q_t0': q_t0, 'q_t1': q_t1, 'q_t2': q_t2, 'q_t3': q_t3, 'q_t4': q_t4, 't': t, 'y': y, 'x': x,
            'index': index}

def load_truth():
    """
    loading ground truth data
    """

    data = load(file_path)
    mu_0 = data['mu_0']
    mu_1 = data['mu_1']
    mu_2 = data['mu_2']

    return mu_0, mu_1, mu_2


def load_data():
    """
    loading train test experiment results
    """

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['q_t2'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['index'].reshape(-1, 1), data['eps'].reshape(-1, 1)


def truncate_by_g(attribute, g, level=0.01):

    if level==0.:
        return attribute
    else:
        g = g.reshape(3,len(attribute)).T
        g_ind = np.sum(g<level, axis=1)
        #keep_these = np.logical_and(g >= level, g <= 1.-level)
        return attribute[g_ind==0]


def truncate_all_by_g(q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y, truncate_level=0.05):
    """
    Helper function to clean up nuisance parameter estimates.

    """

    g = g.reshape((max(t) + 1)[0], len(y)).T

    g_ind = np.sum(g < truncate_level, axis=1)

    q_t0 = q_t0[g_ind == 0]
    q_t1 = q_t1[g_ind == 0]
    q_t2 = q_t2[g_ind == 0]
    q_t3 = q_t3[g_ind == 0]
    q_t4 = q_t4[g_ind == 0]
    g = g[g_ind == 0, :].flatten(order='F').reshape(-1, 1)
    t = t[g_ind == 0]
    y = y[g_ind == 0]


    return q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y



'''def cross_entropy(y, p):
    return -np.mean((y*np.log(p) + (1.-y)*np.log(1.-p)))


def mse(x, y):
    return np.mean(np.square(x-y))'''