from shared_pkgs.imports import *

def load_and_format_covariates(file_path):

    data = pandas.read_csv(file_path, delimiter=',')

    binfeats = ["bw","b.head","preterm","birth.o","nnhealth","momage"]
    contfeats = ["sex","twin","b.marr","mom.lths","mom.hs",	"mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was",'momwhite','momblack','momhisp']

    perm = binfeats + contfeats
    x = data[perm]
    return x

def load_other_vars(file_path):
    data = pandas.read_csv(file_path, delimiter=',')
    t, y, y0, y1, y2 = data['z'], data['y'], data['y_0'], data['y_1'],  data['y_2']
    mu_0, mu_1, mu_2 =  data['mu_0'], data['mu_1'], data['mu_2']
    return t.values.reshape(-1, 1), y, y0, y1, y2, mu_0, mu_1, mu_2


# Auxiliary function
def split_output(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    q_t2 = y_scaler.inverse_transform(yt_hat[:, 2].reshape(-1, 1).copy())
    g0 = yt_hat[:, 3].copy()
    g1 = yt_hat[:, 4].copy()
    g2 = yt_hat[:, 5].copy()
    #g = Concatenate(0)([tf.transpose(g0), tf.transpose(g1), tf.transpose(g2)])
    g = np.concatenate((g0,g1,g2), axis=0)

    if yt_hat.shape[1] == 8:
        eps = yt_hat[:, 6:8]
    else:
        eps = np.zeros_like(yt_hat[:, 2]) #??

    y = y_scaler.inverse_transform(y.copy())
    
    var = "average propensity for t=0: {}, t=1: {}, and t=2: {}".format(g0.mean(), g1.mean(), g2.mean())
    
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'q_t2': q_t2, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def load_truth(replication, n_rep):
    """
    loading ground truth data
    """

    file_path = '/home/bvelasco/Dragonnet/dragonnet/result/3_treats/reps_{}/{}/simulation_outputs.npz'.format(n_rep, replication)
    data = load(file_path)
    mu_0 = data['mu_0']
    mu_1 = data['mu_1']
    mu_2 = data['mu_2']

    return mu_0, mu_1, mu_2


def load_data(replication=1, model='baseline', train_test='test', n_rep='1'):
    """
    loading train test experiment results
    """

    file_path = '/home/bvelasco/Dragonnet/dragonnet/result/3_treats/reps_{}/'.format(n_rep)
    data = load(file_path + '{}/{}/0_replication_{}.npz'.format(replication, model, train_test))

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['q_t2'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['index'].reshape(-1, 1), data['eps'].reshape(-1, 1)



def calibrate_g(g, t):
    """
    Improve calibration of propensity scores by fitting 1 parameter (temperature) logistic regression on heldout data

    :param g: raw propensity score estimates
    :param t: treatment assignments
    :return:
    """

    logit_g = logit(g).reshape(-1,1)
    calibrator = lm.LogisticRegression(fit_intercept=False, C=1e6)  # no intercept or regularization
    calibrator.fit(logit_g, t)
    calibrated_g = calibrator.predict_proba(logit_g)[:,1]
    return calibrated_g


def truncate_by_g(attribute, g, level=0.01):
    if level==0.:
        return attribute
    else:
        keep_these = np.logical_and(g >= level, g <= 1.-level)
        return attribute[keep_these]


def truncate_all_by_g(q_t0, q_t1, q_t2, g, t, y, truncate_level=0.05):
    """
    Helper function to clean up nuisance parameter estimates.

    """

    orig_g = np.copy(g)

    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    q_t2 = truncate_by_g(np.copy(q_t2), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)

    return q_t0, q_t1, q_t2, g, t, y



def cross_entropy(y, p):
    return -np.mean((y*np.log(p) + (1.-y)*np.log(1.-p)))


def mse(x, y):
    return np.mean(np.square(x-y))