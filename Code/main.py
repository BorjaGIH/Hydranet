import os
import sys

import pandas as pd

from shared_pkgs.imports import *
from Helpers.helper_funcs import *
from Neural_Net.neural_net import *
from Neural_Net.losses import *
from Estimation.estimators import *


def train_and_predict_hydra(num_treats, t, y_unscaled, x, targeted_regularization, loss, val_split, batch_size):
    verbose = 0
    y_unscaled = y_unscaled.values.reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []

    hydranet = make_hydranet(x.shape[1], num_treats, 0.01)

    metrics = [hydranet_loss, regression_loss, categorical_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=5, hydranet_loss=loss)
    else:
        loss = loss

    # Get train and test indexes
    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=val_split, random_state=1, shuffle=True)

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)

    # With Adam
    hydranet.compile(
        optimizer=Adam(lr=1e-3),
        loss=loss, metrics=metrics, run_eagerly=False)

    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0)
    ]
    
    hydranet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=100,
                  batch_size=batch_size, verbose=verbose)
    
    # with SGD
    sgd_lr = 1e-5
    momentum = 0.9
    hydranet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
                      loss=loss, metrics=metrics, run_eagerly=False)
    
    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]
    
    hydranet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=200,
                  batch_size=batch_size, verbose=verbose)
    

    # Plot metrics to monitor the training process
    '''plt.figure()
    plt.plot(hydranet.history.history['loss'])
    plt.plot(hydranet.history.history['val_loss'])
    plt.legend(["Train", "Test"])
    plt.title("Loss")
    plt.show() # Training and validation losses
    
    plt.figure()
    plt.plot(hydranet.history.history['track_epsilon'])
    plt.plot(hydranet.history.history['val_track_epsilon'])
    plt.legend(["Train", "Test"])
    plt.title("Epsilon value (Regularization term)")
    plt.show() # Epsilon'''

    yt_hat_test = hydranet.predict(x_test)
    yt_hat_train = hydranet.predict(x_train)
    
    train_outputs += [split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
    test_outputs += [split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]

    K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_b2bd(t, y_unscaled, x, targeted_regularization, loss, val_split, batch_size):
    verbose = 0
    y_unscaled = y_unscaled.values.reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []

    dragonnet = make_dragonnet(x.shape[1], 0.01)

    metrics = [regression_loss_dr, binary_classification_loss, treatment_accuracy_dr, track_epsilon_dr]

    if targeted_regularization:
        loss = make_tarreg_loss_dr(ratio=2, dragonnet_loss=loss)
    else:
        loss = loss

    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=1, shuffle=True)

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)

    # With Adam
    opt = Adam(lr=1e-3)

    dragonnet.compile(
        optimizer=opt,
        loss=loss, metrics=metrics,
        run_eagerly=False)

    adam_callbacks = [
        TerminateOnNaN(),
        # PlotLearning(),
        EarlyStopping(monitor='val_loss', patience=5, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',min_delta=1e-8, cooldown=0, min_lr=0)]

    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=100,
                  batch_size=batch_size, verbose=verbose)

    # with SGD
    sgd_lr = 1e-5
    momentum = 0.9
    opt = SGD(lr=sgd_lr, momentum=momentum, nesterov=True)
    dragonnet.compile(optimizer=opt,
        loss=loss, metrics=metrics,
        run_eagerly=False)

    sgd_callbacks = [
        TerminateOnNaN(),
        # PlotLearning(),
        EarlyStopping(monitor='val_loss', patience=20, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',min_delta=0., cooldown=0, min_lr=0)]

    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=200,
                  batch_size=batch_size, verbose=verbose)


    # Plot metrics to monitor the training process
    '''plt.figure()
    plt.plot(dragonnet.history.history['loss'])
    plt.plot(dragonnet.history.history['val_loss'])
    plt.legend(["Train", "Test"])
    plt.title("Loss")
    plt.show() # Training and validation losses

    plt.figure()
    plt.plot(dragonnet.history.history['treatment_accuracy'])
    plt.plot(dragonnet.history.history['val_treatment_accuracy'])
    plt.legend(["Train", "Test"])
    plt.title("Treatment prediction accuracy")
    plt.show() # Treatment prediction accuracy

    plt.figure()
    plt.plot(dragonnet.history.history['track_epsilon'])
    plt.plot(dragonnet.history.history['val_track_epsilon'])
    plt.legend(["Train", "Test"])
    plt.title("Epsilon value (Regularization term)")
    plt.show() # Epsilon'''

    yt_hat_test = dragonnet.predict(x_test)
    yt_hat_train = dragonnet.predict(x_train)

    train_outputs += [split_output_dr(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
    test_outputs += [split_output_dr(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]

    K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_tlearn(dataset, t, y_unscaled, x, val_split):
    y_unscaled = y_unscaled.values.reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []

    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=val_split, random_state=1, shuffle=True)

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    if dataset=='ihdp':
        covars = ["bw", "b.head", "preterm", "birth.o", "nnhealth", "momage", "sex", "twin", "b.marr", "mom.lths","mom.hs",
                  "mom.scoll", "cig", "first", "booze", "drugs", "work.dur", "prenatal", "ark", "ein", "har", "mia","pen",
                  "tex", "was", 'momwhite', 'momblack', 'momhisp']
        cols = covars + ['y', 'z']
        xyt_train = pandas.DataFrame(np.concatenate([x_train, y_train, t_train], 1),columns=cols)
        X, T, y = covars, "z", "y"
    else:
        covars = ['x{}'.format(i) for i in range(30)]
        cols = covars + ['y', 'z']
        xyt_train = pandas.DataFrame(np.concatenate([x_train, y_train, t_train], 1), columns=cols)
        X, T, y = covars, "z", "y"

    m0 = LGBMRegressor(max_depth=2, min_child_samples=60)
    m1 = LGBMRegressor(max_depth=2, min_child_samples=60)
    m2 = LGBMRegressor(max_depth=2, min_child_samples=60)
    m3 = LGBMRegressor(max_depth=2, min_child_samples=60)
    m4 = LGBMRegressor(max_depth=2, min_child_samples=60)

    m0.fit(xyt_train.query(f"{T}==0")[X], xyt_train.query(f"{T}==0")[y])
    m1.fit(xyt_train.query(f"{T}==1")[X], xyt_train.query(f"{T}==1")[y])
    m2.fit(xyt_train.query(f"{T}==2")[X], xyt_train.query(f"{T}==2")[y])
    m3.fit(xyt_train.query(f"{T}==3")[X], xyt_train.query(f"{T}==3")[y])
    m4.fit(xyt_train.query(f"{T}==4")[X], xyt_train.query(f"{T}==4")[y])

    yt_hat_test = np.concatenate([m0.predict(x_test[X]).reshape(-1, 1), m1.predict(x_test[X]).reshape(-1, 1),
                                  m2.predict(x_test[X]).reshape(-1, 1), m3.predict(x_test[X]).reshape(-1, 1),
                                  m4.predict(x_test[X]).reshape(-1, 1)], axis=1)
    yt_hat_train = np.concatenate([m0.predict(xyt_train[X]).reshape(-1, 1), m1.predict(xyt_train[X]).reshape(-1, 1),
                                   m2.predict(xyt_train[X]).reshape(-1, 1), m3.predict(xyt_train[X]).reshape(-1, 1),
                                   m4.predict(xyt_train[X]).reshape(-1, 1)], axis=1)


    train_outputs += [split_output_t(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]

    test_outputs += [split_output_t(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]

    return test_outputs, train_outputs


def run_train(input_dir, output_dir, dataset, num_treats, loss, loss_dr, val_split, batch_size):

    simulation_files = sorted(glob.glob("{}/*.csv".format(input_dir)))

    for idx, simulation_file in enumerate(simulation_files):
        print(simulation_file)

        simulation_output_dir = os.path.join(output_dir, str(idx))

        os.makedirs(simulation_output_dir, exist_ok=True)

        x = load_and_format_covariates(simulation_file, dataset)
        if num_treats==5:
            t, y, y0, y1, y2, y3, y4, mu_0, mu_1, mu_2, mu_3, mu_4 = load_other_vars(simulation_file)
            np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),\
                                t=t, y0=y0, y1=y1, y2=y2, y3=y3, y4=y4, mu_0=mu_0, mu_1=mu_1, mu_2=mu_2, mu_3=mu_3, mu_4=mu_4)
        else:
            t, y, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9,\
                mu_0, mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9 = load_other_vars(simulation_file)
            np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                                t=t, y0=y0, y1=y1, y2=y2, y3=y3, y4=y4, y5=y5, y6=y6, y7=y7, y8=y8, y9=y9,\
                                mu_0=mu_0, mu_1=mu_1, mu_2=mu_2, mu_3=mu_3, mu_4=mu_4, mu_5=mu_5, mu_6=mu_6, mu_7=mu_7, mu_8=mu_8, mu_9=mu_9)

        ############# RUN THE DIFFERENT ESTIMATORS ##############

        # T-learner
        test_outputs_tlearn, train_output_tlearn = train_and_predict_tlearn(dataset, t, y, x, val_split=val_split)

        output_dir_tlearn = os.path.join(simulation_output_dir, "T_learn/baseline")
        os.makedirs(output_dir_tlearn, exist_ok=True)

        # Save outputs for each split
        for num, output in enumerate(test_outputs_tlearn):
            np.savez_compressed(os.path.join(output_dir_tlearn, "{}_replication_test.npz".format(num)), **output)
        for num, output in enumerate(train_output_tlearn):
            np.savez_compressed(os.path.join(output_dir_tlearn, "{}_replication_train.npz".format(num)), **output)

        # Hydranet baseline and Hydranet T-reg
        for is_targeted_regularization in [False, True]:

            test_outputs_hy, train_output_hy = train_and_predict_hydra(num_treats, t, y, x, targeted_regularization=is_targeted_regularization,loss=loss, val_split=val_split, batch_size=batch_size)

            if is_targeted_regularization:
                output_dir_hy = os.path.join(simulation_output_dir, "Hydranet/targeted_regularization")
            else:
                output_dir_hy = os.path.join(simulation_output_dir, "Hydranet/baseline")
            os.makedirs(output_dir_hy, exist_ok=True)

            # Save outputs for each split
            for num, output in enumerate(test_outputs_hy):
                np.savez_compressed(os.path.join(output_dir_hy, "{}_replication_test.npz".format(num)),**output)
            for num, output in enumerate(train_output_hy):
                np.savez_compressed(os.path.join(output_dir_hy, "{}_replication_train.npz".format(num)),**output)


        # Back to back Dragonnets
        n_binary_estim = ['ate1_0', 'ate2_0', 'ate3_0', 'ate4_0']
        for estimator in n_binary_estim:
            x_d = pandas.concat([x, pandas.Series(t.flatten(), name='t')], axis=1)

            # Keep only the data for the current ate computation
            t_i = n_binary_estim.index(estimator) + 1
            x_d = x_d[(x_d.t == t_i) | (x_d.t == 0)]
            x_d.drop('t', axis=1, inplace=True)
            y_d = y[(t == t_i).flatten() | (t == 0).flatten()].reset_index(drop=True)
            y0_d = y0
            mu_0_d = mu_0

            if estimator == 'ate1_0':
                y1_d = y1
                mu_1_d = mu_1
            elif estimator == 'ate2_0':
                y1_d = y2
                mu_1_d = mu_2
            elif estimator == 'ate3_0':
                y1_d = y3
                mu_1_d = mu_3
            elif estimator == 'ate4_0':
                y1_d = y4
                mu_1_d = mu_4

            t_d = t[(t == t_i) | (t == 0)].reshape(-1, 1)

            # When computing ates other than ate1_0, change t to be equal to 1
            if t_i != 1:
                t_d[t_d > 1] = 1

            simulation_output_dir = os.path.join(output_dir, str(idx), 'b2bd', estimator)
            os.makedirs(simulation_output_dir, exist_ok=True)
            np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"), t=t_d, y0=y0_d, y1=y1_d, mu_0=mu_0_d,mu_1=mu_1_d)

            for is_targeted_regularization in [False, True]:
                test_outputs_b2bd, train_output_b2bd = train_and_predict_b2bd(t_d, y_d, x_d,
                                                                       targeted_regularization=is_targeted_regularization,
                                                                       loss=loss_dr,
                                                                       val_split=val_split,
                                                                       batch_size=batch_size)

                if is_targeted_regularization:
                    output_dir_b2bd = os.path.join(simulation_output_dir, "targeted_regularization")
                else:
                    output_dir_b2bd = os.path.join(simulation_output_dir, "baseline")
                os.makedirs(output_dir_b2bd, exist_ok=True)

                # Save outputs for each split
                for num, output in enumerate(test_outputs_b2bd):
                    np.savez_compressed(os.path.join(output_dir_b2bd, "{}_replication_test.npz".format(num)), **output)
                for num, output in enumerate(train_output_b2bd):
                    np.savez_compressed(os.path.join(output_dir_b2bd, "{}_replication_train.npz".format(num)), **output)

                
def collect_results_syn(input_dir):

    result_dict = {'train':
                    {'true': [],
                    'naive': {'baseline': []},
                    'b2bd': {'ate1_0': {'baseline': [], 'targeted_regularization': []},
                            'ate2_0': {'baseline': [], 'targeted_regularization': []},
                            'ate3_0': {'baseline': [], 'targeted_regularization': []},
                            'ate4_0': {'baseline': [], 'targeted_regularization': []}},
                    'T_learn': {'baseline': []},
                    'Hydranet': {'baseline': [], 'targeted_regularization': []}
                     },
                   'test':
                    {'true': [],
                    'naive': {'baseline': []},
                    'b2bd': {'ate1_0': {'baseline': [], 'targeted_regularization': []},
                            'ate2_0': {'baseline': [], 'targeted_regularization': []},
                            'ate3_0': {'baseline': [], 'targeted_regularization': []},
                            'ate4_0': {'baseline': [], 'targeted_regularization': []}},
                    'T_learn': {'baseline': []},
                    'Hydranet': {'baseline': [], 'targeted_regularization': []}
                     },
                   }

    estimator = ['Hydranet', 'b2bd', 'T_learn']
    input_folders = sorted(os.listdir(input_dir))

    # Retrieve values
    for idx, folder in enumerate(input_folders):
        # Repetition level (0, 1, 2...)
        # True data
        truth_dat_path = os.path.join(input_dir, folder, 'simulation_outputs.npz')
        a, b, c, d, e = load_truth(truth_dat_path)

        for estim in estimator:
            # Estimator level (0: b2bd, Hydra, T-learn; 1: b2bd, Hydra, T-learn; ...)
            # Result data
            estim_dat_path = os.path.join(input_dir, folder, estim)

            for split in ['train', 'test']:
                # Split level (0: b2bd: train, test; 0: Hydra: train, test; 0: T-learn: train, test...)

                if estim=='Hydranet':
                    base_dat_path = os.path.join(estim_dat_path, 'baseline', ('0_replication_' + split + '.npz'))

                    # From Hydranet take also the TRUE value and the BIASED value
                    q_t0, q_t1, q_t2, q_t3, q_t4, g, y, t, index = load_data(base_dat_path)
                    mu_0, mu_1, mu_2, mu_3, mu_4 = a[index], b[index], c[index], d[index], e[index]

                    truth1_0 = (mu_1 - mu_0).mean()
                    truth2_0 = (mu_2 - mu_0).mean()
                    truth3_0 = (mu_3 - mu_0).mean()
                    truth4_0 = (mu_4 - mu_0).mean()

                    biased1_0 = y[t == 1].mean() - y[t == 0].mean()
                    biased2_0 = y[t == 2].mean() - y[t == 0].mean()
                    biased3_0 = y[t == 3].mean() - y[t == 0].mean()
                    biased4_0 = y[t == 4].mean() - y[t == 0].mean()

                    result_dict[split]['true'].append([truth1_0, truth2_0, truth3_0, truth4_0])
                    result_dict[split]['naive']['baseline'].append([biased1_0, biased2_0, biased3_0, biased4_0])

                    for model in ['baseline', 'targeted_regularization']:
                        # Model level (when applicable)
                        base_dat_path = os.path.join(estim_dat_path, model, ('0_replication_' + split + '.npz'))
                        q_t0, q_t1, q_t2, q_t3, q_t4, g, y, t, index = load_data(base_dat_path)
                        # Compute estimator
                        psi = psi_naive(q_t0, q_t1, q_t2, q_t3, q_t4, g,truncate_level=0.)

                        result_dict[split][estim][model].append(psi)


                elif estim=='b2bd':
                    for ate in ['ate1_0', 'ate2_0', 'ate3_0', 'ate4_0']:
                        for model in ['baseline', 'targeted_regularization']:
                            # Model level (when applicable)
                            base_dat_path = os.path.join(estim_dat_path, ate, model, ('0_replication_' + split + '.npz'))
                            q_t0, q_t1, g, y, t, index = load_data_dr(base_dat_path)
                            # Compute estimator
                            psi = psi_naive_dr(q_t0, q_t1, g,truncate_level=0.)

                            result_dict[split][estim][ate][model].append(psi)

                    # Postprocess: join ates
                    result_dict[split][estim]['baseline'] = list(zip(result_dict[split]['b2bd']['ate1_0']['baseline'], result_dict[split]['b2bd']['ate2_0']['baseline'], result_dict[split]['b2bd']['ate3_0']['baseline'], result_dict[split]['b2bd']['ate4_0']['baseline']))
                    result_dict[split][estim]['targeted_regularization'] = list(zip(result_dict[split]['b2bd']['ate1_0']['targeted_regularization'], result_dict[split]['b2bd']['ate2_0']['targeted_regularization'], result_dict[split]['b2bd']['ate3_0']['targeted_regularization'], result_dict[split]['b2bd']['ate4_0']['targeted_regularization']))


                elif estim=='T_learn':
                    base_dat_path = os.path.join(estim_dat_path, 'baseline', ('0_replication_' + split + '.npz'))
                    q_t0, q_t1, q_t2, q_t3, q_t4, y, t, index = load_data_t(base_dat_path)
                    # Compute estimator
                    psi = psi_naive(q_t0, q_t1, q_t2, q_t3, q_t4, g,truncate_level=0.)

                    result_dict[split][estim]['baseline'].append(psi)

                else:
                    sys.exit('wrong estimator list')

    # Postprocess: delete individual ates of dragonnet
    for i in range(1,5):
        del result_dict['train']['b2bd']['ate{}_0'.format(i)]
        del result_dict['test']['b2bd']['ate{}_0'.format(i)]

    # Compute averages
    for estim in estimator:
        # Estimator level (0: b2bd, Hydra, T-learn; 1: b2bd, Hydra, T-learn; ...)

        for split in ['train', 'test']:
            # Split level (0: b2bd: train, test; 0: Hydra: train, test; 0: T-learn: train, test...)

            if estim == 'Hydranet':
                # True and naive values
                result_dict[split]['true'] = np.mean(result_dict[split]['true'], axis=0)
                result_dict[split]['naive']['baseline'] = np.mean(result_dict[split]['naive']['baseline'], axis=0)
                result_dict[split]['naive']['ae'] = np.sum(np.abs(result_dict[split]['true'] - result_dict[split]['naive']['baseline']))
                result_dict[split]['naive']['pe'] = result_dict[split]['naive']['ae']/np.sum(result_dict[split]['true']) *100

                for model in ['baseline', 'targeted_regularization']:
                    result_dict[split][estim][model] = np.mean(result_dict[split][estim][model], axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.sum(np.abs(result_dict[split][estim][model] - result_dict[split]['true']))
                    result_dict[split][estim]['{}_pe'.format(model)] = result_dict[split][estim]['{}_ae'.format(model)]/np.sum(result_dict[split]['true']) *100

            elif estim == 'b2bd':
                for model in ['baseline', 'targeted_regularization']:
                    result_dict[split][estim][model] = np.mean(result_dict[split][estim][model], axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.sum(np.abs(result_dict[split][estim][model] - result_dict[split]['true']))
                    result_dict[split][estim]['{}_pe'.format(model)] = result_dict[split][estim]['{}_ae'.format(model)]/np.sum(result_dict[split]['true']) *100

            elif estim == 'T_learn':
                result_dict[split][estim]['baseline'] = np.mean(result_dict[split][estim]['baseline'], axis=0)
                result_dict[split][estim]['baseline_ae'] = np.sum(np.abs(result_dict[split][estim]['baseline']- result_dict[split]['true']))
                result_dict[split][estim]['baseline_pe'] = result_dict[split][estim]['baseline_ae']/np.sum(result_dict[split]['true']) *100

    return result_dict

    '''        
    # Compute error, from average true and average estimated values
    
    biased_error = abs(true_val - biased_val).sum(axis=1) #change to sum
    biased_error_val = biased_error.mean()
    hydranet_error = abs(estim_dict[model] - true_val).sum(axis=1)
    hydranet_error_val = hydranet_error.mean()
    
    # Check big-small error t-reg difference
    ####################################################
    if model=='baseline':
        baseline_low = np.where(hydranet_error<hydranet_error_val)[0]
        baseline_high = np.where(hydranet_error>hydranet_error_val)[0]
    
    hydranet_error = hydranet_error[baseline_high]
    hydranet_error_val = hydranet_error.mean()
    ####################################################
    
    # Compute error, from average true and average estimated values
    result_dict['N'] = n_reps
    result_dict['Avg true value'] = np.mean(true_val, axis=0)
    result_dict['Avg biased estimate'] = np.mean(biased_val, axis=0)
    result_dict['Naive estimator error'] = biased_error_val
    result_dict[model]['Avg hydranet est.'] = np.mean(estim_dict[model], axis=0)
    result_dict[model]['Hydranet error'] = hydranet_error_val
    
    # Compute bootstrap 95% CI intervals for the error
    alg_name = '{} est error CIs'.format(model)
    naive_ci_l, naive_ci_u = bootstrap((biased_error,), statistic=np.mean, method='basic', random_state = 3).confidence_interval
    hydra_ci_l, hydra_ci_u = bootstrap((hydranet_error,), statistic=np.mean, method='basic', random_state = 3).confidence_interval
    result_dict['Naive est error CIs'] = naive_ci_l, naive_ci_u
    result_dict[alg_name] = hydra_ci_l, hydra_ci_u
#'''


def analyse_results_syn(all_res_dict, main_param, output_dir):
    # Print figures and generate tables
    reform = {(outerKey, innerKey): values for outerKey, innerDict in all_res_dict[main_param].items() for innerKey, values in innerDict.items()}
    all_res_df = pd.DataFrame(reform)
    df_train = all_res_df.iloc[:,all_res_df.columns.get_level_values(1)=='train']
    df_test = all_res_df.iloc[:,all_res_df.columns.get_level_values(1)=='test']
    df_train.columns = df_train.columns.droplevel(1)
    df_test.columns = df_test.columns.droplevel(1)
    df_train = df_train.T
    df_test = df_test.T

    fig, ax = plt.subplots()

    line1, = ax.plot(df_train['naive'].apply(lambda x: x['ae']), marker='o')
    line2, = ax.plot(df_train['b2bd'].apply(lambda x: x['baseline_ae']), marker='o')
    line3, = ax.plot(df_train['b2bd'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    line4, = ax.plot(df_train['T_learn'].apply(lambda x: x['baseline_ae']), marker='o')
    line5, = ax.plot(df_train['Hydranet'].apply(lambda x: x['baseline_ae']), marker='o')
    line6, = ax.plot(df_train['Hydranet'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    plt.legend(handles=[line1, line2, line3, line4, line5, line6], \
               labels=['Naive', 'B2BD Baseline', 'B2BD T-reg', 'T-learner', 'Hydranet Baseline', 'Hydranet T-reg'])
    plt.xlabel('Error')
    plt.ylabel(main_param)
    os.makedirs(os.path.join(output_dir, main_param), exist_ok=True)
    fig.savefig(os.path.join(output_dir, main_param + '_in-sample'))
    #plt.show()

    fig, ax = plt.subplots()

    line1, = ax.plot(df_test['naive'].apply(lambda x: x['ae']), marker='o')
    line2, = ax.plot(df_test['b2bd'].apply(lambda x: x['baseline_ae']), marker='o')
    line3, = ax.plot(df_test['b2bd'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    line4, = ax.plot(df_test['T_learn'].apply(lambda x: x['baseline_ae']), marker='o')
    line5, = ax.plot(df_test['Hydranet'].apply(lambda x: x['baseline_ae']), marker='o')
    line6, = ax.plot(df_test['Hydranet'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    plt.legend(handles=[line1, line2, line3, line4, line5, line6],\
               labels=['Naive', 'B2BD Baseline', 'B2BD T-reg', 'T-learner', 'Hydranet Baseline', 'Hydranet T-reg'])
    plt.xlabel('Percentual error')
    plt.ylabel(main_param)
    fig.savefig(os.path.join(output_dir, main_param + '_out-sample'))
    #plt.show()


def collect_results_ihdp(input_dir):
    result_dict = {'train':
                       {'true': [],
                        'naive': {'baseline': []},
                        'b2bd': {'ate1_0': {'baseline': [], 'targeted_regularization': []},
                                 'ate2_0': {'baseline': [], 'targeted_regularization': []},
                                 'ate3_0': {'baseline': [], 'targeted_regularization': []},
                                 'ate4_0': {'baseline': [], 'targeted_regularization': []}},
                        'T_learn': {'baseline': []},
                        'Hydranet': {'baseline': [], 'targeted_regularization': []}
                        },
                   'test':
                       {'true': [],
                        'naive': {'baseline': []},
                        'b2bd': {'ate1_0': {'baseline': [], 'targeted_regularization': []},
                                 'ate2_0': {'baseline': [], 'targeted_regularization': []},
                                 'ate3_0': {'baseline': [], 'targeted_regularization': []},
                                 'ate4_0': {'baseline': [], 'targeted_regularization': []}},
                        'T_learn': {'baseline': []},
                        'Hydranet': {'baseline': [], 'targeted_regularization': []}
                        },
                   }

    estimator = ['Hydranet', 'b2bd', 'T_learn']
    input_folders = sorted(os.listdir(input_dir))

    # Retrieve values
    for idx, folder in enumerate(input_folders):
        # Repetition level (0, 1, 2...)
        # True data
        truth_dat_path = os.path.join(input_dir, folder, 'simulation_outputs.npz')
        a, b, c, d, e = load_truth(truth_dat_path)

        for estim in estimator:
            # Estimator level (0: b2bd, Hydra, T-learn; 1: b2bd, Hydra, T-learn; ...)
            # Result data
            estim_dat_path = os.path.join(input_dir, folder, estim)

            for split in ['train', 'test']:
                # Split level (0: b2bd: train, test; 0: Hydra: train, test; 0: T-learn: train, test...)

                if estim == 'Hydranet':
                    base_dat_path = os.path.join(estim_dat_path, 'baseline', ('0_replication_' + split + '.npz'))

                    # From Hydranet take also the TRUE value and the BIASED value
                    q_t0, q_t1, q_t2, q_t3, q_t4, g, y, t, index = load_data(base_dat_path)
                    mu_0, mu_1, mu_2, mu_3, mu_4 = a[index], b[index], c[index], d[index], e[index]

                    truth1_0 = (mu_1 - mu_0).mean()
                    truth2_0 = (mu_2 - mu_0).mean()
                    truth3_0 = (mu_3 - mu_0).mean()
                    truth4_0 = (mu_4 - mu_0).mean()

                    biased1_0 = y[t == 1].mean() - y[t == 0].mean()
                    biased2_0 = y[t == 2].mean() - y[t == 0].mean()
                    biased3_0 = y[t == 3].mean() - y[t == 0].mean()
                    biased4_0 = y[t == 4].mean() - y[t == 0].mean()

                    result_dict[split]['true'].append([truth1_0, truth2_0, truth3_0, truth4_0])
                    result_dict[split]['naive']['baseline'].append([biased1_0, biased2_0, biased3_0, biased4_0])

                    for model in ['baseline', 'targeted_regularization']:
                        # Model level (when applicable)
                        base_dat_path = os.path.join(estim_dat_path, model, ('0_replication_' + split + '.npz'))
                        q_t0, q_t1, q_t2, q_t3, q_t4, g, y, t, index = load_data(base_dat_path)
                        # Compute estimator
                        psi = psi_naive(q_t0, q_t1, q_t2, q_t3, q_t4, g, truncate_level=0.)

                        result_dict[split][estim][model].append(psi)


                elif estim == 'b2bd':
                    for ate in ['ate1_0', 'ate2_0', 'ate3_0', 'ate4_0']:
                        for model in ['baseline', 'targeted_regularization']:
                            # Model level (when applicable)
                            base_dat_path = os.path.join(estim_dat_path, ate, model,
                                                         ('0_replication_' + split + '.npz'))
                            q_t0, q_t1, g, y, t, index = load_data_dr(base_dat_path)
                            # Compute estimator
                            psi = psi_naive_dr(q_t0, q_t1, g, truncate_level=0.)

                            result_dict[split][estim][ate][model].append(psi)

                    # Postprocess: join ates
                    result_dict[split][estim]['baseline'] = list(zip(result_dict[split]['b2bd']['ate1_0']['baseline'],
                                                                     result_dict[split]['b2bd']['ate2_0']['baseline'],
                                                                     result_dict[split]['b2bd']['ate3_0']['baseline'],
                                                                     result_dict[split]['b2bd']['ate4_0']['baseline']))
                    result_dict[split][estim]['targeted_regularization'] = list(
                        zip(result_dict[split]['b2bd']['ate1_0']['targeted_regularization'],
                            result_dict[split]['b2bd']['ate2_0']['targeted_regularization'],
                            result_dict[split]['b2bd']['ate3_0']['targeted_regularization'],
                            result_dict[split]['b2bd']['ate4_0']['targeted_regularization']))


                elif estim == 'T_learn':
                    base_dat_path = os.path.join(estim_dat_path, 'baseline', ('0_replication_' + split + '.npz'))
                    q_t0, q_t1, q_t2, q_t3, q_t4, y, t, index = load_data_t(base_dat_path)
                    # Compute estimator
                    psi = psi_naive(q_t0, q_t1, q_t2, q_t3, q_t4, g, truncate_level=0.)

                    result_dict[split][estim]['baseline'].append(psi)

                else:
                    sys.exit('wrong estimator list')

    # Postprocess: delete individual ates of dragonnet
    for i in range(1, 5):
        del result_dict['train']['b2bd']['ate{}_0'.format(i)]
        del result_dict['test']['b2bd']['ate{}_0'.format(i)]

    # Compute averages
    for estim in estimator:
        # Estimator level (0: b2bd, Hydra, T-learn; 1: b2bd, Hydra, T-learn; ...)

        for split in ['train', 'test']:
            # Split level (0: b2bd: train, test; 0: Hydra: train, test; 0: T-learn: train, test...)

            if estim == 'Hydranet':
                # True and naive values
                result_dict[split]['true'] = np.mean(result_dict[split]['true'], axis=0)
                result_dict[split]['naive']['baseline'] = np.mean(result_dict[split]['naive']['baseline'], axis=0)
                result_dict[split]['naive']['ae'] = np.sum(
                    np.abs(result_dict[split]['true'] - result_dict[split]['naive']['baseline']))
                result_dict[split]['naive']['pe'] = result_dict[split]['naive']['ae'] / np.sum(
                    result_dict[split]['true']) * 100

                for model in ['baseline', 'targeted_regularization']:
                    result_dict[split][estim][model] = np.mean(result_dict[split][estim][model], axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.sum(
                        np.abs(result_dict[split][estim][model] - result_dict[split]['true']))
                    result_dict[split][estim]['{}_pe'.format(model)] = result_dict[split][estim][
                                                                           '{}_ae'.format(model)] / np.sum(
                        result_dict[split]['true']) * 100

            elif estim == 'b2bd':
                for model in ['baseline', 'targeted_regularization']:
                    result_dict[split][estim][model] = np.mean(result_dict[split][estim][model], axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.sum(
                        np.abs(result_dict[split][estim][model] - result_dict[split]['true']))
                    result_dict[split][estim]['{}_pe'.format(model)] = result_dict[split][estim][
                                                                           '{}_ae'.format(model)] / np.sum(
                        result_dict[split]['true']) * 100

            elif estim == 'T_learn':
                result_dict[split][estim]['baseline'] = np.mean(result_dict[split][estim]['baseline'], axis=0)
                result_dict[split][estim]['baseline_ae'] = np.sum(
                    np.abs(result_dict[split][estim]['baseline'] - result_dict[split]['true']))
                result_dict[split][estim]['baseline_pe'] = result_dict[split][estim]['baseline_ae'] / np.sum(
                    result_dict[split]['true']) * 100

    return result_dict

    '''        
    # Compute error, from average true and average estimated values

    biased_error = abs(true_val - biased_val).sum(axis=1) #change to sum
    biased_error_val = biased_error.mean()
    hydranet_error = abs(estim_dict[model] - true_val).sum(axis=1)
    hydranet_error_val = hydranet_error.mean()

    # Check big-small error t-reg difference
    ####################################################
    if model=='baseline':
        baseline_low = np.where(hydranet_error<hydranet_error_val)[0]
        baseline_high = np.where(hydranet_error>hydranet_error_val)[0]

    hydranet_error = hydranet_error[baseline_high]
    hydranet_error_val = hydranet_error.mean()
    ####################################################

    # Compute error, from average true and average estimated values
    result_dict['N'] = n_reps
    result_dict['Avg true value'] = np.mean(true_val, axis=0)
    result_dict['Avg biased estimate'] = np.mean(biased_val, axis=0)
    result_dict['Naive estimator error'] = biased_error_val
    result_dict[model]['Avg hydranet est.'] = np.mean(estim_dict[model], axis=0)
    result_dict[model]['Hydranet error'] = hydranet_error_val

    # Compute bootstrap 95% CI intervals for the error
    alg_name = '{} est error CIs'.format(model)
    naive_ci_l, naive_ci_u = bootstrap((biased_error,), statistic=np.mean, method='basic', random_state = 3).confidence_interval
    hydra_ci_l, hydra_ci_u = bootstrap((hydranet_error,), statistic=np.mean, method='basic', random_state = 3).confidence_interval
    result_dict['Naive est error CIs'] = naive_ci_l, naive_ci_u
    result_dict[alg_name] = hydra_ci_l, hydra_ci_u
#'''


def analyse_results_ihdp(all_res_dict, output_dir):
    # Print figures and generate tables
    all_res_df = pd.DataFrame(all_res_dict)
    df_train = all_res_df['train']
    df_test = all_res_df['test']

    file = os.path.join(output_dir, 'summary.txt')
    with open(file, 'w') as sumfile:
        sumfile.write('In-sample\n')
        sumfile.write('Naive estimator error: {}\n'.format(df_train['naive']['ae']))
        sumfile.write('B2BD baseline error: {}\n'.format(df_train['b2bd']['baseline_ae']))
        sumfile.write('B2BD T-reg error: {}\n'.format(df_train['b2bd']['targeted_regularization_ae']))
        sumfile.write('T-learner error: {}\n'.format(df_train['T_learn']['baseline_ae']))
        sumfile.write('Hydranet baseline error: {}\n'.format(df_train['Hydranet']['baseline_ae']))
        sumfile.write('Hydranet T-reg error: {}\n'.format(df_train['Hydranet']['targeted_regularization_ae']))
        sumfile.write('*******\n')

        sumfile.write('Out-sample\n')
        sumfile.write('Naive estimator error: {}\n'.format(df_test['naive']['ae']))
        sumfile.write('B2BD baseline error: {}\n'.format(df_test['b2bd']['baseline_ae']))
        sumfile.write('B2BD T-reg error: {}\n'.format(df_test['b2bd']['targeted_regularization_ae']))
        sumfile.write('T-learner error: {}\n'.format(df_test['T_learn']['baseline_ae']))
        sumfile.write('Hydranet baseline error: {}\n'.format(df_test['Hydranet']['baseline_ae']))
        sumfile.write('Hydranet T-reg error: {}\n'.format(df_test['Hydranet']['targeted_regularization_ae']))
        sumfile.write('*******\n')

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_treats", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="synthetic", choices=['synthetic', 'ihdp'])
    parser.add_argument("--input_dir", type=str, default="/home/bvelasco/Hydranet/")
    parser.add_argument("--output_dir", type=str, default="/home/bvelasco/Hydranet/Results/")
    parser.add_argument("--main_param", type=str, choices=["data_size", 'n_confs', 'bias'])
    parser.add_argument("--device", type=str, default='GPU', choices=["GPU", "CPU"])
    parser.add_argument('--loss', type=eval, default=hydranet_loss)
    parser.add_argument('--loss_dr', type=eval, default=dragonnet_loss_binarycross_dr)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--Train", type=bool, default=False)
    parser.add_argument("--Analyze", type=bool, default=False)

    args = parser.parse_args()
    num_treats = args.num_treats  # or 10
    dataset = args.dataset  # or 'synthetic'
    main_param = args.main_param  # or data_size or n_confs
    device = args.device
    input_dir = args.input_dir
    output_dir = args.output_dir
    loss = args.loss
    loss_dr = args.loss_dr
    val_split = args.val_split
    batch_size = args.batch_size
    Train = args.Train
    Analyze = args.Analyze

    # System arguments
    
    main_param_dict = {'bias':[2,5,10,30],
                        'n_confs':[2, 5, 10, 18],
                        'data_size':[1000, 2000, 5000, 10000]
                      }
    all_res_dict = {'bias': {2:[], 5:[], 10:[], 30:[]},
                       'n_confs': {2:[], 5:[], 10:[], 18:[]},
                       'data_size': {1000:[], 2000:[], 5000:[], 10000:[]}
                       }
    tf.compat.v1.disable_eager_execution()

    # Set seeds
    random.seed(1)
    np.random.seed(1)

    # Train
    if Train:
        print('Train')
        # Configure GPU resources if training in GPU
        if device == 'CPU':
            print('Training in CPU')
        elif device == 'GPU':
            # tf.compat.v1.set_random_seed(1)
            print('Training in GPU')
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
            else:
                print('Device is set to "GPU" but no GPU was found')
                sys.exit()

        input_dir = os.path.join(input_dir, 'Input_data/')
        output_dir = os.path.join(output_dir, 'Results_NN/')

        with tf.device(device):
            if dataset=='synthetic':
                for val in main_param_dict[main_param]:
                    # Build paths
                    input_dir_ = input_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param) + '/{}/'.format(val)
                    output_dir_ = output_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param) + '/{}/'.format(val)
                    run_train(input_dir=input_dir_, output_dir=output_dir_, dataset=dataset, num_treats=num_treats, loss=loss, loss_dr=loss_dr, val_split=val_split, batch_size=batch_size)
            elif dataset=='ihdp':
                # Build paths
                input_dir_ = input_dir + dataset + '/{}_treats/'.format(num_treats)
                output_dir_ = output_dir + dataset + '/{}_treats/'.format(num_treats)
                run_train(input_dir=input_dir_, output_dir=output_dir_, dataset=dataset, num_treats=num_treats, loss=loss, loss_dr=loss_dr, val_split=val_split, batch_size=batch_size)
    else:
        print('Do not train')
            
    # Analyze
    if Analyze:
        print('Analyze')
        input_dir = os.path.join(input_dir, 'Results/Results_NN/')
        output_dir = os.path.join(output_dir, 'Results_CI/')

        if dataset == 'synthetic':
            for val in main_param_dict[main_param]:
                # Build paths
                input_dir_ = input_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param) + '/{}/'.format(val)
                all_res_dict[main_param][val] = collect_results_syn(input_dir_)
            output_dir_ = output_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param)
            analyse_results_syn(all_res_dict, main_param, output_dir_)

        elif dataset == 'ihdp':
            # Build paths
            input_dir_ = input_dir + dataset + '/{}_treats/'.format(num_treats)
            output_dir_ = output_dir + dataset + '/{}_treats/'.format(num_treats)
            res_dict = collect_results_ihdp(input_dir_)
            analyse_results_ihdp(res_dict, output_dir_)

    else:
        print('Do not analyze')

    print('Done')
    
if __name__ == '__main__':
    main()
