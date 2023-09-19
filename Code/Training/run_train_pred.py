from shared_pkgs.imports import *
from Helpers.helper_funcs import *
from Neural_Net.neural_net import *
from Neural_Net.losses import *

def train_and_predict_hydra(num_treats, t, y_unscaled, x_unscaled, targeted_regularization, loss, val_split, batch_size, eager_exec):
    print('Hydranet, T-reg:', targeted_regularization)
    verbose = 0
    
    y_unscaled = y_unscaled.values.reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    x_scaler = StandardScaler().fit(x_unscaled)
    x = x_scaler.transform(x_unscaled)
        
    train_outputs = []
    test_outputs = []

    hydranet = make_hydranet(x.shape[1], num_treats, 0.01)

    metrics = [hydranet_loss, regression_loss, categorical_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=4, hydranet_loss_=loss)
    else:
        loss = loss

    # Get train and test indexes
    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=val_split, random_state=1, shuffle=True)

    x_train, x_test = x[train_index,:], x[test_index,:] # when scaled
    #x_train, x_test = x.iloc[train_index], x.iloc[test_index] # when unscaled
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)

    # With Adam
    hydranet.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=loss, metrics=metrics, run_eagerly=eager_exec)

    # Save best model callback
    save_best_model_adam = SaveBestModel()
              
    adam_callbacks = [
        save_best_model_adam,
        TerminateOnNaN(),
        #PlotLearning(),
        EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=verbose, mode='auto', min_delta=1e-8, cooldown=0, min_lr=0)
        ]
    
    hydranet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split, epochs=100,
                  batch_size=batch_size, verbose=verbose)
    
    # Set best weigts from callback
    hydranet.set_weights(save_best_model_adam.best_weights)
    
    # with SGD
    sgd_lr = 1e-5
    momentum = 0.9
    hydranet.compile(optimizer=SGD(learning_rate=sgd_lr, momentum=momentum, nesterov=True),
                      loss=loss, metrics=metrics, run_eagerly=eager_exec)
    
    # Save best model callback
    save_best_model_sgd = SaveBestModel()

    sgd_callbacks = [
        save_best_model_sgd,
        TerminateOnNaN(),
        #PlotLearning(),
        EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=verbose, mode='auto',min_delta=0., cooldown=0, min_lr=0)
        ]
    
    hydranet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split, epochs=200,
                  batch_size=batch_size, verbose=verbose)
    
    # Set best weigts from callback
    hydranet.set_weights(save_best_model_sgd.best_weights)

    # Plot metrics to monitor the training process
    '''plt.figure()
    plt.plot(hydranet.history.history['loss'])
    plt.plot(hydranet.history.history['val_loss'])
    plt.legend(["Train", "Test"])
    plt.title("Hydranet Loss")
    plt.show() # Training and validation losses

    plt.figure()
    plt.plot(hydranet.history.history['treatment_accuracy'])
    plt.plot(hydranet.history.history['val_treatment_accuracy'])
    plt.legend(["Train", "Test"])
    plt.title("Hydranet Treatment prediction accuracy")
    plt.show()  # Treatment prediction accuracy
    
    plt.figure()
    plt.plot(hydranet.history.history['track_epsilon'])
    plt.plot(hydranet.history.history['val_track_epsilon'])
    plt.legend(["Train", "Test"])
    plt.title("Epsilon (T-reg = {})".format(targeted_regularization))
    plt.show() # Epsilon'''

    
    yt_hat_test = hydranet.predict(x_test)
    yt_hat_train = hydranet.predict(x_train)
    
    train_outputs += [split_output(yt_hat_train, t_train, y_train, y_scaler, x_scaler, x_train, train_index)]
    test_outputs += [split_output(yt_hat_test, t_test, y_test, y_scaler, x_scaler, x_test, test_index)]

    K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_b2bd(t, y_unscaled, x, targeted_regularization, loss, val_split, batch_size):
    print('B2BD, T-reg:', targeted_regularization)
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
    opt = Adam(learning_rate=1e-3)

    dragonnet.compile(
        optimizer=opt,
        loss=loss, metrics=metrics,
        run_eagerly=False)

    adam_callbacks = [
        TerminateOnNaN(),
        #PlotLearning(),
        EarlyStopping(monitor='val_loss', patience=5, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',min_delta=1e-8, cooldown=0, min_lr=0)]

    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split, epochs=150,
                  batch_size=batch_size, verbose=verbose)

    # with SGD
    sgd_lr = 1e-5
    momentum = 0.9
    opt = SGD(learning_rate=sgd_lr, momentum=momentum, nesterov=True)
    dragonnet.compile(optimizer=opt,
        loss=loss, metrics=metrics,
        run_eagerly=False)

    sgd_callbacks = [
        TerminateOnNaN(),
        #PlotLearning(),
        EarlyStopping(monitor='val_loss', patience=20, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',min_delta=0., cooldown=0, min_lr=0)]

    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split, epochs=200,
                  batch_size=batch_size, verbose=verbose)


    # Plot metrics to monitor the training process
    '''plt.figure()
    plt.plot(dragonnet.history.history['loss'])
    plt.plot(dragonnet.history.history['val_loss'])
    plt.legend(["Train", "Test"])
    plt.title("B2BD Loss")
    plt.show() # Training and validation losses

    plt.figure()
    plt.plot(dragonnet.history.history['treatment_accuracy_dr'])
    plt.plot(dragonnet.history.history['val_treatment_accuracy_dr'])
    plt.legend(["Train", "Test"])
    plt.title("B2BD Treatment prediction accuracy")
    plt.show() # Treatment prediction accuracy

    plt.figure()
    plt.plot(dragonnet.history.history['track_epsilon_dr'])
    plt.plot(dragonnet.history.history['val_track_epsilon_dr'])
    plt.legend(["Train", "Test"])
    plt.title("B2BD Epsilon value (Regularization term)")
    plt.show() # Epsilon'''

    yt_hat_test = dragonnet.predict(x_test)
    yt_hat_train = dragonnet.predict(x_train)

    train_outputs += [split_output_dr(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
    test_outputs += [split_output_dr(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]

    K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_tlearn(dataset, t, y_unscaled, x, val_split):
    print('T-learn')
    
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


def run_train(input_dir, output_dir, dataset, num_treats, loss, loss_dr, val_split, batch_size, reps, eager_exec):    
    #tf.debugging.enable_check_numerics()

    simulation_files = sorted(glob.glob("{}/*.csv".format(input_dir)), key=lambda x: int( x.split('.')[0].split('_')[-1] ))[reps[0]:reps[1]]

    for idx, simulation_file in enumerate(simulation_files):
        print(simulation_file)

        simulation_output_dir = os.path.join(output_dir, str(idx))

        os.makedirs(simulation_output_dir, exist_ok=True)

        x = load_and_format_covariates(simulation_file, dataset)
        
        t, y, y0, y1, y2, y3, y4, mu_0, mu_1, mu_2, mu_3, mu_4 = load_other_vars(simulation_file)
        
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),t=t, y0=y0, y1=y1, y2=y2, y3=y3, y4=y4, mu_0=mu_0, mu_1=mu_1, mu_2=mu_2, mu_3=mu_3, mu_4=mu_4)
        
        ############# RUN THE DIFFERENT ESTIMATORS ##############
        
        ################# T-learner
        test_outputs_tlearn, train_output_tlearn = train_and_predict_tlearn(dataset, t, y, x, val_split=val_split)

        output_dir_tlearn = os.path.join(simulation_output_dir, "T_learn/baseline")
        os.makedirs(output_dir_tlearn, exist_ok=True)

        # Save outputs for each split
        for num, output in enumerate(test_outputs_tlearn):
            np.savez_compressed(os.path.join(output_dir_tlearn, "{}_replication_test.npz".format(num)), **output)
        for num, output in enumerate(train_output_tlearn):
            np.savez_compressed(os.path.join(output_dir_tlearn, "{}_replication_train.npz".format(num)), **output)

        ################# Hydranet baseline and Hydranet T-reg
        for is_targeted_regularization in [False, True]:

            test_outputs_hy, train_output_hy = train_and_predict_hydra(num_treats, t, y, x, targeted_regularization=is_targeted_regularization,loss=loss, val_split=val_split, batch_size=batch_size, eager_exec=eager_exec)

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


        ################# Back to back Dragonnets
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
