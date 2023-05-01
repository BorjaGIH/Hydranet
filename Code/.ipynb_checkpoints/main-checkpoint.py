from shared_pkgs.imports import *
from Helpers.helper_funcs import *
from Neural_Net.neural_net import *
from Neural_Net.losses import *
from Estimation.estimators import *


def train_and_predict_dragons(t, y_unscaled, x, targeted_regularization=False, output_dir='',
                              knob_loss=dragonnet_loss_cross, ratio=1., dragon='', val_split=0.2, batch_size=64):
    verbose = 0
    y_unscaled = y_unscaled.values.reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []

    if dragon == 'tarnet':
        dragonnet = make_tarnet(x.shape[1], 0.01)

    elif dragon == 'dragonnet':
        print("I am here making dragonnet")
        dragonnet = make_dragonnet(x.shape[1], 0.01)
        dot_img_file = '/model_1.png'
        #tf.keras.utils.plot_model(dragonnet, to_file=dot_img_file, show_shapes=True)

    metrics = [regression_loss, categorical_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss

    # Get train and test indexes, then shuffle
    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=1, shuffle=False)
    #random.shuffle(train_index)
    random.shuffle(test_index)
    
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)

    import time;
    start_time = time.time()

    # With Adam
    dragonnet.compile(
        optimizer=Adam(lr=1e-3),
        loss=loss, metrics=metrics, run_eagerly=False)

    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0)
    ]
    
    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=100,
                  batch_size=batch_size, verbose=verbose)
    
    # with SGD
    sgd_lr = 1e-5
    momentum = 0.9
    dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), 
                      loss=loss, metrics=metrics, run_eagerly=False)
    
    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]
    
    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=300,
                  batch_size=batch_size, verbose=verbose)
    
    elapsed_time = time.time() - start_time

    # Plot metrics to monitor the training process
    plt.figure()
    plt.plot(dragonnet.history.history['loss'])
    plt.plot(dragonnet.history.history['val_loss'])
    plt.legend(["Train", "Test"])
    plt.title("Loss")
    plt.show() # Training and validation losses
    
    plt.figure()
    plt.plot(dragonnet.history.history['track_epsilon'])
    plt.plot(dragonnet.history.history['val_track_epsilon'])
    plt.legend(["Train", "Test"])
    plt.title("Epsilon value (Regularization term)")
    plt.show() # Epsilon
    
    print("elapsed_time: ", elapsed_time)
    
    yt_hat_test = dragonnet.predict(x_test)
    yt_hat_train = dragonnet.predict(x_train)
    
    print("Train: ", end="")
    train_outputs += [split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
    
    print("Test: ", end="")
    test_outputs += [split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]
    
    print("*****************************")
    
    K.clear_session()

    return test_outputs, train_outputs


def run_ihdp(data_base_dir, #='/home/bvelasco/Hydranet_script/Input_data/ihdp/3_treats',
             output_dir, #='/home/bvelasco/Hydranet_script/Results/Results_NN/ihdp/3_treats/',
             knob_loss=dragonnet_loss_cross,
             ratio=1., dragon=''):
    print("The dragon is {}".format(dragon))

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))[1:50]

    for idx, simulation_file in enumerate(simulation_files):

        simulation_output_dir = os.path.join(output_dir, str(idx))

        os.makedirs(simulation_output_dir, exist_ok=True)

        x = load_and_format_covariates_ihdp(simulation_file)
        t, y, y0, y1, y2, mu_0, mu_1, mu_2 = load_all_other_crap(simulation_file)

        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            t=t, y0=y0, y1=y1, y2=y2, mu_0=mu_0, mu_1=mu_1, mu_2=mu_2)

        for is_targeted_regularization in [False, True]:
        #for is_targeted_regularization in [True]:
            print("Is targeted regularization: {}".format(is_targeted_regularization))
            if dragon == 'nednet':
                test_outputs, train_output = train_and_predict_ned(t, y, x,
                                                                   targeted_regularization=is_targeted_regularization,
                                                                   output_dir=simulation_output_dir,
                                                                   knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                                   val_split=0.2, batch_size=64)
            else:
                
                test_outputs, train_output = train_and_predict_dragons(t, y, x,
                                                                       targeted_regularization=is_targeted_regularization,
                                                                       output_dir=simulation_output_dir,
                                                                       knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                                       val_split=0.2, batch_size=64)
                

            if is_targeted_regularization:
                train_output_dir = os.path.join(simulation_output_dir, "targeted_regularization")
            else:
                train_output_dir = os.path.join(simulation_output_dir, "baseline")
            os.makedirs(train_output_dir, exist_ok=True)

            # save the outputs of for each split (1 per npz file)
            for num, output in enumerate(test_outputs):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),**output)

            for num, output in enumerate(train_output):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),**output)

                
def make_table(train_test, n_replication, n_reps):
    
    result_dict = {'N': [], 'Avg true value': [], 'Avg biased estimate': [], 'Naive estimator error': [],
                   'baseline': {'Avg hydranet est.': [], 'Hydranet error': []},
                   'targeted_regularization': { 'Avg hydranet est.': [], 'Hydranet error': []},
                   'Other params': [],
                 }
    
    estim_dict = {
            'baseline': [], 'targeted_regularization': []
            }
    
    #for model in ['baseline']:
    for model in ['baseline', 'targeted_regularization']:
        true_val, biased_val = [], []
        for rep in range(n_replication):
            q_t0, q_t1, q_t2, g, t, y_dragon, index, eps = load_data(rep, model, train_test, n_reps)
            a, b, c = load_truth(rep, n_reps)
            mu_0, mu_1, mu_2 = a[index], b[index], c[index]
            data_len = len(y_dragon)

            truth1_0 = (mu_1 - mu_0).mean()
            truth2_0 = (mu_2 - mu_0).mean()
            
            biased1_0 = y_dragon[t==1].mean() - y_dragon[t==0].mean()
            biased2_0 = y_dragon[t==2].mean() - y_dragon[t==0].mean()

            psi_vn, psi_n = get_estimate(q_t0, q_t1, q_t2, g, t, y_dragon, index, eps, truncate_level=0.)

            true_val.append([truth1_0, truth2_0])
            biased_val.append([biased1_0, biased2_0])
            estim_dict[model].append(psi_n)
        
        # Compute error, from average true and average estimated values
        true_val = np.asarray(true_val)
        biased_val = np.asarray(biased_val)
        
        biased_error = abs(true_val - biased_val).sum(axis=1) #change to sum
        biased_error_val = biased_error.mean()
        hydranet_error = abs(estim_dict[model] - true_val).sum(axis=1)
        hydranet_error_val = hydranet_error.mean()
        
        # Check big-small error t-reg difference
        ####################################################
        '''if model=='baseline':
            baseline_low = np.where(hydranet_error<hydranet_error_val)[0]
            baseline_high = np.where(hydranet_error>hydranet_error_val)[0]
        
        hydranet_error = hydranet_error[baseline_high]
        hydranet_error_val = hydranet_error.mean()'''
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
        
    
    return result_dict


def main():
    random.seed(1)
    np.random.seed(1)
    
    n_reps = [1]
    Train = True
    Analyze = True
        
    if Train:
        print('Train')
        tf.compat.v1.set_random_seed(1)
        tf.compat.v1.disable_eager_execution()
        
        for n_rep in n_reps:
            with tf.device('GPU:1'):
                run_ihdp(data_base_dir='/home/bvelasco/Hydranet_script/Input_data/ihdp/3_treats/reps_{}'.format(n_rep),
                         output_dir='/home/bvelasco/Hydranet_script/Results/Results_NN/ihdp/3_treats/reps_{}/'.format(n_rep),
                         dragon='dragonnet', ratio=5) # ratio is the beta parameter of the targeted regularization loss function
    else:
        print('Do not train')
            
    # Analyze
    if Analyze:
        print('Analyze')
        result_table = []
        train_or_test = 'train'
        
        result_table.append(Parallel(n_jobs=15)(delayed(make_table)(train_test=train_or_test, n_replication=15, n_reps=n_rep)
                           for n_rep in [1]))
        pandas.DataFrame(result_table[0]).to_csv('/home/bvelasco/Hydranet_script/Results/Results_CI/ihdp/3_treats/results_{}.csv'.format( train_or_test))
    else:
        print('Do not analyze')
    
if __name__ == '__main__':
    main()