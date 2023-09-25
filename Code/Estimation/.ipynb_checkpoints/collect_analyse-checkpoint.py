from shared_pkgs.imports import *
from Estimation.estimators import *

def collect_results_syn(input_dir, dr_flag, num_treats, reps):

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
    input_folders = sorted(os.listdir(input_dir), key=lambda x: int(x))[reps[0]:reps[1]] # OK

    # Retrieve values
    for idx, folder in enumerate(input_folders):
        idx = int(folder)
        
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
                    
                    # Sanity check
                    #if  max(t)!=(num_treats-1):
                    #    print('{} {}'.format(estim,folder))
                    #    continue
                        
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
                        
                        # Sanity check
                        #if max(t)!=(num_treats-1):
                        #    print('{} {}'.format(estim,folder))
                        #    continue

                        # Compute estimator
                        if model=='baseline' and dr_flag:
                            psi = psi_aiptw(q_t0, q_t1, q_t2, q_t3, q_t4, g, t, y, num_treats, truncate_level=0.01)
                        else:
                            psi = psi_naive(q_t0, q_t1, q_t2, q_t3, q_t4, g, truncate_level=0.)

                        result_dict[split][estim][model].append(psi)

                        
                elif estim=='b2bd':
                    for ate in ['ate1_0', 'ate2_0', 'ate3_0', 'ate4_0']:
                        for model in ['baseline', 'targeted_regularization']:
                            # Model level (when applicable)
                            base_dat_path = os.path.join(estim_dat_path, ate, model, ('0_replication_' + split + '.npz'))
                            q_t0, q_t1, g, y, t, index = load_data_dr(base_dat_path)
                            
                            # Sanity check
                            #if max(t)!=(num_treats-1):
                            #    print('{} {}'.format(estim,folder))
                            #    continue
                                
                            # Compute estimator
                            psi = psi_naive_dr(q_t0, q_t1, g,truncate_level=0.)

                            result_dict[split][estim][ate][model].append(psi)

                    # Postprocess: join ates
                    result_dict[split][estim]['baseline'] = list(zip(result_dict[split]['b2bd']['ate1_0']['baseline'], result_dict[split]['b2bd']['ate2_0']['baseline'], result_dict[split]['b2bd']['ate3_0']['baseline'], result_dict[split]['b2bd']['ate4_0']['baseline']))
                    result_dict[split][estim]['targeted_regularization'] = list(zip(result_dict[split]['b2bd']['ate1_0']['targeted_regularization'], result_dict[split]['b2bd']['ate2_0']['targeted_regularization'], result_dict[split]['b2bd']['ate3_0']['targeted_regularization'], result_dict[split]['b2bd']['ate4_0']['targeted_regularization']))

                elif estim=='T_learn':
                    base_dat_path = os.path.join(estim_dat_path, 'baseline', ('0_replication_' + split + '.npz'))
                    q_t0, q_t1, q_t2, q_t3, q_t4, y, t, index = load_data_t(base_dat_path)
                    
                    # Sanity check
                    #if max(t)!=(num_treats-1):
                    #    print('{} {}'.format(estim,folder))
                    #    continue
                        
                    # Compute estimator
                    psi = psi_naive(q_t0, q_t1, q_t2, q_t3, q_t4, g,truncate_level=0.)

                    result_dict[split][estim]['baseline'].append(psi)

                else:
                    sys.exit('wrong estimator list')

    # Postprocess: delete individual ates of dragonnet
    for i in range(1,5):
        del result_dict['train']['b2bd']['ate{}_0'.format(i)]
        del result_dict['test']['b2bd']['ate{}_0'.format(i)]
    

    # Compute averages and CIs
    for estim in estimator:
        # Estimator level (0: b2bd, Hydra, T-learn; 1: b2bd, Hydra, T-learn; ...)

        for split in ['train', 'test']:
            # Split level (0: b2bd: train, test; 0: Hydra: train, test; 0: T-learn: train, test...)

            if estim == 'Hydranet':
                # True and naive estim values
                true_vec = np.asarray(result_dict[split]['true'])
                naive_vec = np.asarray(result_dict[split]['naive']['baseline'])

                # Sanity check
                #if (naive_vec.shape[0] == 0) or (naive_vec.shape[0]!=true_vec.shape[0]):
                #    continue
                    
                naive_ci_l, naive_ci_u = bootstrap((np.nansum(np.abs((true_vec - naive_vec)), axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                result_dict[split]['true'] = np.nanmean(true_vec, axis=0)
                result_dict[split]['naive']['baseline'] = np.nanmean(naive_vec, axis=0)
                result_dict[split]['naive']['baseline_ae'] = np.nanmean(np.nansum(np.abs(true_vec - naive_vec), axis=1))
                result_dict[split]['naive']['baseline_pe'] = np.nansum(np.abs(true_vec - naive_vec))/np.nansum(np.abs(true_vec)) *100
                result_dict[split]['naive']['baseline_ae_ciu'] = naive_ci_u
                result_dict[split]['naive']['baseline_ae_cil'] = naive_ci_l

                for model in ['baseline', 'targeted_regularization']:
                    hydra_vec = np.asarray(result_dict[split][estim][model])
                    #print(hydra_vec)
                    #print('***')
                    # Sanity check
                    #if (hydra_vec.shape[0] == 0) or (hydra_vec.shape[0]!=true_vec.shape[0]):
                    #    print('Shape problem hydra')
                    #    continue
                    
                    Hydra_ci_l, Hydra_ci_u = bootstrap((np.nansum(np.abs(true_vec - hydra_vec),axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                    result_dict[split][estim][model] = np.nanmean(hydra_vec, axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.nanmean(np.nansum(np.abs(true_vec - hydra_vec), axis=1))
                    result_dict[split][estim]['{}_pe'.format(model)] = np.nansum(np.abs(true_vec - hydra_vec))/np.nansum(np.abs(true_vec)) *100
                    result_dict[split][estim]['{}_ae_ciu'.format(model)] = Hydra_ci_u
                    result_dict[split][estim]['{}_ae_cil'.format(model)] = Hydra_ci_l
            
            elif estim == 'b2bd':
                for model in ['baseline', 'targeted_regularization']:
                    b2bd_vec = np.asarray(result_dict[split][estim][model])
                    #print(b2bd_vec)
                    #print('***')
                    # Sanity check
                    #if (b2bd_vec.shape[0] == 0) or (b2bd_vec.shape[0]!=true_vec.shape[0]):
                    #    continue
                    
                    b2bd_ci_l, b2bd_ci_u = bootstrap((np.nansum(np.abs(true_vec - b2bd_vec),axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                    result_dict[split][estim][model] = np.nanmean(b2bd_vec, axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.nanmean(np.nansum(np.abs(true_vec - b2bd_vec), axis=1))
                    result_dict[split][estim]['{}_pe'.format(model)] = np.nansum(np.abs(b2bd_vec - naive_vec))/np.nansum(np.abs(true_vec)) *100
                    result_dict[split][estim]['{}_ae_ciu'.format(model)] = b2bd_ci_u
                    result_dict[split][estim]['{}_ae_cil'.format(model)] = b2bd_ci_l

            elif estim == 'T_learn':
                tlearn_vec = np.asarray(result_dict[split][estim]['baseline'])

                # Sanity check
                #if (tlearn_vec.shape[0] == 0) or (tlearn_vec.shape[0]!=true_vec.shape[0]):
                #    print('Shape problem')
                #    print(tlearn_vec.shape[0])
                #    print(true_vec.shape[0])
                #    print(tlearn_vec)
                #    continue
                
                tlearn_ci_l, tlearn_ci_u = bootstrap((np.nansum(np.abs(true_vec - tlearn_vec),axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                result_dict[split][estim]['baseline'] = np.nanmean(tlearn_vec, axis=0)
                result_dict[split][estim]['baseline_ae'] = np.nanmean(np.nansum(np.abs(true_vec - tlearn_vec), axis=1))
                result_dict[split][estim]['baseline_pe'] = np.nansum(np.abs(tlearn_vec - naive_vec))/np.nansum(np.abs(true_vec)) *100
                result_dict[split][estim]['baseline_ae_ciu'] = tlearn_ci_u
                result_dict[split][estim]['baseline_ae_cil'] = tlearn_ci_l
            
            else:
                sys.exit('wrong estimator list')
            
    return result_dict


def analyse_results_syn(all_res_dict, main_param_dict, main_param, output_dir, dr_flag):
    # Process results dict
    reform = {(outerKey, innerKey): values for outerKey, innerDict in all_res_dict[main_param].items() for innerKey, values in innerDict.items()}
    all_res_df = pd.DataFrame(reform)
    df_train = all_res_df.iloc[:,all_res_df.columns.get_level_values(1)=='train']
    df_test = all_res_df.iloc[:,all_res_df.columns.get_level_values(1)=='test']
    df_train.columns = df_train.columns.droplevel(1)
    df_test.columns = df_test.columns.droplevel(1)
    df_train = df_train.T
    df_test = df_test.T
    os.makedirs(os.path.join(output_dir, main_param), exist_ok=True)

    # Figures
    ###### ABSOLUTE ERROR ######
    fig, ax = plt.subplots()
    line1, = ax.plot(df_train['naive'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_train['naive'].index, df_train['naive'].apply(lambda x: x['baseline_ae_cil']), df_train['naive'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line2, = ax.plot(df_train['b2bd'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_train['naive'].index,df_train['b2bd'].apply(lambda x: x['baseline_ae_cil']), df_train['b2bd'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line3, = ax.plot(df_train['b2bd'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    ax.fill_between(df_train['naive'].index,df_train['b2bd'].apply(lambda x: x['targeted_regularization_ae_cil']), df_train['b2bd'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)
    line4, = ax.plot(df_train['T_learn'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_train['naive'].index,df_train['T_learn'].apply(lambda x: x['baseline_ae_cil']), df_train['T_learn'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line5, = ax.plot(df_train['Hydranet'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_train['naive'].index,df_train['Hydranet'].apply(lambda x: x['baseline_ae_cil']), df_train['Hydranet'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line6, = ax.plot(df_train['Hydranet'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    ax.fill_between(df_train['naive'].index,df_train['Hydranet'].apply(lambda x: x['targeted_regularization_ae_cil']), df_train['Hydranet'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)

    # Handles and labels for the plots
    handles = [line1, line2, line3, line4, line5, line6]
    if dr_flag:
        labels = ['Naive', 'B2BD Baseline', 'B2BD T-reg', 'T-learner', 'Hydranet Baseline (DR)', 'Hydranet T-reg']
    else:
        labels = ['Naive', 'B2BD Baseline', 'B2BD T-reg', 'T-learner', 'Hydranet Baseline', 'Hydranet T-reg']

    plt.legend(handles=handles, labels= labels)
    plt.xlabel(main_param.replace('_',' ').capitalize())
    plt.ylabel('Error')
    fig.savefig(os.path.join(output_dir, main_param + '_ae' + '_in-sample.pdf'))

    fig, ax = plt.subplots()
    line1, = ax.plot(df_test['naive'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_test['naive'].index, df_test['naive'].apply(lambda x: x['baseline_ae_cil']),df_test['naive'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line2, = ax.plot(df_test['b2bd'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_test['naive'].index, df_test['b2bd'].apply(lambda x: x['baseline_ae_cil']),df_test['b2bd'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line3, = ax.plot(df_test['b2bd'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    ax.fill_between(df_test['naive'].index, df_test['b2bd'].apply(lambda x: x['targeted_regularization_ae_cil']),df_test['b2bd'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)
    line4, = ax.plot(df_test['T_learn'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_test['naive'].index, df_test['T_learn'].apply(lambda x: x['baseline_ae_cil']),df_test['T_learn'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line5, = ax.plot(df_test['Hydranet'].apply(lambda x: x['baseline_ae']), marker='o')
    ax.fill_between(df_test['naive'].index, df_test['Hydranet'].apply(lambda x: x['baseline_ae_cil']),df_test['Hydranet'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line6, = ax.plot(df_test['Hydranet'].apply(lambda x: x['targeted_regularization_ae']), marker='o')
    ax.fill_between(df_test['naive'].index, df_test['Hydranet'].apply(lambda x: x['targeted_regularization_ae_cil']),df_test['Hydranet'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)

    plt.legend(handles=handles, labels=labels)
    plt.xlabel(main_param.replace('_',' ').capitalize())
    plt.ylabel('Error')
    fig.savefig(os.path.join(output_dir, main_param + '_ae' + '_out-sample.pdf'))

    ###### RELATIVE ERROR #######
    fig, ax = plt.subplots()
    line1, = ax.plot(df_train['naive'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_train['naive'].index, df_train['naive'].apply(lambda x: x['ae_cil']),df_train['naive'].apply(lambda x: x['ae_ciu']), alpha=.5)
    line2, = ax.plot(df_train['b2bd'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_train['naive'].index, df_train['b2bd'].apply(lambda x: x['baseline_ae_cil']),df_train['b2bd'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line3, = ax.plot(df_train['b2bd'].apply(lambda x: x['targeted_regularization_pe']), marker='o')
    #ax.fill_between(df_train['naive'].index, df_train['b2bd'].apply(lambda x: x['targeted_regularization_ae_cil']),df_train['b2bd'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)
    line4, = ax.plot(df_train['T_learn'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_train['naive'].index, df_train['T_learn'].apply(lambda x: x['baseline_ae_cil']),df_train['T_learn'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line5, = ax.plot(df_train['Hydranet'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_train['naive'].index, df_train['Hydranet'].apply(lambda x: x['baseline_ae_cil']),df_train['Hydranet'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line6, = ax.plot(df_train['Hydranet'].apply(lambda x: x['targeted_regularization_pe']), marker='o')
    #ax.fill_between(df_train['naive'].index, df_train['Hydranet'].apply(lambda x: x['targeted_regularization_ae_cil']),df_train['Hydranet'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)

    plt.legend(handles=handles, labels=labels)
    plt.xlabel(main_param.replace('_',' ').capitalize())
    plt.ylabel('Percentual Error')
    fig.savefig(os.path.join(output_dir, main_param + '_pe' + '_in-sample.pdf'))

    fig, ax = plt.subplots()
    line1, = ax.plot(df_test['naive'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_test['naive'].index, df_test['naive'].apply(lambda x: x['pe_cil']),df_test['naive'].apply(lambda x: x['ae_ciu']), alpha=.5)
    line2, = ax.plot(df_test['b2bd'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_test['naive'].index, df_test['b2bd'].apply(lambda x: x['baseline_ae_cil']),df_test['b2bd'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line3, = ax.plot(df_test['b2bd'].apply(lambda x: x['targeted_regularization_pe']), marker='o')
    #ax.fill_between(df_test['naive'].index, df_test['b2bd'].apply(lambda x: x['targeted_regularization_ae_cil']),df_test['b2bd'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)
    line4, = ax.plot(df_test['T_learn'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_test['naive'].index, df_test['T_learn'].apply(lambda x: x['baseline_ae_cil']),df_test['T_learn'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line5, = ax.plot(df_test['Hydranet'].apply(lambda x: x['baseline_pe']), marker='o')
    #ax.fill_between(df_test['naive'].index, df_test['Hydranet'].apply(lambda x: x['baseline_ae_cil']),df_test['Hydranet'].apply(lambda x: x['baseline_ae_ciu']), alpha=.5)
    line6, = ax.plot(df_test['Hydranet'].apply(lambda x: x['targeted_regularization_pe']), marker='o')
    #ax.fill_between(df_test['naive'].index, df_test['Hydranet'].apply(lambda x: x['targeted_regularization_ae_cil']),df_test['Hydranet'].apply(lambda x: x['targeted_regularization_ae_ciu']), alpha=.5)

    plt.legend(handles=handles, labels=labels)
    plt.xlabel(main_param.replace('_',' ').capitalize())
    plt.ylabel('Percentual Error')
    fig.savefig(os.path.join(output_dir, main_param + '_pe' + '_out-sample.pdf'))


    #### Summary file
    file = os.path.join(output_dir, 'summary.txt')
    with open(file, 'w') as sumfile:
        sumfile.write('In-sample\n')
        sumfile.write('Naive estimator error:\n')
        sumfile.write('{}\n'.format(df_train['naive'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('B2BD baseline error:\n')
        sumfile.write('{}\n'.format(df_train['b2bd'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('B2BD T-reg error:\n')
        sumfile.write('{}\n'.format(df_train['b2bd'].apply(lambda x: x['targeted_regularization_ae']).to_string()))
        sumfile.write('T-learner estimator error:\n')
        sumfile.write('{}\n'.format(df_train['T_learn'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('Hydranet baseline error:\n')
        sumfile.write('{}\n'.format(df_train['Hydranet'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('Hydranet  T-reg error:\n')
        sumfile.write('{}\n'.format(df_train['Hydranet'].apply(lambda x: x['targeted_regularization_ae']).to_string()))
        sumfile.write('*******\n')

        sumfile.write('Out-sample\n')
        sumfile.write('Naive estimator error:\n')
        sumfile.write('{}\n'.format(df_test['naive'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('B2BD baseline error:\n')
        sumfile.write('{}\n'.format(df_test['b2bd'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('B2BD T-reg error:\n')
        sumfile.write('{}\n'.format(df_test['b2bd'].apply(lambda x: x['targeted_regularization_ae']).to_string()))
        sumfile.write('T-learner estimator error:\n')
        sumfile.write('{}\n'.format(df_test['T_learn'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('Hydranet baseline error:\n')
        sumfile.write('{}\n'.format(df_test['Hydranet'].apply(lambda x: x['baseline_ae']).to_string()))
        sumfile.write('Hydranet  T-reg error:\n')
        sumfile.write('{}\n'.format(df_test['Hydranet'].apply(lambda x: x['targeted_regularization_ae']).to_string()))
        sumfile.write('*******\n')

    #### Generate latex table
    generate_result_table_syn(os.path.join(output_dir, 'summ_table.txt'), df_train, df_test, main_param_dict, main_param)


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
    input_folders = sorted(os.listdir(input_dir), key=lambda x: int(x)) # OK

    # Retrieve values
    for idx, folder in enumerate(input_folders):
        idx = int(folder)
        
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
                        #print(truth_dat_path)
                        #print(psi)


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
                    result_dict[split][estim]['baseline'] = list(
                        zip(result_dict[split]['b2bd']['ate1_0']['baseline'],
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
                # True and naive estim values
                true_vec = np.asarray(result_dict[split]['true'])
                naive_vec = np.asarray(result_dict[split]['naive']['baseline'])
                naive_ci_l, naive_ci_u = bootstrap((np.sum(np.abs((true_vec - naive_vec)), axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                result_dict[split]['true'] = np.mean(true_vec, axis=0)
                result_dict[split]['naive']['baseline'] = np.mean(naive_vec, axis=0)
                result_dict[split]['naive']['baseline_ae'] = np.sum(np.abs(true_vec - naive_vec), axis=1).mean()
                result_dict[split]['naive']['baseline_pe'] = np.sum(np.abs(true_vec - naive_vec))/np.sum(np.abs(true_vec)) *100
                result_dict[split]['naive']['baseline_ae_ciu'] = naive_ci_u
                result_dict[split]['naive']['baseline_ae_cil'] = naive_ci_l

                for model in ['baseline', 'targeted_regularization']:
                    hydra_vec = np.asarray(result_dict[split][estim][model])
                    Hydra_ci_l, Hydra_ci_u = bootstrap((np.sum(np.abs(true_vec - hydra_vec),axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                    result_dict[split][estim][model] = np.mean(hydra_vec, axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.sum(np.abs(true_vec - hydra_vec), axis=1).mean()
                    result_dict[split][estim]['{}_pe'.format(model)] = np.sum(np.abs(true_vec - hydra_vec))/np.sum(np.abs(true_vec)) *100
                    result_dict[split][estim]['{}_ae_ciu'.format(model)] = Hydra_ci_u
                    result_dict[split][estim]['{}_ae_cil'.format(model)] = Hydra_ci_l

            elif estim == 'b2bd':
                for model in ['baseline', 'targeted_regularization']:
                    b2bd_vec = np.asarray(result_dict[split][estim][model])
                    b2bd_ci_l, b2bd_ci_u = bootstrap((np.sum(np.abs(true_vec - b2bd_vec),axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                    result_dict[split][estim][model] = np.mean(b2bd_vec, axis=0)
                    result_dict[split][estim]['{}_ae'.format(model)] = np.sum(np.abs(true_vec - b2bd_vec), axis=1).mean()
                    result_dict[split][estim]['{}_pe'.format(model)] = np.sum(np.abs(b2bd_vec - naive_vec))/np.sum(np.abs(true_vec)) *100
                    result_dict[split][estim]['{}_ae_ciu'.format(model)] = b2bd_ci_u
                    result_dict[split][estim]['{}_ae_cil'.format(model)] = b2bd_ci_l

            elif estim == 'T_learn':
                tlearn_vec = np.asarray(result_dict[split][estim]['baseline'])
                tlearn_ci_l, tlearn_ci_u = bootstrap((np.sum(np.abs(true_vec - tlearn_vec),axis=1),), statistic=np.mean, method='basic', random_state=3).confidence_interval

                result_dict[split][estim]['baseline'] = np.mean(tlearn_vec, axis=0)
                result_dict[split][estim]['baseline_ae'] = np.sum(np.abs(true_vec - tlearn_vec), axis=1).mean()
                result_dict[split][estim]['baseline_pe'] = np.sum(np.abs(tlearn_vec - naive_vec))/np.sum(np.abs(true_vec)) *100
                result_dict[split][estim]['baseline_ae_ciu'] = tlearn_ci_u
                result_dict[split][estim]['baseline_ae_cil'] = tlearn_ci_l


    return result_dict


def analyse_results_ihdp(all_res_dict, output_dir):
    # Print figures and generate tables
    all_res_df = pd.DataFrame(all_res_dict)
    df_train = all_res_df['train']
    df_test = all_res_df['test']

    os.makedirs(os.path.join(output_dir), exist_ok=True)
    file = os.path.join(output_dir, 'summary.txt')
    with open(file, 'w') as sumfile:
        sumfile.write('In-sample\n')
        sumfile.write('Naive estimator error: {}\n'.format(df_train['naive']['baseline_ae']))
        sumfile.write('B2BD baseline error: {}\n'.format(df_train['b2bd']['baseline_ae']))
        sumfile.write('B2BD T-reg error: {}\n'.format(df_train['b2bd']['targeted_regularization_ae']))
        sumfile.write('T-learner error: {}\n'.format(df_train['T_learn']['baseline_ae']))
        sumfile.write('Hydranet baseline error: {}\n'.format(df_train['Hydranet']['baseline_ae']))
        sumfile.write('Hydranet T-reg error: {}\n'.format(df_train['Hydranet']['targeted_regularization_ae']))
        sumfile.write('*******\n')

        sumfile.write('Out-sample\n')
        sumfile.write('Naive estimator error: {}\n'.format(df_test['naive']['baseline_ae']))
        sumfile.write('B2BD baseline error: {}\n'.format(df_test['b2bd']['baseline_ae']))
        sumfile.write('B2BD T-reg error: {}\n'.format(df_test['b2bd']['targeted_regularization_ae']))
        sumfile.write('T-learner error: {}\n'.format(df_test['T_learn']['baseline_ae']))
        sumfile.write('Hydranet baseline error: {}\n'.format(df_test['Hydranet']['baseline_ae']))
        sumfile.write('Hydranet T-reg error: {}\n'.format(df_test['Hydranet']['targeted_regularization_ae']))
        sumfile.write('*******\n')

    # Generate result table
    generate_result_table_ihdp(os.path.join(output_dir,'summ_table.txt'), df_train, df_test, err_type='ae')