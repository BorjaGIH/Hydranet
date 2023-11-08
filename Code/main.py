import os
import sys
from shared_pkgs.imports import *
from Estimation.collect_analyse import *
from Training.run_train_pred import *

def main():

    # Comment for new branch creation (development): commit 2

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_treats", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="synthetic", choices=['synthetic', 'ihdp'])
    parser.add_argument("--input_dir", type=str, default="/home/bvelasco/Hydranet/")
    parser.add_argument("--output_dir", type=str, default="/home/bvelasco/Hydranet/Results/Stable/")
    parser.add_argument("--main_param", type=str, choices=["data_size", 'n_confs', 'bias', 'positivity'])
    parser.add_argument("--main_param_size", type=int, default=None)
    parser.add_argument("--device", type=str, default='GPU', choices=["GPU", "CPU"])
    parser.add_argument('--loss', type=eval, default=hydranet_loss)
    parser.add_argument('--loss_dr', type=eval, default=dragonnet_loss_binarycross_dr)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--Train", type=eval, default=False, choices= [True, False])
    parser.add_argument("--Analyze", type=eval, default=False , choices=[True, False])
    parser.add_argument("--DR_flag", type=eval, default=False, choices=[True, False])
    parser.add_argument("--reps_start", type=int, default=0)
    parser.add_argument("--reps_end", type=int, default=20)
    parser.add_argument("--trainmode", type=str, default='sequential', choices=["parallel", "sequential"])
    parser.add_argument("--eager_exec", type=bool, default=True)
    parser.add_argument("--meta_learner", type=str, default='T', choices=['T','X'])

    args = parser.parse_args()
    num_treats = args.num_treats
    dataset = args.dataset
    main_param = args.main_param
    main_param_size = args.main_param_size
    device = args.device
    input_dir = args.input_dir
    output_dir = args.output_dir
    loss = args.loss
    loss_dr = args.loss_dr
    val_split = args.val_split
    batch_size = args.batch_size
    Train = args.Train
    Analyze = args.Analyze
    dr_flag = args.DR_flag
    reps = [args.reps_start, args.reps_end]
    trainmode = args.trainmode
    eager_exec = args.eager_exec
    meta_learn = args.meta_learner

    # Result dicts
    main_param_dict = {'bias':[2,5,10,30],
                       'positivity':[60, 70, 80, 90, 95, 98],
                       'n_confs':[2, 5, 10, 18],
                       'data_size':[1000, 2000, 5000, 8000]
                      }
    
    all_res_dict = {'bias': {2:[], 5:[], 10:[], 30:[]},
                    'positivity': {60:[], 70:[], 80:[], 90:[], 95:[], 98:[]},
                    'n_confs': {2:[], 5:[], 10:[], 18:[]},
                    'data_size': {1000:[], 2000:[], 5000:[], 8000:[]} 
                   }
    
    # Avoid TF being too verbose
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Eager execution
    if eager_exec:
        tf.compat.v1.enable_eager_execution()
    else:
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

        base_input_dir = os.path.join(input_dir, 'Input_data/')
        base_output_dir = os.path.join(output_dir, 'Results_NN/')
        

        with tf.device(device):
            # Set seeds
            tf.compat.v1.set_random_seed(1)
            if dataset=='synthetic':
                
                # Iterate along main param values or use only one value
                if main_param_size==None:
                    main_param_iterator = main_param_dict[main_param]
                else:
                    main_param_iterator = [main_param_dict[main_param][main_param_dict[main_param].index(main_param_size)]]
                    
                if trainmode=='sequential':
                    for val in main_param_iterator:
                        # Build paths
                        input_dir_ = base_input_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param) + '/{}/'.format(val)
                        output_dir_ = base_output_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param) + '/{}/'.format(val)
                        run_train(input_dir=input_dir_, output_dir=output_dir_, dataset=dataset, num_treats=num_treats, loss=loss, loss_dr=loss_dr, val_split=val_split, batch_size=batch_size, reps=reps, eager_exec=eager_exec, meta_learn=meta_learn)
                elif trainmode=='parallel':
                    '''(Parallel(n_jobs=20)(delayed(run_train)(input_dir=input_dir_, output_dir=output_dir_, dataset=dataset, num_treats=num_treats, loss=loss, loss_dr=loss_dr, val_split=val_split, batch_size=batch_size, reps=reps)
                   for val in main_param_dict[main_param]))'''
                    sys.exit('Parallel mode not implemented')
                else:
                    sys.exit('Wrong trainmode')
            elif dataset=='ihdp':
                # Build paths
                input_dir_ = base_input_dir + dataset + '/{}_treats/'.format(num_treats)
                output_dir_ = base_output_dir + dataset + '/{}_treats/'.format(num_treats)
                run_train(input_dir=input_dir_, output_dir=output_dir_, dataset=dataset, num_treats=num_treats, loss=loss, loss_dr=loss_dr, val_split=val_split, batch_size=batch_size, reps=reps, eager_exec=eager_exec, meta_learn=meta_learn)
    else:
        print('Do not train')
            
    # Analyze
    if Analyze:
        print('Analyze')        
        base_output_dir = os.path.join(output_dir, 'Results_NN/')
        base_input_dir = base_output_dir
        base_output_dir = os.path.join(output_dir, 'Results_CI/')

        if dataset == 'synthetic':
            
            # Iterate along main param values or use only one value
            if main_param_size==None:
                main_param_iterator = main_param_dict[main_param]
            else:
                main_param_iterator = [main_param_dict[main_param][main_param_dict[main_param].index(main_param_size)]]
                    
            for val in main_param_iterator:
                # Build paths
                input_dir_ = base_input_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param) + '/{}/'.format(val)
                all_res_dict[main_param][val] = collect_results_syn(input_dir_, dr_flag, num_treats, reps=reps)
            output_dir_ = base_output_dir + dataset + '/{}_treats/'.format(num_treats) + str(main_param)
            analyse_results_syn(all_res_dict, main_param_dict, main_param, output_dir_, dr_flag)

        elif dataset == 'ihdp':
            # Build paths
            input_dir_ = base_input_dir + dataset + '/{}_treats/'.format(num_treats)
            output_dir_ = base_output_dir + dataset + '/{}_treats/'.format(num_treats)
            res_dict = collect_results_ihdp(input_dir_)
            analyse_results_ihdp(res_dict, output_dir_)

    else:
        print('Do not analyze')

    print('Done')
    
if __name__ == '__main__':
    main()
