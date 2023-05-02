import numpy as np
import pandas as pd
import os

num_treats = 5 # or 10
num_covars = 30
bias = 5
num_reps = 2
dataset = 'synthetic' # or 'synthetic'
output_parent_dir = '/home/bvelasco/Hydranet/Input_data'
np.random.seed(1)

if dataset=='ihdp':
    bias_size=10
    n_confs=2
    data_size=None

    # Output dir
    output_dir = os.path.join(output_parent_dir, dataset, '{}_treats/'.format(num_treats))

    # Read data
    covars = pd.read_csv(os.path.join(output_parent_dir,'ihdp.csv'))

    # Drop unneeded
    covars.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Fix 'first' var
    covars['first'] = covars['first'] - 1

    # Continuous vars
    conts = ['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage']
    # Binary vars
    binar = [var for var in list(covars.columns.values) if var not in conts]

    # Normalize continuous vars
    for col in conts:
        covars[col] = (covars[col] - covars[col].mean()) / covars[col].std()

    # Paramters
    t_vals = np.array([0, 1, 2, 3, 4])
    beta_vals = np.arange(0.0, 0.5, 0.1)  # beta values
    beta_probs = np.array([0.7, 0.15, 0.1, 0.03, 0.02])  # beta probs
    w = np.ones((covars.shape[0], covars.shape[1] - 1)) * 0.5  # offset matrix
    sigma = 1  # std

    temp_all = pd.DataFrame(columns=['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage', 'sex',
                                     'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll', 'cig', 'first',
                                     'booze', 'drugs', 'work.dur', 'prenatal', 'ark', 'ein', 'har', 'mia',
                                     'pen', 'tex', 'was', 'momwhite', 'momblack', 'momhisp', 'mu_0', 'mu_1',
                                     'mu_2', 'mu_3', 'mu_4', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y', 'z'])

    for i in range(num_reps):
        # Treatment assignment (probabilistic based on covariate, non-random)
        t_momblack = np.random.choice(t_vals, covars[covars.momblack == 1].shape[0], replace=True,p=np.array([0.8, 0.05, 0.05, 0.05, 0.05]))
        t_momwhite = np.random.choice(t_vals, covars[covars.momwhite == 1].shape[0], replace=True,p=np.array([0.05, 0.8, 0.05, 0.05, 0.05]))
        t_momhisp = np.random.choice(t_vals, covars[covars.momhisp == 1].shape[0], replace=True,p=np.array([0.05, 0.05, 0.8, 0.05, 0.05]))
        t_preterm_big = np.random.choice(t_vals, covars[covars.preterm > 8].shape[0], replace=True,p=np.array([0.05, 0.05, 0.05, 0.8, 0.05]))
        t_preterm_small = np.random.choice(t_vals, covars[(covars.momblack == 1) & (covars.preterm < 8)].shape[0],replace=True, p=np.array([0.05, 0.05, 0.05, 0.05, 0.8]))

        covars.loc[(covars['momblack'] == 1), 'treat'] = t_momblack
        covars.loc[(covars['momwhite'] == 1), 'treat'] = t_momwhite
        covars.loc[(covars['momhisp'] == 1), 'treat'] = t_momhisp
        covars.loc[((covars['preterm'] > 6) & (covars['momhisp'] == 1)), 'treat'] = t_preterm_big
        covars.loc[((covars['preterm'] < 6) & (covars['momblack'] == 1)), 'treat'] = t_preterm_small

        covars.reset_index(inplace=True, drop=True)
        z = covars['treat']
        covars.drop(['treat'], axis=1, inplace=True)
        covars_tab = covars.copy()

        # Parameters
        beta = np.random.choice(beta_vals, covars.shape[1], replace=True, p=beta_probs)  # multiplying vector
        w = np.ones((covars.shape)) * 0.5  # offset matrix

        # Response surface functions (function and sampling from distribution)
        f_0 = lambda x: np.exp(np.matmul((x + w), beta)) + bias_size * x[:,26]  # additive term for displacing the mean of the normal distribution based on T
        f_1 = lambda x: np.log(np.abs(np.matmul(x, beta)) + 1e-10) + bias_size * x[:, 25]
        f_2 = lambda x: np.matmul(x, beta) + bias_size * x[:, 27]
        f_3 = lambda x: np.exp(np.matmul((x + w), beta)) + bias_size * x[:, 2]
        f_4 = lambda x: np.log(np.abs(np.matmul((x + w), beta)) + 1e-10) + bias_size * x[:, 2]
        y_f = lambda x: np.random.normal(x, sigma)

        # Compute true effect
        covars_tab['mu_0'] = f_0(covars.to_numpy())
        covars_tab['mu_1'] = f_1(covars.to_numpy())
        covars_tab['mu_2'] = f_2(covars.to_numpy())
        covars_tab['mu_3'] = f_3(covars.to_numpy())
        covars_tab['mu_4'] = f_4(covars.to_numpy())

        # Sample from normal distribution
        covars_tab['y_0'] = y_f(covars_tab.mu_0)
        covars_tab['y_1'] = y_f(covars_tab.mu_1)
        covars_tab['y_2'] = y_f(covars_tab.mu_2)
        covars_tab['y_3'] = y_f(covars_tab.mu_3)
        covars_tab['y_4'] = y_f(covars_tab.mu_4)

        y = covars_tab.y_0.copy()
        y[z == 0] = covars_tab.y_0[z == 0].values
        y[z == 1] = covars_tab.y_1[z == 1].values
        y[z == 2] = covars_tab.y_2[z == 2].values
        y[z == 3] = covars_tab.y_3[z == 3].values
        y[z == 4] = covars_tab.y_4[z == 4].values

        covars_tab['y'] = y
        covars_tab['z'] = z.astype(int)

        temp_all = pd.concat([temp_all, covars_tab], axis=0)

        filename = os.path.join(output_dir, 'ihdp_{}.csv'.format(i))
        covars_tab.to_csv(filename)

    print('Done ihdp')

elif dataset=='synthetic':
    var_dict = {'bias':{'bias':[1,5,10,30], 'data_size':[2000], 'n_confs':2},
                'n_confs':{'bias':10, 'data_size':[2000], 'n_confs':[2, 5, 10, 18]},
                'data_size':{'bias':10, 'data_size':[1000, 2000, 5000, 10000], 'n_confs':2}
                }

    for main_param in ['bias', 'data_size', 'n_confs']:
        #print(main_param)
        #print(var_dict[main_param]['bias'])
        #print(var_dict[main_param]['n_confs'])
        #print(var_dict[main_param]['data_size'])

        bias = var_dict[main_param]['bias']
        n_confs = var_dict[main_param]['n_confs']
        data_size = var_dict[main_param]['data_size']

        for i in var_dict[main_param][main_param]:
            output_dir = os.path.join(output_parent_dir, dataset, '{}_treats/'.format(num_treats), main_param, str(i))
            print(output_dir)

            for j in range(num_reps):
                # Generate covariates

                columns = ['x{}'.format(i) for i in range(30)]+['mu_0', 'mu_1','mu_2', 'mu_3', 'mu_4', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y', 'z']
                temp_all = pd.DataFrame(columns=columns)

                for data_size_val in data_size:
                    X = np.random.rand(data_size_val, num_covars)
                    z = np.random.randint(low=0, high=num_treats, size=(data_size_val,1))
                    y = np.random.randn(data_size_val,1)
                    y0 = np.random.randn(data_size_val,1)
                    y1 = np.random.randn(data_size_val,1)
                    y2 = np.random.randn(data_size_val,1)
                    y3 = np.random.randn(data_size_val,1)
                    y4 = np.random.randn(data_size_val,1)
                    mu_0 = np.random.randn(data_size_val,1)
                    mu_1 = np.random.randn(data_size_val,1)
                    mu_2 = np.random.randn(data_size_val,1)
                    mu_3 = np.random.randn(data_size_val,1)
                    mu_4 = np.random.randn(data_size_val,1)

                    data = np.concatenate([X,mu_0, mu_1, mu_2, mu_3, mu_4, y0, y1, y2, y3, y4, y, z], axis=1)
                    data = pd.DataFrame(data, columns=columns)

                    filename = os.path.join(output_dir, 'ihdp_{}.csv'.format(j))
                    os.makedirs(output_dir, exist_ok=True)
                    data.to_csv(filename)


    print('Done synthetic')

else:
    sys.exit('Wrong dataset')