import numpy as np
import pandas as pd
import os
import sys

num_treats = 5
num_covars = 30
num_reps = 20
dataset = 'synthetic'
output_parent_dir = '/home/bvelasco/Hydranet/Input_data'
np.random.seed(1)


def treatment_assignment_op(x, num_treats, p_yes=0.8, p_no=0.2):
    t_vals = [i for i in range(num_treats)]
    x_new=x.copy()
    for j in t_vals:
        probs = np.tile(p_no / (float(len(t_vals) - 1)), len(t_vals))
        probs[j] = p_yes
        x_new[x == j] = np.random.choice(t_vals, size=sum(x == j), replace=True, p=probs)
    return x_new

def plot_histograms(data, z, y_toplot, bins, path):
    ax = data[y_toplot].hist(bins=bins)
    data[y_toplot][data.z == 1].hist(bins=bins)
    ax.legend(['Y{}'.format(z), 'Y[T={}]'.format(z)]);
    fig = ax.get_figure()
    fig.savefig(os.path.join(path,'y{}_y_{}.pdf'.format(z,z)))
    ax.clear()

def analyse_generated_data(temp_all, path):

    file = os.path.join(path,'summary.txt')
    with open(file, 'w') as sumfile:
        sumfile.write('Percentage of T==0: {}\n'.format(sum(temp_all['z'] == 0) / len(temp_all)))
        sumfile.write('Percentage of T==1: {}\n'.format(sum(temp_all['z'] == 1) / len(temp_all)))
        sumfile.write('Percentage of T==2: {}\n'.format(sum(temp_all['z'] == 2) / len(temp_all)))
        sumfile.write('Percentage of T==3: {}\n'.format(sum(temp_all['z'] == 3) / len(temp_all)))
        sumfile.write('Percentage of T==4: {}\n'.format(sum(temp_all['z'] == 4) / len(temp_all)))
        sumfile.write('Max T {}\n'.format(temp_all['z'].max()))
        sumfile.write('*******\n')

        ### Mean values of the true effects

        sumfile.write('True average y_0: {}\n'.format(temp_all['y_0'].mean()))
        sumfile.write('True average y_1: {}\n'.format(temp_all['y_1'].mean()))
        sumfile.write('True average y_2: {}\n'.format(temp_all['y_2'].mean()))
        sumfile.write('True average y_3: {}\n'.format(temp_all['y_3'].mean()))
        sumfile.write('True average y_4: {}\n'.format(temp_all['y_4'].mean()))
        sumfile.write('*******\n')

        ### Mean values of the observable, biased effects

        sumfile.write('Biased average y0: {}\n'.format(temp_all.y[temp_all.z == 0].mean()))
        sumfile.write('Biased average y1: {}\n'.format(temp_all.y[temp_all.z == 1].mean()))
        sumfile.write('Biased average y2: {}\n'.format(temp_all.y[temp_all.z == 2].mean()))
        sumfile.write('Biased average y3: {}\n'.format(temp_all.y[temp_all.z == 3].mean()))
        sumfile.write('Biased average y4: {}\n'.format(temp_all.y[temp_all.z == 4].mean()))
        sumfile.write('*******\n')

        ### Individual and total bias
        
        sumfile.write('results_bias 0: {}\n'.format(abs((temp_all['y_0'].mean()) - (temp_all.y[temp_all.z == 0].mean()))))
        sumfile.write('results_bias 0: {}\n'.format(abs((temp_all['y_1'].mean()) - (temp_all.y[temp_all.z == 1].mean()))))
        sumfile.write('results_bias 2: {}\n'.format(abs((temp_all['y_2'].mean()) - (temp_all.y[temp_all.z == 2].mean()))))
        sumfile.write('results_bias 3: {}\n'.format(abs((temp_all['y_3'].mean()) - (temp_all.y[temp_all.z == 3].mean()))))
        sumfile.write('results_bias 4: {}\n'.format(abs((temp_all['y_4'].mean()) - (temp_all.y[temp_all.z == 4].mean()))))

        b0 = abs((temp_all.y[temp_all.z == 0].mean()) - (temp_all.y_0).mean()) / abs(temp_all.y_0).mean() * 100
        b1 = abs((temp_all.y[temp_all.z == 1].mean()) - (temp_all.y_1).mean()) / abs(temp_all.y_1).mean() * 100
        b2 = abs((temp_all.y[temp_all.z == 2].mean()) - (temp_all.y_2).mean()) / abs(temp_all.y_2).mean() * 100
        b3 = abs((temp_all.y[temp_all.z == 3].mean()) - (temp_all.y_3).mean()) / abs(temp_all.y_3).mean() * 100
        b4 = abs((temp_all.y[temp_all.z == 4].mean()) - (temp_all.y_4).mean()) / abs(temp_all.y_4).mean() * 100
        sumfile.write('*******\n')

        sumfile.write('results_bias 0 perc: {}\n'.format(b0))
        sumfile.write('results_bias 1 perc: {}\n'.format(b1))
        sumfile.write('results_bias 2 perc: {}\n'.format(b2))
        sumfile.write('results_bias 3 perc: {}\n'.format(b3))
        sumfile.write('results_bias 3 perc: {}\n'.format(b4))
        sumfile.write('total: {}\n'.format(b0 + b1 + b2 + b3 + b4))
        sumfile.write('*******\n')

    ### Individual histograms

    for z in range(num_treats):
        plot_histograms(data=temp_all, z=z, y_toplot='y_{}'.format(z), bins=100, path=path)


if dataset=='ihdp':
    bias_size=10

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
        f_0 = lambda x,t: np.exp(np.matmul((x + w), beta)) + bias_size * x[:,26] + t**2  # additive term for displacing the mean of the normal distribution based on T
        f_1 = lambda x,t: np.log(np.abs(np.matmul(x, beta)) + 1e-10) + bias_size * x[:, 25] * t
        f_2 = lambda x,t: np.matmul(x, beta) + bias_size * x[:, 27] + t**2
        f_3 = lambda x,t: np.exp(np.matmul((x + w), beta) + t) + bias_size * x[:, 2]
        f_4 = lambda x,t: np.log(np.abs(np.matmul((x + w), beta)) + 1e-10) + bias_size * x[:, 2] * t
        y_f = lambda x: np.random.normal(x, sigma)

        # Compute true effect
        covars_tab['mu_0'] = f_0(covars.to_numpy(),0)
        covars_tab['mu_1'] = f_1(covars.to_numpy(),1)
        covars_tab['mu_2'] = f_2(covars.to_numpy(),2)
        covars_tab['mu_3'] = f_3(covars.to_numpy(),3)
        covars_tab['mu_4'] = f_4(covars.to_numpy(),4)

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

        filename = os.path.join(output_dir, 'ihdp_{}.csv'.format(i))
        os.makedirs(output_dir, exist_ok=True)
        covars_tab.to_csv(filename)

        # Accumulate dataset to compute statistics
        temp_all = pd.concat([temp_all, covars_tab], axis=0)

    analyse_generated_data(temp_all,output_dir)
    print('Done ihdp')

elif dataset=='synthetic':
    var_dict = {'bias':{'bias':[2, 5, 10, 30], 'positivity': 80, 'data_size':2000, 'n_confs':2},
                'positivity': {'bias':20, 'positivity': [60, 70, 80, 95], 'data_size':2000, 'n_confs':2},
                'n_confs':{'bias':20, 'positivity': 80,'data_size':2000, 'n_confs':[2, 5, 10, 18]},
                'data_size':{'bias':20, 'positivity': 80,'data_size':[1000, 2000, 5000, 10000], 'n_confs':2}
                }

    for main_param in ['bias','n_confs','data_size', 'positivity']:

        bias_vals = var_dict[main_param]['bias']
        n_confs_vals = var_dict[main_param]['n_confs']
        data_size_vals = var_dict[main_param]['data_size']
        pos_vals = var_dict[main_param]['positivity']

        for i in range(len(var_dict[main_param][main_param])):

            output_dir = os.path.join(output_parent_dir, dataset, '{}_treats/'.format(num_treats), main_param, str(var_dict[main_param][main_param][i]))
            print(output_dir)

            if main_param=='bias':
                bias=bias_vals[i]
                pos = pos_vals
                data_size=data_size_vals
                n_confs=n_confs_vals
            elif main_param=='positivity':
                bias = bias_vals
                pos = pos_vals[i]
                data_size=data_size_vals
                n_confs = n_confs_vals
            elif main_param=='n_confs':
                bias = bias_vals
                pos = pos_vals
                data_size = data_size_vals
                n_confs=n_confs_vals[i]
            elif main_param=='data_size':
                bias = bias_vals
                pos = pos_vals
                data_size=data_size_vals[i]
                n_confs = n_confs_vals
            
            # Transform "positivity" to 0-1 scale
            pos = pos/100.

            columns = ['x{}'.format(i) for i in range(num_covars)]+['mu_0', 'mu_1','mu_2', 'mu_3', 'mu_4', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y', 'z']
            temp_all = pd.DataFrame(columns=columns)

            for j in range(num_reps):
                # Parameters
                sigma=1

                # Covariates
                X = np.random.uniform(-1,1,size=(data_size, num_covars))
                X = pd.DataFrame(X, columns=['x{}'.format(i) for i in range(num_covars)])

                # Treatment
                sel_covar_names = ['x{}'.format(i) for i in range(n_confs)]
                covars = X[sel_covar_names]
                z_ini = 10*covars**2
                z_ini = z_ini.sum(axis=1)
                
                z = np.round((num_treats-1) * ((z_ini - z_ini.min()) / (z_ini.max() - z_ini.min())))
                      
                #z = np.round((num_treats+1) * ((z_ini - z_ini.min()) / (z_ini.max() - z_ini.min())))
                #z[z>=num_treats] = num_treats-1
                      
                z_f = treatment_assignment_op(z, num_treats, pos, 1-pos)

                # Output
                # Baseline effect
                u0 = np.random.uniform(size=(1, num_covars))
                v0 = u0 / np.linalg.norm(u0)

                # T effect
                ut = np.random.uniform(size=(1, num_covars))
                vt = ut / np.linalg.norm(ut)

                f0 = lambda x, t: 30*np.matmul(v0,x.T) + 10*t**2*np.matmul(vt,x.T)
                f1 = lambda x, t: 20*np.matmul(v0,x.T) + 5*t*bias*np.matmul(vt,x.T)
                f2 = lambda x, t: 10*np.matmul(v0,x.T) + 5*np.log(np.abs(bias*t*np.matmul(vt,x.T)))

                # True effect
                mu_0 = f0(X.to_numpy(),0).reshape(-1, 1)
                mu_1 = f1(X.to_numpy(),1).reshape(-1, 1)
                mu_2 = f2(X.to_numpy(),2).reshape(-1, 1)
                mu_3 = f1(X.to_numpy(),3).reshape(-1, 1)
                mu_4 = f0(X.to_numpy(),4).reshape(-1, 1)


                # Sample from normal distribution
                y_f = lambda x: np.random.normal(x, sigma)
                y_0 = y_f(mu_0).reshape(-1,1)
                y_1 = y_f(mu_1).reshape(-1,1)
                y_2 = y_f(mu_2).reshape(-1,1)
                y_3 = y_f(mu_3).reshape(-1,1)
                y_4 = y_f(mu_4).reshape(-1,1)

                y = y_0.copy()
                y[z == 0] = y_0[z == 0]
                y[z == 1] = y_1[z == 1]
                y[z == 2] = y_2[z == 2]
                y[z == 3] = y_3[z == 3]
                y[z == 4] = y_4[z == 4]

                z = z.values.reshape(-1,1)
                covars_tab = np.concatenate([X,mu_0, mu_1, mu_2, mu_3, mu_4, y_0, y_1, y_2, y_3, y_4, y, z], axis=1)
                covars_tab = pd.DataFrame(covars_tab, columns=columns)

                filename = os.path.join(output_dir, 'ihdp_{}.csv'.format(j))
                os.makedirs(output_dir, exist_ok=True)
                covars_tab.to_csv(filename)

                # Accumulate dataset to compute statistics
                temp_all = pd.concat([temp_all, covars_tab], axis=0)

            analyse_generated_data(temp_all, output_dir)

    print('Done synthetic')

else:
    sys.exit('Wrong dataset')

