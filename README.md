# Hydranet
Repository of Hydranet, a neural network for the estimation of multivalued treatment effects.

## Requirements
The required packages can be found in requirements.txt

## Data
In order to generate data for running experiments, run Input_data/data_generator.py. Change ```output_parent_dir``` as required, and set ```dataset``` to 'ihdp' or 'synthetic'.

## Experiments
Run the following command to use Hydranet and replicate experiments and results

``
python Code/main.py --input_dir='<path_to_Hydranet>' --output_dir='<path_to_Hydranet/Results>' --Train=True --Analyze=True --dataset=<'ihdp' or 'synthetic'> --main_param=<'bias', 'n_confs' or 'data_size'> --device=<'CPU' or 'GPU'>
`` 
