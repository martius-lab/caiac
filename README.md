# CAIAC (Causal Action Influence Aware Counterfactual Data Augmentation)


Code for the paper "[Causal Action Influence Aware Counterfactual Data Augmentation](https://arxiv.org/pdf/2405.18917)", accepted to ICML 2024.

## Installation

CAIAC can be installed by unzipping the code file, accessing the folder and doing the following (I recommend creating a separate virtual environment for it).

```
cd causal_skill_learning
virtualenv venv
pip install -e .
```


## Training 
CAIAC requires 2 steps: training a world model from a dataset required for CAI computation and training the downstream algorithm leveraging the counterfactual data augmentation.

### Traning world model

#### Downloading the dataset
To train the world model you need to download the provided datasets (in the folder `data` of the zipped file ) and put it inside the `causal_slr` folder in the CAIAC codebase.

Activate the environment:

```
source venv/bin/activate
```

Run the code:

```
python3 main.py  'path_to_name_world_model_yaml_file'
```

where  `path_to_name_world_model_yaml_file` is a YAML file containing the parameters for the world model training.

For Fetch-Push experiments:
* `causal_slr/model_training/configs/fetch/fetchpush.yaml`

For Fetch-Pick&Lift experiments:
* `causal_slr/model_training/configs/fpp/fpp.yaml`

For Franka-Kitchen experiments:
* `causal_slr/model_training/configs/kitchen/config.yaml`

For the FrankaKitchen experiments the path to the training dataset can be set in the YAML file using `path_data` variable:

* `kitchen/all_datasets_and_random.npy` (for the all_tasks experiment)
* `kitchen/mw_counterfactual_kettle.npy` (for the toy example)

For the Fetch-Push and Fetch-Pick&Lift experiments the data is automatically loaded but one can modify the ratios of expert data (`expert_data`), random data (`random_data`) and ratio of the original dataset size (`shrink_dataset`) by modifying the corresponding parameters in the YAML file.

The trained world models will be saved in:
`experiments/model_learning/{sweep_name}/timestamp`
where {sweep_name} parameter can be set in the corresponding YAML file.

### Counterfactual data creation and training downstream algorithms

Run the code:

```
python3 main.py  'path_to_alg_yaml_file'
```

where 'path_to_alg_yaml_file' is a YAML file containing the parameters for the downstream learning algorithm and counterfactuals samples creation.

For Fetch-Push experiments:
* `causal_slr/rl_training/configs/rl_push.yaml`

For Fetch-Pick&Lift experiments:
* `causal_slr/rl_training/configs/fpp.yaml`

For Franka-Kitchen experiments:
* `causal_slr/skill_training/configs/mw_kettle/config.yaml` (for toy experiment)
* `causal_slr/skill_training/configs/kitchen/config.yaml` (for all tasks experiment)

In the corresponding YAML file one can specify which influence detection `scorer_cls` method to be used:
* `cai` is the default, as used for CAIAC
* `coda` for the CoDA baseline (more below)
* `mask` for the CoDA-action ablation baseline (more below)
* one can set `prob_counterfactual` or `ratio_cf` param to 0.0, to implement the No-Augmentation baseline.

In the YAML file one needs to provide the {SAVED_MODEL_PATH} to the trained world model by setting the variable: `preset_mdp_config_path` in the corresponding YAML file.

For the kitchen experiments the path to the training dataset can be set in the YAML file under {dataset_name} (although it should be the same one as the one used for training the world model)
    - kitchen/all_datasets_and_random.npy (for the all_tasks experiment)
    - kitchen/mw_counterfactual_kettle.npy (for the toy example)
Additionally, the user can decide which tasks to evaluate the agent on by setting the `tasks` parameter (e.g.: `tasks: ['microwave', 'kettle']`) and to whether randomly initialize the objects at evaluation time using the `random_init_obs` variable (we set it to True for all our experiments.)


For the Fetch-Push examples the data is automatically loaded but one can modify the ratios of expert data (`expert_data`), random data (`random_data`) and ratio of the original dataset size (`shr_data`) by modifying the corresponding parameters in the YAML file (although we used the same ratios as the ones used for training the world model)

## Visualize trained agents:
```
python3 main_eval.py 'path_to_conf_yaml'
```

where 'path_to_conf_yaml' is the path the configuration created in the experiments results folder, ex: experiments/skill_learning/caiac_kitchen/conf.yaml


## Minor code adaptation (to be modified)
Currently, to be able to modify the required tasks in the Franka-Kitchen environment, our adapted Franka-Kitchen implementation (inheriting from D4RL env) requires a slight modification in the source code of D4RL.
For reproducibility, hence, please add the following function in the *class KitchenV0* defined in `venv/lib/python3.8/site-packages/d4rl/kitchen/adept_envs/franka/kitchen_multitask_v0.py`:

```
def set_tasks(self, tasks):
    self.TASK_ELEMENTS = tasks
```

## Data collection for Fetch-Push and Fetch-Pick&Lift

The data for the Fetch-Push and Fetch-Pick&Lift experiments were obtained by training an online agent via DDPG on the environment `DisentangledFetchPush-2B-v0`. The scripts for doing so are in the `data_collection` folder.
(Requirements mpi4py)
For training run the code:
```
mpirun -np 8 python3 data_collection/train_online_agent.py  --env-name DisentangledFetchPush-2B-v0
```

The trained models will be saved in `data_collection/saved_models/{env-name}/ (unless the save_dir variable is modified in the args)

To collect data with the trained agent:
```
python3 data_collection/collect_data.py --env-name DisentangledFetchPush-2B-v0
```

in case you want to run a random policy add the `--random` as an argument.

## Visualizing trained agents

To visualize the trained models:

```
python3 data_collection/run_trained_agent.py --env-name {env_name} --model-path {path_to_model}
```




## Baselines
We implemented several baselines, namely CoDA, its ablation CoDA-action and RSC.

### Training world model

For CoDA, CoDA-action and RSC, first you need to train the world model (we used a transformer) and the parameters in the following YAML files.
For Fetch-Push experiments:
* `causal_slr/model_training/configs/fetch/transformer.yaml` (CoDA and CoDA-action)

For Fetch-Pick&Lift experiments:
* `causal_slr/model_training/configs/fpp/dyna_transformer.yaml` (CoDA and CoDA-action)

For Franka-Kitchen experiments:
* `causal_slr/model_training/configs/kitchen/config_transformer.yaml` (CoDA and CoDA-action)

### Counterfactual data creation and training downstream algorithms

For Fetch-Push experiments:
* `causal_slr/rl_training/configs/rl_push.yaml` (CoDA and CoDA-action) changing `scorer_cls` to `coda` or `mask` respectively. 

For Fetch-Pick&Lift experiments:
* `causal_slr/rl_training/configs/fpp.yaml` (CoDA and CoDA-action) changing `scorer_cls` to `coda` or `mask` respectively. 


For Franka-Kitchen experiments:
* `causal_slr/skill_training/configs/mw_kettle/config.yaml` (for toy experiment) (CoDA and CoDA-action) changing `scorer_cls` to `coda` or `mask` respectively. 
* `causal_slr/skill_training/configs/kitchen/config.yaml` (for all tasks experiment) (CoDA and CoDA-action) changing `scorer_cls` to `coda` or `mask` respectively.


# Citation
If you use our work or some of our ideas, please consider citing us :)

```
@article{urpi2024causal,
  title={Causal Action Influence Aware Counterfactual Data Augmentation},
  author={Urp{\'\i}, N{\'u}ria Armengol and Bagatella, Marco and Vlastelica, Marin and Martius, Georg},
  journal={arXiv preprint arXiv:2405.18917},
  year={2024}
}
```