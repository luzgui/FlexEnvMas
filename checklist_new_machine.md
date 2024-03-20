# checklist for running new experiments in different machines

## Training

* After defining the number of agents:
  * In "models2.py" define the number of agents in order to define the size of the CC NN
  * in "data_process.py" select the loads and clusters

* In "trainable.py" define 'n_iters' and 'checkpoint_freq'
* In "main_train.py"
  * define the experiment name, 'exp_name'
  * define 'resources'

* In 'experiment_build.py'
  * check 'num_sgd_iter' and 'train_batch_size'

After checking this points please run imports again

## Testing

* make sure the 'exp_name' match the experiment folder name

* Check the number of agents in the trials/checkpoints:
  * In "models2.py" define the number of agents in order to define the size of the CC NN

  * In 'test_build.py'
    * define the testing laods and clusters
    * define 'best_config['env_config']['init_condition']'

*
