# checklist for running new experiments in different machines

* After defining the number of agents:
  * In "models2.py" define the number of agents in order to define the size of the CC NN
  * in "data_process.py" select the loads and clusters

* In "trainable.py" define 'n_iters' and 'checkpoint_freq'
* In "main_train.py"
  * define the experiment name, 'exp_name'
  * define 'resources'

After checking this points please run imports again
