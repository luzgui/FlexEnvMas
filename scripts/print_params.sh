#!/usr/bin/env bash
#This script is intended to be be used inside RLlib trial folders that contain the checkpoints and
# all the trial files.
# It opens the params.json file and prints the parameters defined in this script
# It is usage is to check what are the best HP and nay config for a scpecific trial when perfomrming HPO

# List of keys you care about (adjust as you like)
KEYS=(
  "clip_param"
  "vf_clip_param"
  "vf_loss_coeff"
  "lr"
  "train_batch_size"
  "num_sgd_iter"
  "sgd_minibatch_size"
  "entropy_coeff"
  "lambda_"
  "seed"
  "kl_coeff"


)

# 1) Check if params.json exists in the *current* directory
if [[ ! -f "params.json" ]]; then
  echo "No params.json in $(pwd)"
  exit 1
fi

# 3) For each key, print "key: value" if it exists (null is also printed)
for key in "${KEYS[@]}"; do
  value=$(jq -r --arg k "$key" '.[$k] // "KEY_NOT_FOUND"' params.json)
  echo "$key: $value"
done
