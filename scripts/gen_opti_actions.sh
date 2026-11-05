#!/bin/bash




# Path to your YAML file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
echo "Project root: $PROJECT_ROOT"
source "$SCRIPT_DIR/global_config.sh" #global variables
# echo $SCRIPT_DIR

echo "Base test environment is $base_test_env"
sleep 5

exp_conf="$SCRIPT_DIR/../configs/exp_name.yaml"

test="$SCRIPT_DIR/../testings/test.py" # for generating optimal sols
test_fixed="$SCRIPT_DIR/../testings/test_fixed.py" #for generating solutions with optimal actions

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# exp_name='NA'
#
# yq eval ".exp_name = \"$base_test_env\"" -i "$exp_conf"
# yq eval ".train_env = \"$base_test_env\"" -i "$exp_conf"
# yq eval ".eval_env = \"$base_test_env\"" -i "$exp_conf"

while IFS=, read -r env_name dataset_files; do

  yq eval '.shouldOpti = True' -i "$exp_conf"
  yq eval '.shouldTest = False' -i "$exp_conf"

  echo "Updating 'test_name' key to: $env_name"
  yq eval ".test_env = \"$env_name\"" -i "$exp_conf"
  sleep 2

  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate ray210
  cd $SCRIPT_DIR/../testings || exit

  pwd
  sleep 2
  python "$test"

done < "$names_file"


# echo "Performing tests with the optimal actions"
# echo "Updating 'test_name' and 'exp_name'  key to: $base_test_env"
# yq -i -y ".test_name = \"$base_test_env\"" "$exp_conf"
# yq -i -y ".exp_name = \"$base_test_env\"" "$exp_conf"
#
#
# while IFS=, read -r new_name; do
#   actions_source="optimal_${new_name}"
#   echo "actions from : '$actions_source'"
#   sleep 2
#
#   echo "Updating 'actions_from_folder' key to: $actions_source"
#   yq -i -y ".actions_from_folder = \"$actions_source\"" "$exp_conf"
#
#   sleep 2
#
#   cd $SCRIPT_DIR/../testings || exit
#
#   sleep 2
#   python "$test_fixed"
#
# done < "#$names"
