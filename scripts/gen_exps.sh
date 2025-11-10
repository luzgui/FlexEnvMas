#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $SCRIPT_DIR
names="exp_names_data_files_test.csv"
base_test_env="test011"

while IFS=, read -r new_name data_file; do
  echo "Base test environment is $base_test_env"
  echo "exp_name: $new_name"
  echo "data file: $data_file"
  sleep 2

  cd "$SCRIPT_DIR"/../configs || exit

  sudo cp -r $base_test_env "$new_name"

  if [ -d "$new_name" ]; then
    echo "generated config folder"
    chmod -R 777 "$new_name"
  else
    echo "error in generating config foder"
  fi

  cd "$new_name" || exit
  echo "we are at"
  pwd
  sleep 2
  chmod -R 777
  echo "Updating 'exp_name' key to: $new_name"
  # yq eval ".exp_name = \"$new_name\"" -i "experiment_config.yaml"
  yq -i -y ".exp_name = \"$new_name\"" "experiment_config.yaml"


  echo "Updating 'datase_file' key to: $data_file"
  # yq eval ".dataset_file = \"$data_file\"" -i "problem_config.yaml"
  yq -i -y ".dataset_file = \"$data_file\"" "problem_config.yaml"
done < "$names"

#
# while IFS= read -r new_name; do
#   echo "Processing $new_name"
#   echo "Updating 'exp_name' key to: $new_name"
#   #yq eval ".exp_name = \"$new_name\"" -i "$exp_conf"
#   yq -i -y ".exp_name = \"$new_name\"" "$exp_conf"
#
#   # batcat $exp_name
#
#   sleep 2
#
#   echo "Updating 'test_name' key to: $new_name"
#   #yq eval ".test_name = \"$new_name\"" -i "$test_con"
#   yq -i -y ".test_name = \"$new_name\"" "$test_conf"
#   # batcat $test_config
#   sleep 2
#
#
# done < "$names"
#
# # base_test_env="test011"
# echo "Base test environment is $base_test_env"
# sleep 5
#
#
# # Path to your YAML file
# # SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # echo $SCRIPT_DIR
#
# # exp_conf="$SCRIPT_DIR/../configs/exp_name.yaml"
# # test_conf="$SCRIPT_DIR/../configs/test_config.yaml"
# # echo $yaml_file
#
# # test="$SCRIPT_DIR/../testings/test.py"
# # test_fixed="$SCRIPT_DIR/../testings/test_fixed.py"
#
# #Iterate over each name and update the YAML key 'name' for each one
# for new_name in "${names[@]}"; do
#   echo "Updating 'exp_name' key to: $new_name"
#   #yq eval ".exp_name = \"$new_name\"" -i "$exp_conf"
#   yq -i -y ".exp_name = \"$new_name\"" "$exp_conf"
#
#   # batcat $exp_name
#
#   sleep 2
#
#   echo "Updating 'test_name' key to: $new_name"
#   #yq eval ".test_name = \"$new_name\"" -i "$test_con"
#   yq -i -y ".test_name = \"$new_name\"" "$test_conf"
#   # batcat $test_config
#   sleep 2
#
#   #ray210
#   cd $SCRIPT_DIR/../testings || exit
#
#   sleep 2
#   python "$test"
#
# done
#
#
# # echo "Performing tests with the optimal actions"
# # echo "Updating 'test_name' and 'exp_name'  key to: $base_test_env"
# # yq -i -y ".test_name = \"$base_test_env\"" "$test_conf"
# # yq -i -y ".exp_name = \"$base_test_env\"" "$exp_conf"
# #
# #
# # for new_name in "${names[@]}"; do
# #   actions_source="optimal_${new_name}"
# #   echo "actions from : '$actions_source'"
# #   sleep 2
# #
# #   echo "Updating 'actions_from_folder' key to: $actions_source"
# #   yq -i -y ".actions_from_folder = \"$actions_source\"" "$test_conf"
# #
# #   sleep 2
# #
# #   cd $SCRIPT_DIR/../testings || exit
# #
# #   sleep 2
# #   python "$test_fixed"
# #
# # done
