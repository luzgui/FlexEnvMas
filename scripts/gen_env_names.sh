# #!/bin/bash
# ####
# # Generates a csv file with the names of the environments and the corresponding
# dataset

# epsilon=(20 10 8 5 4 3 2 1 08 06 04 02 01 008 006 004 002)
epsilon=(500 200 100 50 10 5 1)

# Range for number_2
start_num2=0
end_num2=0

# The DP environments are transformations of base_test_env
# base_test_env="tenv2002"
mech_to_use= "optstream"
sensitivity="app"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/global_config.sh"

csv_folder="$SCRIPT_DIR/../Data"

# output_file=$names_file
> "$names_file"  # Clear the file before starting

for i in "${epsilon[@]}"; do
    for ((j=start_num2; j<=end_num2; j++)); do

        search_string="eps${i}_${j}"
        filename="${base_test_env}$search_string"
        matches=""

        # Search CSV filenames for E${i}_${j}
        while IFS= read -r -d '' csv_file; do
            csv_name="${csv_file##*/}"

            if [[ "$csv_name" == *"$search_string"* &&
                  "$csv_name" == *"$mech_to_use"* &&
                  "$csv_name" == *"$sensitivity"* ]]; then
                if [[ -n "$matches" ]]; then
                    matches+=";"
                fi
                matches+="$csv_name"
            fi

        done < <(find "$csv_folder" -maxdepth 1 -type f -name "*.csv" -print0)

        echo "Generating: $filename"
        printf '%s,%s\n' "$filename" "$matches" >> "$SCRIPT_DIR/$names_file"
    done
echo $matches
done

echo "created $names_file"
count=$(wc -l < "$names_file")
echo "$count envs names created"
