source ../params.sh

savename="filter_existing_method"

for model_size in "${backbones[@]}";
do
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        k=${k_list[$i]}
        shot=${best_shot_list[$i]}
        
        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered_dclip --reps=5 --model_size=${model_size}

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered_dclip --reps=5 --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done
done