source ../params.sh

savename="equal_number"

for model_size in "${backbones[@]}";
do
    # comparative
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        n=${n_list[$i]}
        
        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=comparative --n=${n} --reps=5 --model_size=${model_size}

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=comparative --n=${n} --reps=5 --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done
done