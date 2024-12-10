source ../params.sh

savename="baselines"

for model_size in "${backbones[@]}";
do
    # CLIP
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        
        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=clip --model_size=${model_size}

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=clip --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # DCLIP
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        
        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=dclip --model_size=${model_size}

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=dclip --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # WaffleCLIP
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        n=${n_list[$i]}
        
        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=waffle --n=${n} --reps=5 --model_size=${model_size}

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=waffle --n=${n} --reps=5 --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done
done