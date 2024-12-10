source ../params.sh

savename="ours"

for model_size in "${backbones[@]}";
do
    # comparative
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        
        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size}

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done

    # comparative + filtering
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        k=${k_list[$i]}
        
        case $model_size in
        "ViT-B/32")
            shot="${best_shot_list[$i]}"
            ;;
        "ViT-L/14")
            shot="${ViT_L_14_shot_list[$i]}"
            ;;
        "RN50")
            shot="${RN50_shot_list[$i]}"
            ;;
        *)
            echo "model_size error: $model_size"
            shot=""
            ;;
        esac

        # for seed in "${seed_list[@]}"
        # do
        #     python ../../src/filter_descriptors.py --dataset=${dataset} --k=${k} --shot=${shot} --seed=${seed} --loaddir=comparative_descriptors --savedir=filtered_descriptors --model_size=${model_size}
        # done

        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered --reps=5 --model_size=${model_size}

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --k=${k} --shot=${shot} --mode=filtered --reps=5 --model_size=${model_size} --label_before_text="A photo of a ${concept}: a "
        fi
    done
done