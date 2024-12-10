source ../params.sh

savename="using_only_descriptor"

for model_size in "${backbones[@]}";
do
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        concept=${concepts[$i]}
        
        python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size} --use_only_descriptor=True

        if [[ -n $concept ]]; then
            python ../../src/base_main.py --savename=${savename} --dataset=${dataset} --mode=comparative --model_size=${model_size} --use_only_descriptor=True --label_before_text="A photo of a ${concept}: a "
        fi
    done
done