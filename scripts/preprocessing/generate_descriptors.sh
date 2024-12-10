source ../params.sh

for model_size in "${backbones[@]}";
do
    for ((i=0; i<${#datasets[@]}; i++));
    do
        dataset=${datasets[$i]}
        
        python ../../src/generate_batch_jsonl.py --dataset=${dataset} --model_size=${model_size}
        python ../../src/jsonl_to_descriptor.py --dataset=${dataset} --model_size=${model_size}
    done
done