datasets=("imagenet" "imagenetv2" "caltech256" "cifar100" "cub" "eurosat" "places365" "food101" "pets" "dtd" "flowers102" "fgvcaircraft" "cars")
concepts=("" "" "" "" "bird" "satellite photo" "place" "food" "pet" "texture" "flower" "aircraft" "car")
n_list=(10 10 10 10 10 9 10 10 10 10 10 10 10)
k_list=(10 10 10 20 5 10 15 20 20 5 5 15 15)
best_shot_list=(64 32 32 32 16 32 64 64 16 32 8 8 16)
ViT_L_14_shot_list=(64 64 32 8 16 1 64 32 8 8 4 32 16)
RN50_shot_list=(64 32 32 16 16 2 64 32 32 8 4 2 16)
max_shot_list=(64 64 32 64 16 64 64 64 64 32 8 32 16)
backbones=("ViT-B/32" "ViT-L/14" "RN50")
seed_list=(1 2 3 4 5)