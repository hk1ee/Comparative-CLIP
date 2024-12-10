source ../params.sh

for model_size in "ViT-B/32" "ViT-L/14" "RN50"
do
    python ../../src/extract_image_features.py --model_size=${model_size}
done