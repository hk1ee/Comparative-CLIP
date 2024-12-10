from tqdm import tqdm
import os
import json
import torch
import clip
import numpy as np
import tools
import pickle
import random
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Filtering to retain only useful descriptors.')
parser.add_argument('--dataset', type=str, required=True, choices=tools.DATASETS, help='Dataset to filter descriptors.')
parser.add_argument('--shot', type=int, default=8, help='Number of images.')
parser.add_argument('--k', type=int, default=20, help='Number of descriptors to save after filtering (in order of similarity scores).')
parser.add_argument('--loaddir', type=str, default='comparative_descriptors', help='Path to load descriptors.')
parser.add_argument('--savedir', type=str, default='filtered_descriptors', help='Path to save filtered descriptors.')
parser.add_argument('--model_size', type=str, default='ViT-B/32', choices=tools.BACKBONES, help='Pretrained CLIP model to use.')
parser.add_argument('--seed', type=int, default=0, help='seed')
opt = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(opt.model_size, device=device)

def calculate_cosine_similarity(text_features, image_features):
    text_features = text_features.float()
    image_features = image_features.float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)
    return torch.mm(text_features, image_features.T)

def calculate_mean_features(feature_dict, n):
    mean_features = {}
    for class_name, features in feature_dict.items():
        if len(features) < n:
            raise ValueError(f"The number of images in class {class_name} is less than {n}.")
        selected_indices = random.sample(range(len(features)), n)
        selected_features = [features[idx] for idx in selected_indices]
        selected_features_array = np.array(selected_features)        
        selected_features_tensor = torch.tensor(selected_features_array, dtype=torch.float32, device=device)
        mean_feature = torch.mean(selected_features_tensor, axis=0)
        mean_features[class_name] = mean_feature
    return mean_features

def get_threshold_similarity(class_name, image_feature):
    sentence = f"A photo of a {class_name}."
    text_tokens = clip.tokenize([sentence]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    similarity = calculate_cosine_similarity(text_features, image_feature.unsqueeze(0)).squeeze().item()
    return similarity

def main():
    tools.seed_everything(opt.seed)
    
    model_size = opt.model_size.replace('/', '-')
    
    with open(f'../descriptors/{opt.loaddir}/descriptors_{opt.dataset}.json', 'r') as file:
        descriptor_dict = json.load(file)
    
    with open(f'../precomputed_image_features/{model_size}/{opt.dataset}.pkl', 'rb') as file:
        feature_dict = pickle.load(file)
        mean_features = calculate_mean_features(feature_dict, opt.shot)
    
    results = {}

    for class_name in tqdm(descriptor_dict.keys(), desc=f"Filtering top {opt.k} descriptors in {opt.dataset}..."):
        descriptors = list(set(descriptor_dict[class_name]))
        mean_feature = mean_features[class_name]
        similarity_threshold = min(get_threshold_similarity(class_name, mean_feature), 0.3)
        
        scores = []
        for descriptor in descriptors:
            sentences = [f"A photo of a {class_name}, {tools.make_descriptor_sentence(descriptor)}."]
            text_tokens = clip.tokenize(sentences).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
            # mean_feature = mean_feature.unsqueeze(0)
            similarity_score = calculate_cosine_similarity(text_features, mean_feature).squeeze().item()
            if similarity_score > similarity_threshold:
                scores.append((descriptor, similarity_score))
        
        sorted_descriptors = sorted(scores, key=lambda x: x[1], reverse=True)
        top_k_descriptors = [desc[0] for desc in sorted_descriptors[:min(len(sorted_descriptors), opt.k)]]
        results[class_name] = top_k_descriptors
    
    filtered_descriptor_path = os.path.join('../descriptors', opt.savedir, model_size, f'descriptors_{opt.dataset}' + (f'_{opt.seed}' if opt.seed != 0 else '') + '.json')

    os.makedirs(os.path.dirname(filtered_descriptor_path), exist_ok=True)

    with open(filtered_descriptor_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()