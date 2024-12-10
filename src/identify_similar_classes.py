import json
import torch
import pickle
import random
import argparse
import numpy as np
from transformers import CLIPTokenizer, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import tools
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._utils')

parser = argparse.ArgumentParser(description='Identify similar classes.')
parser.add_argument('--dataset', type=str, required=True, choices=tools.DATASETS, help='Dataset to find similar classes.')
parser.add_argument('--n', type=int, help='Number of the similar classes.')
parser.add_argument('--mode', type=str, default='top', choices=['top','random'], help='Choose the operation mode.')
parser.add_argument('--savedir', type=str, default='similar_classes')

opt = parser.parse_args()

tools.seed_everything(1)

def get_class_names(dataset):
    # HACK
    if dataset == 'imagenetv2':
        class_names = tools.openai_imagenet_classes
    else:
        with open(f'../descriptors/dclip_descriptors/descriptors_{dataset}.json', 'rb') as file:
            data = json.load(file)
        class_names = list(data.keys())

    return class_names

def compute_text_features(class_names, batch_size=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    
    text_features = []
    for i in range(0, len(class_names), batch_size):
        batch_class_names = class_names[i:i+batch_size]
        text_inputs = tokenizer(batch_class_names, padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            batch_text_features = model.get_text_features(**text_inputs)
        batch_text_features = batch_text_features.detach().cpu().numpy()
        text_features.append(batch_text_features)

    text_features = np.concatenate(text_features, axis=0)

    return text_features

def find_most_similar_classes(class_names, sim_matrix, n):
    top_n_similar_classes = {}
    for i, class_name in enumerate(class_names):
        similarities = sim_matrix[i]
        similarities[i] = -1
        top_n_indices = np.argsort(similarities)[-n:]
        top_n_classes = [class_names[j] for j in top_n_indices]
        top_n_similar_classes[class_name] = top_n_classes

    return top_n_similar_classes

def find_random_classes(class_names, n):
    random_dict = {}
    
    for class_name in class_names:
        selected = random.sample(class_names, min(n + 1, len(class_names)))
        if class_name in selected:
            selected.remove(class_name)
        else:
            selected = selected[:n]
        random_dict[class_name] = selected
        
    return random_dict

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
  
    print(f"Similar classes extracted and saved to {file_path}")

def main():
    class_names = get_class_names(opt.dataset)

    text_features = compute_text_features(class_names)
    text_sim_matrix = cosine_similarity(text_features)
    
    if opt.mode == 'top':
        sim_classes = find_most_similar_classes(class_names, text_sim_matrix, n=opt.n)
    elif opt.mode == 'random':
        sim_classes = find_random_classes(class_names, n=opt.n)

    save_data(sim_classes, f'../{opt.savedir}/{opt.dataset}.pkl')

if __name__ == "__main__":
    main()