from tqdm import tqdm
import os
import json
import tools
import random
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Randomly retain a specified number of descriptors.')
parser.add_argument('--dataset', type=str, required=True, choices=tools.DATASETS, help='Dataset to filter descriptors.')
parser.add_argument('--k', type=int, default=20, help='Number of descriptors to randomly save.')
parser.add_argument('--loaddir', type=str, default='comparative_descriptors', help='Path to load descriptors.')
parser.add_argument('--savedir', type=str, default='filtered_descriptors', help='Path to save filtered descriptors.')
parser.add_argument('--seed', type=int, default=0, help='Seed for random selection.')
opt = parser.parse_args()

def main():
    tools.seed_everything(opt.seed)

    with open(f'../descriptors/{opt.loaddir}/descriptors_{opt.dataset}.json', 'r') as file:
        descriptor_dict = json.load(file)
    
    results = {}
    for class_name in tqdm(descriptor_dict.keys(), desc=f"Filtering random {opt.k} descriptors in {opt.dataset}..."):
        descriptors = list(set(descriptor_dict[class_name]))
        descriptors.sort()  # Ensure consistent order
        top_k_descriptors = random.sample(descriptors, min(len(descriptors), opt.k))
        results[class_name] = top_k_descriptors
    
    filtered_descriptor_path = os.path.join('../descriptors', opt.savedir, f'descriptors_{opt.dataset}' + (f'_{opt.seed}' if opt.seed != 0 else '') + '.json')
    os.makedirs(os.path.dirname(filtered_descriptor_path), exist_ok=True)

    with open(filtered_descriptor_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
    