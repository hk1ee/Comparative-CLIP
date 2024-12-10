import json
import pickle
import random
import argparse
import tools

parser = argparse.ArgumentParser(description='Generate descriptors using GPT.')
parser.add_argument('--mode', type=str, required=True, choices=['comparative', 'dclip'], help='Choose the operation mode.')
parser.add_argument('--dataset', type=str, required=True, choices=tools.DATASETS, help='Dataset to generate descriptors.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')
opt = parser.parse_args()

def get_class_names(dataset):
    # HACK
    if dataset == 'imagenetv2':
        class_names = tools.openai_imagenet_classes
    else:
        with open(f'../descriptors/dclip_descriptors/descriptors_{dataset}.json', 'rb') as file:
            data = json.load(file)
        class_names = list(data.keys())

    return class_names

def generate_comparative_prompt(target_category, similar_category):
    qa_template = """Q: What are useful features for distinguishing a {target} from a {similar} in the photo? Give the list of features, each beginning with a hyphen (-).
A: There are several useful visual features to tell the photo is a {target}, not a {similar}.\n{features}"""
    
    examples = [
        ("soccer stadium", "baseball stadium", ["goal posts", "soccer ball", "shorts uniform", "rectangular field", "corner flags"]),
        ("harbor", "beach", ["docks", "breakwaters", "fishing boats", "cargo ships", "cranes"]),
        ("river", "lake", ["movement of water", "ripples", "wave", "bank", "delta"]),
        ("apple pie", "waffle", ["a golden-brown, flaky crust", "apple pieces", "spiced apple mixture", "round and deep shape", "light to deep brown color"]),
        ("anthurium", "hibiscus", ["shiny, heart-shaped spathe", "spadix", "arrow-shaped leaves", "glossy or leather-like leaf texture", "green or white colored spathes"]),
        ("abyssinian", "russian blue", ["ticked coat pattern", "warm reddish or golden fur", "large, alert ears", "slender body shape", "vibrant green or gold eyes"]),
        ("black-footed albatross", "frigatebird", ["a large, thick bill", "larger, with a dark plumage", "long, narrow wings", "webbed feet", "white underwings"]),
        ("beer glass", "coffee mug", ["beer", "carbonation bubbles", "clear glass", "rim shape", "logos"]),
        ("Ford model T", "golf cart", ["metal body panels", "doors", "convertible roof", "larger, rubber tires", "Ford logo"]),
        ("electric guitar", "acoustic guitar", ["solid body", "magnetic pickups", "control knobs", "input jack", "amplifier"])
    ]
    
    candidates = [qa_template.format(target=target, similar=similar, features='\n'.join(f'- {feature}' for feature in features)) for target, similar, features in examples]
    
    randomly_selected = random.sample(candidates, 2)
    custom_prompt = qa_template.format(target=target_category, similar=similar_category, features="")
    
    prompt = '\n\n'.join(randomly_selected + [custom_prompt])
    
    return prompt

def generate_dclip_prompt(target_category):
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo? Give the list of features, each beginning with a hyphen (-).
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet
Q: What are useful visual features for distinguishing a television in a photo? Give the list of features, each beginning with a hyphen (-).
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control
Q: What are useful features for distinguishing a {target_category} in a photo? Give the list of features, each beginning with a hyphen (-).
A: There are several useful visual features to tell there is a {target_category} in a photo:
-
"""

def main():
    tools.seed_everything(opt.seed)

    class_names = get_class_names(opt.dataset)

    batch_requests = []

    if opt.mode == 'comparative':
        with open(f'../similar_classes/{opt.dataset}.pkl', 'rb') as file:
            similar_dict = pickle.load(file)

        for class_name in class_names:
            similar_categories = similar_dict[class_name]

            for similar_category in similar_categories:
                prompt = generate_comparative_prompt(class_name, similar_category)
                
                request = {
                    "custom_id": f"{opt.dataset}~{class_name}~{similar_category}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1000
                    }
                }
                batch_requests.append(request)

        output_filename = f'../batch_API/batch_{opt.dataset}.jsonl'
    
    elif opt.mode == 'dclip':
        for class_name in class_names:
            prompt = generate_dclip_prompt(class_name)
            
            request = {
                "custom_id": f"{opt.dataset}~{class_name}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000
                }
            }
            batch_requests.append(request)

        output_filename = f'../batch_API/batch_dclip_{opt.dataset}.jsonl'

    with open(output_filename, 'w') as file:
        for request in batch_requests:
            file.write(json.dumps(request) + '\n')

    print(f"Saved batch requests to {output_filename}")

if __name__ == "__main__":
    main()
