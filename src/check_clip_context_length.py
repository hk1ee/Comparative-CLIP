import json
import clip
import torch
import argparse

parser = argparse.ArgumentParser(description='Filter JSON elements for CLIP context length.')
parser.add_argument('--dataset', type=str, required=True, help='Dataset.')
parser.add_argument('--concept', type=str, required=True, help='High-level concept.')
opt = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
context_length = model.context_length - 3

def is_within_context_length(class_name, element, context_length):
    """Check if the text is within the context length allowed by the CLIP model."""
    text = f"A photo of a {class_name}, which has {element}." if opt.concept.strip() else f"A photo of a {opt.concept}: a {class_name}, which has {element}."
    try:
        tokens = clip.tokenize([text], context_length=context_length)
        return True
    except RuntimeError as e:
        if "too long for context length" in str(e):
            return False
        else:
            raise e

def filter_dict(input_dict, context_length):
    """Filter dictionary elements based on context length."""
    return {class_name: [element for element in elements if is_within_context_length(class_name, element, context_length)]
            for class_name, elements in input_dict.items()}

def main():
    input_file_path = f"../descriptors/comparative_descriptors/descriptors_{opt.dataset}.json"
    output_file_path = input_file_path

    with open(input_file_path, 'r') as file:
        data = json.load(file)

    filtered_data = filter_dict(data, context_length)

    with open(output_file_path, 'w') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)

    print(f"Filtered data has been saved to {output_file_path}")

if __name__ == "__main__":
    main()
