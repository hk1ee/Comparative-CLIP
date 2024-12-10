import json
import tools
import argparse

parser = argparse.ArgumentParser(description='Process image and text features.')
parser.add_argument('--mode', type=str, required=True, choices=['comparative', 'dclip'], help='Choose the operation mode.')
parser.add_argument('--dataset', type=str, required=True, choices=tools.DATASETS, help='Dataset to find similar classes.')
parser.add_argument('--batch_id', type=str, required=True, help='GPT Batch ID.')

opt = parser.parse_args()

def generate_descriptor(content):
    attributes = [desc[2:] for desc in content.split('\n') if desc.startswith('- ')]
    return [attribute for attribute in attributes]

def process_jsonl(input_file, output_file):
    class_dict = {}

    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            custom_id = data['custom_id']
            content = data['response']['body']['choices'][0]['message']['content']
            
            # Extract class_name
            if opt.mode == 'comparative':
                _, class_name, _ = custom_id.split('~')
            elif opt.mode == 'dclip':
                _, class_name = custom_id.split('~')
            
            # Generate descriptor list
            attributes = generate_descriptor(content)
            
            # Update dictionary
            if class_name in class_dict:
                class_dict[class_name].extend(attributes)
            else:
                class_dict[class_name] = attributes
    
    # Save the dictionary to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(class_dict, outfile, indent=4)
    
    print(f"Saved descriptors: {output_file}")

input_file = f'../batch_API/{opt.batch_id}_output.jsonl'
output_file = f'../descriptors/{opt.mode}_descriptors/descriptors_{opt.dataset}.json'
# output_file = f'../descriptors/random_class_descriptors/descriptors_{opt.dataset}.json'
process_jsonl(input_file, output_file)