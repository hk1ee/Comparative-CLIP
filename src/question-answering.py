from tqdm import tqdm
import json
import torch
import pickle
import argparse
import transformers

parser = argparse.ArgumentParser(description='Question Answering.')
parser.add_argument('--dataset', type=str, required=True, help='Dataset to do question answering.')
parser.add_argument('--type', type=str, required=True, choices=['comparative', 'dclip']) # comparative, dclip
parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use.')

opt = parser.parse_args()

def initialize():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device, # device_map="auto",
    ) 
    return pipeline

def query(pipeline, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]['content']

def generate_question(target_class, similar_classes, descriptors):
    descriptors = ', '.join(descriptors)
    class_lists = ', '.join(sorted([target_class] + similar_classes))

    prompt = f"""You are responsible for determining which class given descriptions are closest to.
You will be given a list of classes, and descriptions that describes one of them.
Your task is to determine which class from the list is best represented by the descriptions provided.
The goal of this task is to validate the effectiveness of the descriptions in accurately representing the correct class, particularly in distinguishing between similar classes.
Please carefully analyze the descriptions and choose the one class from the list that best matches them.

Descriptions: {descriptors}
Class lists: {class_lists}

This is a hard problem. Your answer should be the class name that best matches the descriptions.
Please ignore any explanation of your answer and provide ONLY one class name as the answer.
"""
    return prompt

def main():
    with open(f'../descriptors/{opt.type}_descriptors/descriptors_{opt.dataset}.json', 'rb') as file:
        descriptor_dict = json.load(file)
    
    with open(f'../similar_classes/{opt.dataset}_3.pkl', 'rb') as file:
        similar_dict = pickle.load(file)
    
    llama_pipeline = initialize()
    
    results = dict()
    count = 0
    
    for i, class_name in tqdm(enumerate(descriptor_dict.keys())):
        prompt = generate_question(class_name, similar_dict[class_name], descriptor_dict[class_name])
        class_name = class_name.lower() # for fair eval
        answer = query(llama_pipeline, prompt).strip().lower()
        if class_name == answer: count += 1
        print(f'Target Class: {class_name}, Answer: {answer}, Count: {count}/{i+1}')
        results[class_name] = answer
    
    with open(f'../results/qa_{opt.type}_{opt.dataset}.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    
    with open('../results/qa_results.json', 'a', encoding='utf-8') as qa_results_file:
        json.dump({
            "dataset": opt.dataset, 
            "type": opt.type, 
            "correct_count": f"{count}/{len(descriptor_dict.keys())}"
        }, qa_results_file)
        qa_results_file.write('\n')

if __name__ == "__main__":
    main()