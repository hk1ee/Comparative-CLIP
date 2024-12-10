#%%
import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import clip
import numpy as np
import pickle
from termcolor import colored
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import tqdm

import tools

#%%
parser = argparse.ArgumentParser()
### Base arguments.
parser.add_argument('--mode', type=str, default='clip', choices=tools.METHODS,
                    help='VLM extension to use.')
parser.add_argument('--seed', type=int, default=1, 
                    help='Replication seed.')
parser.add_argument('--batch_size', type=int, default=640, 
                    help='Batchsize, mainly used to compute image embeddings.')
parser.add_argument('--dataset', type=str, default='imagenetv2', choices=tools.DATASETS, 
                    help='Dataset to evaluate on.')
parser.add_argument('--model_size', type=str, default='ViT-B/32', choices=tools.BACKBONES, 
                    help='Pretrained CLIP model to use.')
parser.add_argument('--aggregate', type=str, default='mean', choices=('mean', 'max'), 
                    help='How to aggregate similarites of multiple language embeddings.')
### Text going before and after class names & descriptors.
### In the default case, this would be "A photo of a "<classname> ... <descriptors>"."
parser.add_argument('--label_before_text', type=str, default='A photo of a ', 
                    help='Prompt-part going at the very beginning.')
parser.add_argument('--label_after_text', type=str, default='.', 
                    help='Prompt-part going at the very end.')
###
parser.add_argument('--pre_descriptor_text', type=str, default='', 
                    help='Text that goes right before the descriptor.')
parser.add_argument('--descriptor_separator', type=str, default=', ', 
                    help='Text separating descriptor part and classname.')
###
parser.add_argument('--dont_apply_descriptor_modification', action='store_true',
                    help='Flag. If set, will not use "which (is/has/etc)" before descriptors.')
parser.add_argument('--merge_predictions', action='store_true', 
                    help='Optional flag to merge generated embeddings before computing retrieval scores.')
parser.add_argument('--save_model', type=str, default='', 
                    help='Set to a non-empty filename to store generated language embeddings & scores in a pickle file for all seed-repetitions.')
parser.add_argument('--randomization_budget', type=int, default=15,
                    help='Budget w.r.t. to DCLIP for randomization ablations')
parser.add_argument('--waffle_count', type=int, default=15,
                    help='For WaffleCLIP: Number of randomized descriptor pairs to use')
parser.add_argument('--reps', type=int, default=1, 
                    help='Number of repetitions to run a method for with changing randomization. Default value should be >7 for WaffleCLIP variants.')
parser.add_argument('--savename', type=str, default='result',
                    help='Name of csv-file in which results are stored.')
parser.add_argument('--shot', type=int, default=0,
                    help='[0, 1, 2, 4, 8, 16, 32, 64]')
parser.add_argument('--k', type=int, default=0,
                    help='5, 10, 15, 20')
###
parser.add_argument('--vmf_scale', type=float, default=1)
opt = parser.parse_args()
opt.apply_descriptor_modification = not opt.dont_apply_descriptor_modification

#%% Get dataloader and load model.
tools.seed_everything(opt.seed)
opt, dataset = tools.setup(opt)

print(colored(f"\nLoading model [{opt.model_size}] for dataset [{opt.dataset}] ...\n", "yellow", attrs=["bold"]))

opt.device = device = torch.device('cuda')
model, preprocess = clip.load(opt.model_size, device=device, jit=False)
model.eval()
model.requires_grad_(False)

#%% Compute image embeddings if not already precomputed.
precomputed_encs_folder = '../precomputed_encs'
os.makedirs(precomputed_encs_folder, exist_ok=True)
precomputed_encs_file = os.path.join(
    precomputed_encs_folder, 
    f'{opt.dataset}_{opt.model_size.lower().replace("/", "")}.pkl'
)
    
if os.path.exists(precomputed_encs_file):
    load_res = pickle.load(open(precomputed_encs_file, 'rb'))
else:
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)    
    
    enc_coll = []
    label_coll = []
    with torch.no_grad():
        for batch_number, batch in enumerate(tqdm.tqdm(dataloader, desc='Precomputing image embeddings...')):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            label_coll.append(labels)
            
            image_encodings = F.normalize(model.encode_image(images))
            enc_coll.append(image_encodings.cpu())
    load_res = {'enc': enc_coll, 'labels': label_coll}
    pickle.dump(load_res, open(precomputed_encs_file, 'wb'))
    
encoding_coll = load_res['enc']
label_coll = load_res['labels']

#%% Generate Image Embeddings and compute scores.
accs1 = []
accs5 = []
scores_1 = []
scores_5 = []
encodings = []

num_classes_dict = {
    'imagenet': 1000,
    'imagenetv2': 1000,
    'cub': 200,
    'eurosat': 10,
    'places365': 365,
    'food101': 101,
    'pets': 37,
    'dtd': 47,
    'fgvcaircraft': 70,
    'cars': 196,
    'flowers102': 102,
    'caltech256': 257,
    'cifar100': 100
}

num_classes = num_classes_dict[opt.dataset]

accs1 = []
accs5 = []
accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
accuracy_metric_top5 = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)

filter_modes = ['filtered', 'filtered_dclip', 'filtered_equal_number', 'random_selection_dclip', 'random_selection_comparative']

if opt.mode in filter_modes:
    opt.reps = 5
    assert opt.reps <= 5, "If you want to iterate more than 5 times, you'll need to do some additional filtering."

origin_descriptor_fname = opt.descriptor_fname

for index in range(opt.reps):
    if opt.mode in filter_modes:
        if opt.descriptor_fname.endswith('.json'):
            opt.descriptor_fname = f"{origin_descriptor_fname[:-5]}_{index+1}.json"
        else:
            opt.descriptor_fname = f"{origin_descriptor_fname}_{index+1}"

    description_encodings = tools.compute_description_encodings(opt, model, mode=opt.mode)
    descr_means = torch.cat([x.mean(dim=0).reshape(1, -1) for x in description_encodings.values()])
    descr_means /= descr_means.norm(dim=-1, keepdim=True)

    for batch_number, (image_encodings, labels) in tqdm.tqdm(enumerate(zip(encoding_coll, label_coll)), total=len(encoding_coll), desc='Classifying images...'):
        image_encodings = image_encodings.to(device)
        labels = labels.to(device)

        if opt.merge_predictions:
            image_description_similarity = image_encodings @ descr_means.T
        else:
            image_description_similarity_t = [None] * opt.n_classes
            image_description_similarity_cumulative = [None] * opt.n_classes

            for i, (k, v) in enumerate(description_encodings.items()):
                image_description_similarity_t[i] = image_encodings @ v.T
                image_description_similarity_cumulative[i] = tools.aggregate_similarity(image_description_similarity_t[i], aggregation_method=opt.aggregate)

            image_description_similarity = torch.stack(image_description_similarity_cumulative, dim=1)
        
        accuracy_metric.update(image_description_similarity.softmax(dim=-1), labels)
        accuracy_metric_top5.update(image_description_similarity.softmax(dim=-1), labels)

    acc1 = accuracy_metric.compute().item() * 100
    accs1.append(acc1)
    acc5 = accuracy_metric_top5.compute().item() * 100
    accs5.append(acc5)
    print(f"[Mode = {opt.mode}] Top-1 Accuracy: {acc1:3.2f}%")
    print(f"[Mode = {opt.mode}] Top-5 Accuracy: {acc5:3.2f}%")

    accuracy_metric.reset()
    accuracy_metric_top5.reset()

### Print final results.
print(colored("\nFinal results", "red", attrs=["bold"]))            
print(f'After {opt.reps} reps using mode = {opt.mode} with merge = {opt.merge_predictions}:')
print(colored("Top-1 Accuracy", "white", attrs=["bold"]))
print('Mean Top-1 Acc: {0:3.2f}% +- {1:3.2f}%'.format(np.mean(accs1), np.std(accs1)))
print('Min and Max Top-1 Acc: {0:3.2f}% | {1:3.2f}%'.format(np.min(accs1), np.max(accs1)))
print('All Top-1 Accs: {0}'.format(' | '.join('{0:3.2f}%'.format(x) for x in accs1)))
print(colored("Top-5 Accuracy", "white", attrs=["bold"]))
print('Mean Top-5 Acc: {0:3.2f}% +- {1:3.2f}%'.format(np.mean(accs5), np.std(accs5)))
print('Min and Max Top-5 Acc: {0:3.2f}% | {1:3.2f}%'.format(np.min(accs5), np.max(accs5)))
print('All Top-5 Accs: {0}'.format(' | '.join('{0:3.2f}%'.format(x) for x in accs5)))

### Save results as csv.
import sys
import csv
os.makedirs('../results', exist_ok=True)
savename = '; '.join(x.replace('--','') for x in sys.argv[1:])
with open(f'../results/{opt.savename}.csv', 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([savename, np.mean(accs1), np.std(accs1), np.max(accs1), np.min(accs1), np.mean(accs5), np.std(accs5), np.max(accs5), np.min(accs5)])
    csv_file.close()    
    
### Save model information as pkl.
if opt.save_model != '':
    os.makedirs('../stored_models', exist_ok=True)
    pickle.dump({'scores_1': scores_1, 'scores_5': scores_5, 'encodings': encodings}, open(f'../stored_models/{opt.save_model}_{opt.dataset}.pkl', 'wb'))