import os
import tqdm
import random
import pathlib
from collections import OrderedDict
from typing import Tuple, List, Union, Any

import argparse
import clip
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
import json
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import ImageNet,  Caltech256, CIFAR100, EuroSAT, Food101, Flowers102, Places365, OxfordIIITPet, DTD, FGVCAircraft, StanfordCars
from torch.utils.data import random_split
from datasets import _transform, CUBDataset

# List of methods available to use.
METHODS = [
    ##### baselines
    'clip',
    'dclip',
    'waffle',
    'waffle_and_gpt',
    ##### ours
    'comparative',
    'filtered',
    ##### additional experiment
    'filtered_dclip',
    'filtered_equal_number', # use an equal number of descriptors by filtering (k=5)
    'random_selection_dclip', # use an equal number of descriptors by random selection (k=5)
    'random_selection_comparative', # use an equal number of descriptors by random selection (k=5)
    #####
    'cot_filtered'
]

# List of compatible datasets.
DATASETS = [
    'imagenet', 
    'imagenetv2',
    'caltech256',
    'cifar100',
    'cub', 
    'places365', 
    'dtd', 
    'food101', 
    'eurosat', 
    'pets', 
    'flowers102', 
    'fgvcaircraft', 
    'cars',
]

# Dataset paths - These paths should be modified according to your environment
# All datasets should be placed in the /mnt/datasets/comparative-clip/ directory
IMAGENET_DIR = '/mnt/datasets/comparative-clip/ImageNet2012'
IMAGENETV2_DIR = '/mnt/datasets/comparative-clip/ImageNetV2'
CALTECH256_DIR = '/mnt/datasets/comparative-clip/Caltech256'
CIFAR100_DIR = '/mnt/datasets/comparative-clip/CIFAR100'
CUB_DIR = '/mnt/datasets/comparative-clip/cub200'
EUROSAT_DIR = '/mnt/datasets/comparative-clip/eurosat'
PLACES365_DIR = '/mnt/datasets/comparative-clip/places365'
PETS_DIR = '/mnt/datasets/comparative-clip/pets'
FOOD101_DIR = '/mnt/datasets/comparative-clip/food101'
DTD_DIR = '/mnt/datasets/comparative-clip/dtd'
FLOWERS102_DIR = '/mnt/datasets/comparative-clip/flowers102'
FGVCAIRCRAFT_DIR = '/mnt/datasets/comparative-clip/fgvcaircraft'
CARS_DIR = '/mnt/datasets/comparative-clip/cars'

# List of compatible backbones.
BACKBONES = [
    'RN50',
    'ViT-B/32',
    'ViT-L/14',    
]

# Default CLIP normalization stats
CLIP_STATS = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def setup(opt: argparse.Namespace):
    opt.image_size = 224
    if opt.model_size == 'ViT-L/14@336px' and opt.image_size != 336:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 336.')
        opt.image_size = 336
    elif opt.model_size == 'RN50x4' and opt.image_size != 288:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 288.')
        opt.image_size = 288
    elif opt.model_size == 'RN50x16' and opt.image_size != 384:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 384.')
        opt.image_size = 384
    elif opt.model_size == 'RN50x64' and opt.image_size != 448:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 448.')
        opt.image_size = 448

    opt.descriptor_fname = None

    # PyTorch datasets
    opt.tfms = _transform(opt.image_size)

    if opt.dataset == 'imagenet':
        dsclass = ImageNet        
        opt.data_dir = pathlib.Path(IMAGENET_DIR)
        dataset = dsclass(opt.data_dir, split='val', transform=opt.tfms)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_imagenet'
        
    elif opt.dataset == 'imagenetv2':
        opt.data_dir = pathlib.Path(IMAGENETV2_DIR)
        dataset = ImageNetV2(location=opt.data_dir, transform=opt.tfms)
        opt.classes_to_load = openai_imagenet_classes
        dataset.classes = opt.classes_to_load
        opt.descriptor_fname = 'descriptors_imagenetv2'

    elif opt.dataset == 'caltech256':
        dsclass = Caltech256
        opt.data_dir = pathlib.Path(CALTECH256_DIR)
        dataset = dsclass(opt.data_dir, transform=opt.tfms, download=True)
        torch.manual_seed(1)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _, test_dataset = random_split(dataset, [train_size, test_size])
        dataset=test_dataset
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_caltech256'
    
    elif opt.dataset == 'cifar100':
        dsclass = CIFAR100
        opt.data_dir = pathlib.Path(CIFAR100_DIR)
        dataset = dsclass(opt.data_dir, train=False, transform=opt.tfms, download=False)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_cifar100'

    elif opt.dataset == 'cub':
        opt.data_dir = pathlib.Path(CUB_DIR)
        dataset = CUBDataset(opt.data_dir, train=False, transform=opt.tfms)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_cub'

    elif opt.dataset == 'eurosat':
        dsclass = EuroSAT
        opt.data_dir = pathlib.Path(EUROSAT_DIR)
        dataset = dsclass(opt.data_dir, transform=opt.tfms, download=True)
        torch.manual_seed(1)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _, test_dataset = random_split(dataset, [train_size, test_size])
        dataset=test_dataset
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_eurosat'

    elif opt.dataset == 'places365':
        dsclass = Places365        
        opt.data_dir = pathlib.Path(PLACES365_DIR)
        download = True
        if os.path.exists(os.path.join(opt.data_dir, 'val_256')):
            download = False
        dataset = dsclass(opt.data_dir, split='val', small=True, transform=opt.tfms, download=download)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_places365'

    elif opt.dataset == 'food101':
        dsclass = Food101        
        opt.data_dir = pathlib.Path(FOOD101_DIR)
        dataset = dsclass(opt.data_dir, split='test', transform=opt.tfms, download=True)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_food101'

    elif opt.dataset == 'pets':
        dsclass = OxfordIIITPet   
        opt.data_dir = pathlib.Path(PETS_DIR)
        dataset = dsclass(opt.data_dir, split='test', transform=opt.tfms, download=True)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_pets'
        
    elif opt.dataset == 'dtd':
        dsclass = DTD        
        opt.data_dir = pathlib.Path(DTD_DIR)
        dataset = dsclass(opt.data_dir, split='test', transform=opt.tfms, download=True)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_dtd'

    elif opt.dataset == 'flowers102':
        dsclass = Flowers102
        opt.data_dir = pathlib.Path(FLOWERS102_DIR)
        dataset = dsclass(opt.data_dir, split='test', transform=opt.tfms, download=True)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_flowers102'
        
    elif opt.dataset == 'fgvcaircraft':
        dsclass = FGVCAircraft
        opt.data_dir = pathlib.Path(FGVCAIRCRAFT_DIR)
        dataset = dsclass(opt.data_dir, split='test', annotation_level='family', transform=opt.tfms, download=True)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_fgvcaircraft'
    
    elif opt.dataset == 'cars':
        dsclass = StanfordCars
        opt.data_dir = pathlib.Path(CARS_DIR)
        dataset = dsclass(opt.data_dir, split='test', transform=opt.tfms, download=False)
        opt.classes_to_load = None
        opt.descriptor_fname = 'descriptors_cars'
    
    if opt.descriptor_fname is not None:
        descriptors_path = '../descriptors'
        model_size = opt.model_size.replace('/', '-')

        if opt.mode == 'dclip':
            opt.descriptor_fname = f'{descriptors_path}/dclip_descriptors/' + opt.descriptor_fname
        elif opt.mode == 'filtered_dclip':
            opt.descriptor_fname = f'{descriptors_path}/filtered_dclip_descriptors/{model_size}/' + opt.descriptor_fname
        elif opt.mode == 'comparative':
            opt.descriptor_fname = f'{descriptors_path}/comparative_descriptors/' + opt.descriptor_fname
        elif opt.mode == 'filtered':
            opt.descriptor_fname = f'{descriptors_path}/filtered_descriptors/{model_size}/' + opt.descriptor_fname
        elif opt.mode == 'filtered_equal_number':
            opt.descriptor_fname = f'{descriptors_path}/equal_descriptors/{model_size}/' + opt.descriptor_fname
        elif opt.mode == 'random_selection_dclip':
            opt.descriptor_fname = f'{descriptors_path}/random_dclip_descriptors/' + opt.descriptor_fname
        elif opt.mode == 'random_selection_comparative':
            opt.descriptor_fname = f'{descriptors_path}/random_comparative_descriptors/' + opt.descriptor_fname
        elif opt.mode == 'cot_filtered':
            opt.descriptor_fname = f'{descriptors_path}/COT_filtered_descriptors/' + opt.descriptor_fname
        else:
            opt.descriptor_fname = f'{descriptors_path}/dclip_descriptors/' + opt.descriptor_fname
        
    return opt, dataset

def denormalize(
    images: torch.Tensor, 
    means: Union[Tuple[float],List[float]]=(0.485, 0.456, 0.406), 
    stds: Union[Tuple[float],List[float]]=(0.229, 0.224, 0.225)
):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means
  
def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)
    
def wordify(string: str):
    word = string.replace('_', ' ')
    return word

def make_descriptor_sentence(descriptor: str):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    elif descriptor == '':
        return ""
    else:
        return f"which has {descriptor}"
    
def modify_descriptor(descriptor: str, apply_changes: bool):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor

def load_gpt_descriptions(opt: argparse.Namespace, classes_to_load: List=None, mode: str='clip'):    
    ### Prepare extracted descriptions.
    gpt_descriptions = load_json(opt.descriptor_fname)
    
    # Replace empty descriptor lists if necessary.
    gpt_descriptions = {key: item if len(item) else [''] for key, item in gpt_descriptions.items()}
    
    ### (Lazy - uses gpt descriptions) Use the default CLIP setup.
    if not 'label_to_classname' in vars(opt):
        opt.label_to_classname = list(gpt_descriptions.keys())
        opt.n_classes = len(opt.label_to_classname)
    
    ### (Lazy - uses gpt descriptions) Use the default CLIP setup.    
    if mode == 'clip':
        gpt_descriptions = {l: opt.label_before_text + wordify(l) + opt.label_after_text for l in opt.label_to_classname}
        
    # Get complete list of available descriptions.
    descr_list = [list(values) for values in gpt_descriptions.values()]
    descr_list = np.array([x for y in descr_list for x in y])
    
    # List of available classes.
    key_list = list(gpt_descriptions.keys())                                       
    
    ### Descriptor Makers.
    structured_descriptor_builder = lambda item, cls: f"{opt.pre_descriptor_text}{opt.label_before_text}{wordify(cls)}{opt.descriptor_separator}{modify_descriptor(item, opt.apply_descriptor_modification)}{opt.label_after_text}" if modify_descriptor(item, opt.apply_descriptor_modification) else f"{opt.pre_descriptor_text}{opt.label_before_text}{wordify(cls)}."
    
    ### Use description-based CLIP.
    need_descr_modes = ['dclip', 'comparative', 'filtered', 'filtered_dclip', 'filtered_equal_number', 'random_selection_dclip', 'random_selection_comparative', 'cot_filtered']
    # need_descr_modes = ['filtered', 'filtered_dclip', 'filtered_equal_number', 'random_selection_dclip', 'random_selection_comparative'] # for Experiment-using_only_descriptor

    if mode in need_descr_modes:
        gpt_descriptions = {key: [structured_descriptor_builder(item, key) for item in class_descr_list] for key, class_descr_list in gpt_descriptions.items()}

    ### Use DCLIP with randomly assigned characters and words  (every class gets the same random subset).
    if mode == 'waffle':
        import pickle as pkl
        word_list = pkl.load(open('src/word_list.pkl', 'rb'))

        avg_num_words = int(np.max([np.round(np.mean([len(wordify(x).split(' ')) for x in key_list])), 1]))
        avg_word_length = int(np.round(np.mean([np.mean([len(y) for y in wordify(x).split(' ')]) for x in key_list])))        
        word_list = [x[:avg_word_length] for x in word_list]

        # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
        character_list = [x.split(' ') for x in descr_list]
        character_list = [x.replace(',', '').replace('.', '') for x in np.unique([x for y in character_list for x in y])]
        character_list = np.unique(list(''.join(character_list)))
        
        num_spaces = int(np.round(np.mean([np.sum(np.array(list(x)) == ' ') for x in key_list]))) + 1
        num_chars = int(np.ceil(np.mean([np.max([len(y) for y in x.split(' ')]) for x in key_list])))
            
        num_chars += num_spaces - num_chars%num_spaces
        sample_key = ''
        
        for s in range(num_spaces):
            for _ in range(num_chars//num_spaces):
                sample_key += 'a'
            if s < num_spaces - 1:
                sample_key += ' '
                
        gpt_descriptions = {key: [] for key in gpt_descriptions.keys()}
        
        for key in key_list:
            for _ in range(opt.waffle_count):
                base_word = ''                
                for a in range(avg_num_words):
                    base_word += np.random.choice(word_list, 1, replace=False)[0]
                    if a < avg_num_words - 1:
                        base_word += ' '
                gpt_descriptions[key].append(structured_descriptor_builder(base_word, key))
                noise_word = ''                
                use_key = sample_key if len(key) >= len(sample_key) else key
                for c in sample_key:
                    if c != ' ':
                        noise_word += np.random.choice(character_list, 1, replace=False)[0]
                    else:
                        noise_word += ', '
                gpt_descriptions[key].append(structured_descriptor_builder(noise_word, key))
                                
        match_key = np.random.choice(key_list)
        gpt_descriptions = {key: gpt_descriptions[match_key] for key in key_list}
        for key in gpt_descriptions:
            gpt_descriptions[key] = [x.replace(wordify(match_key), wordify(key)) for x in gpt_descriptions[key]]

    ### Use DCLIP with randomly assigned characters, words and GPT Descriptions (every class gets the same random subset, but tretains heir respective class descriptions).        
    ### However, each descriptor set is subsampled to match the required descriptor budget, so per class, descriptors may still vary a bit.    
    if mode == 'waffle_and_gpt':
        import pickle as pkl
        word_list = pkl.load(open('src/word_list.pkl', 'rb'))

        avg_num_words = int(np.max([np.round(np.mean([len(wordify(x).split(' ')) for x in key_list])), 1]))
        avg_word_length = int(np.round(np.mean([np.mean([len(y) for y in wordify(x).split(' ')]) for x in key_list])))        
        word_list = [x[:avg_word_length] for x in word_list]

        # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
        character_list = [x.split(' ') for x in descr_list]
        character_list = [x.replace(',', '').replace('.', '') for x in np.unique([x for y in character_list for x in y])]
        character_list = np.unique(list(''.join(character_list)))
        
        num_spaces = int(np.round(np.mean([np.sum(np.array(list(x)) == ' ') for x in key_list]))) + 1
        num_chars = int(np.ceil(np.mean([np.max([len(y) for y in x.split(' ')]) for x in key_list])))            
        num_chars += num_spaces - num_chars%num_spaces
        
        sample_key = ''
        for s in range(num_spaces):
            for _ in range(num_chars//num_spaces):
                sample_key += 'a'
            if s < num_spaces - 1:
                sample_key += ' '
                
        base_gpt_descriptions = {key: items for key, items in gpt_descriptions.items()}
        all_descr = [values for values in base_gpt_descriptions.values()]
        all_descr = [x for y in all_descr for x in y]
        gpt_descriptions = {key: [] for key in gpt_descriptions.keys()}

        effective_waffle_count = int(2/3 * opt.waffle_count)        
        
        for key in key_list:
            for sc in range(effective_waffle_count):
                base_word = ''                
                for a in range(avg_num_words):
                    base_word += np.random.choice(word_list, 1, replace=False)[0]
                    if a < avg_num_words - 1:
                        base_word += ' '
                gpt_descriptions[key].append(structured_descriptor_builder(base_word, key))
                noise_word = ''                
                for c in sample_key:
                    if c != ' ':
                        noise_word += np.random.choice(character_list, 1, replace=False)[0]
                    else:
                        noise_word += ', '
                gpt_descriptions[key].append(structured_descriptor_builder(noise_word, key))

        match_key = np.random.choice(key_list)
        gpt_descriptions = {key: gpt_descriptions[match_key] for key in key_list}
        for key in gpt_descriptions:
            gpt_descriptions[key] = [x.replace(wordify(match_key), wordify(key)) for x in gpt_descriptions[key]]

        # For every random character and word descriptor pair, we add a GPT descriptor
        # sampled from the list of available descriptors.
        for key in key_list:
            for sc in range(effective_waffle_count):
                word = np.random.choice(base_gpt_descriptions[key], 1)[0]
                gpt_descriptions[key].append(structured_descriptor_builder(word, key))
                word = np.random.choice(base_gpt_descriptions[key], 1)[0]
                gpt_descriptions[key].append(structured_descriptor_builder(word, key))
        
        # To ensure the correct number of random word sequences, random character sequences and GPT descriptions, we
        # subsample for each class individually. This does result in slightly different descriptor distributions per class.
        for key in key_list:
            gpt_descriptions[key] = list(np.random.choice(gpt_descriptions[key], effective_waffle_count * 3, replace=False))

    ### If a specific class subset should be used, we subsample here:
    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions[c] for c in classes_to_load}
        keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
        for k in keys_to_remove:
            print(f"Skipping descriptions for \"{k}\", not in classes to load")
            gpt_descriptions.pop(k)        
        
    ### Because of the inconsistent class naming, we have to do some re-sorting of the keys.        
    if opt.dataset == 'pets':
        sorted_keys = np.array(list(gpt_descriptions.keys()))[np.argsort([x.lower() for x in gpt_descriptions.keys()])]
        gpt_descriptions = {key: gpt_descriptions[key] for key in sorted_keys}
    
    ### Only produce sample prompt if in the right call.
    ### > Want to avoid printing prompts several times.
    if mode == opt.mode:
        sel_prompt = gpt_descriptions[key_list[1]]
        if isinstance(sel_prompt, list):
            sel_prompt = sel_prompt[0]
        print(f'Sample Prompt: {sel_prompt}')
        
    return gpt_descriptions

def compute_description_encodings(opt: argparse.Namespace, model: Any, mode: str='clip'):
    gpt_descriptions = load_gpt_descriptions(opt, opt.classes_to_load, mode=mode)
    description_encodings = OrderedDict()
    for k, v in tqdm.tqdm(gpt_descriptions.items(), total=len(gpt_descriptions), desc='Encoding Descriptions...'):
        description_encodings[k] = F.normalize(model.encode_text(clip.tokenize(v).to(opt.device)))
    return description_encodings

def aggregate_similarity(similarity_matrix_chunk: torch.Tensor, aggregation_method: str='mean'):
    if aggregation_method == 'max': 
        return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': 
        return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': 
        return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")

openai_imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "eft", 
                           "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", 
                           "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", 
                           "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", 
                           "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox","maillot", 
                           "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", 
                           "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
