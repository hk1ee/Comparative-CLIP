import json
import torch
import pickle
import pathlib
import argparse
from tqdm import tqdm
from torchvision import transforms
from datasets import CUBDataset
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from torchvision.datasets import ImageNet, EuroSAT, Food101, Flowers102, Places365, OxfordIIITPet, DTD, FGVCAircraft, StanfordCars, CIFAR100, Caltech256, SUN397
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import clip
import tools
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Process image and text features.')
parser.add_argument('--dataset', type=str, required=True, choices=tools.DATASETS, help='Dataset to extract features from.')
parser.add_argument('--batch_size', type=int, default=640, help='Batch size for feature extraction.')
parser.add_argument('--image_size', type=int, default=224, help='Image size')
parser.add_argument('--model_size', type=str, required=True, choices=tools.BACKBONES, help='Pretrained CLIP model to use.')
opt = parser.parse_args()

tools.seed_everything(1)

def load_dataset(name):
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

    mean, std = tools.CLIP_STATS
    transform = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.Grayscale(num_output_channels=3),  # RuntimeError: output with shape [1, W, H] doesn't match the broadcast shape [3, W, H]
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if name == 'imagenet' or name == 'imagenetv2':
        dataset = ImageNet(root=tools.IMAGENET_DIR, split='train', transform=transform)
    elif name == 'cub':
        dataset = CUBDataset(root=tools.CUB_DIR, train=True, transform=transform)
    elif name == 'eurosat':
        import shutil
        import ssl
        from glob import glob
        import os
        
        dsclass = EuroSAT
        ssl._create_default_https_context = ssl._create_unverified_context
        opt.data_dir = tools.EUROSAT_DIR
        dataset = dsclass(opt.data_dir, transform=transform, download=True)
        opt.classes_to_load = None
        
        train_folder_path = str(opt.data_dir) + '/train/eurosat/2750/'
        test_folder_path = str(opt.data_dir) + '/test/eurosat/2750/'

        if not os.path.exists(train_folder_path):
            for class_folder in glob(os.path.join(str(opt.data_dir)+'/eurosat/2750/', '*')):
                class_name = os.path.basename(class_folder)
                image_paths = glob(os.path.join(class_folder, '*'))
                
                train_class_folder = os.path.join(train_folder_path, class_name)
                test_class_folder = os.path.join(test_folder_path, class_name)
                os.makedirs(train_class_folder, exist_ok=True)
                os.makedirs(test_class_folder, exist_ok=True)
                
                for i, image_path in enumerate(image_paths):
                    destination_path = train_class_folder if i < 64 else test_class_folder
                    destination_file_path = os.path.join(destination_path, os.path.basename(image_path))

                    shutil.copyfile(image_path, destination_file_path)
            
        dataset = dsclass(str(opt.data_dir) + '/train/', transform=transform, download=False)
    elif name == 'places365':
        dataset = Places365(root=tools.PLACES365_DIR, split='train-standard', small=True, transform=transform)
    elif name == 'pets':
        dataset = OxfordIIITPet(root=tools.PETS_DIR, split='trainval', transform=transform)
    elif name == 'food101':
        dataset = Food101(root=tools.FOOD101_DIR, split='train', transform=transform)
    elif name == 'dtd':
        dataset = DTD(root=tools.DTD_DIR, split='train', transform=transform)
    elif name == 'flowers102':
        dataset = Flowers102(root=tools.FLOWERS102_DIR, split='train', transform=transform)
    elif name == 'fgvcaircraft':
        dataset = FGVCAircraft(root=tools.FGVCAIRCRAFT_DIR, split='train', annotation_level='family', transform=transform)
    elif name == 'cars':
        dataset = StanfordCars(root=tools.CARS_DIR, split='train', transform=transform)
    elif name == 'cifar100':
        dataset = CIFAR100(root=tools.CIFAR100_DIR, train=True, transform=transform)
    elif name == 'caltech256':
        data_dir = pathlib.Path(tools.CALTECH256_DIR)
        dataset = Caltech256(data_dir, transform=transform, download=True)
        torch.manual_seed(1)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, _ = random_split(dataset, [train_size, test_size])
        dataset = train_dataset
    else:
        raise ValueError(f"Dataset {name} is not supported.")
    
    return dataset

def extract_features(device, model, data_loader, class_names):
    model.eval()
    class_features = {}
    print(f'Dataset: {opt.dataset}')
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Extracting Features..'):
            images = images.to(device)
            image_features = model.encode_image(images).cpu().numpy()
            for feature, label in zip(image_features, labels.numpy()):
                class_name = class_names[label]
                if class_name not in class_features:
                    class_features[class_name] = []
                class_features[class_name].append(feature)
    return class_features

def get_class_names(name):
    # HACK
    if name == 'imagenetv2':
        class_names = tools.openai_imagenet_classes
    else:
        with open(f'../descriptors/dclip_descriptors/descriptors_{name}.json', 'rb') as file:
            data = json.load(file)
        class_names = list(data.keys())

    return class_names

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(opt.model_size, device=device)
    model_size = opt.model_size.replace('/', '-')
    
    dataset = load_dataset(opt.dataset)
    class_names = get_class_names(opt.dataset)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=16)

    class_features = extract_features(device, model, data_loader, class_names)

    output_file = f"../precomputed_image_features/{model_size}/{opt.dataset}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(class_features, f)
    
    if opt.dataset == 'imagenet': # duplicate
        output_file = f"../precomputed_image_features/{model_size}/imagenetv2.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(class_features, f)

    print(f"Features extracted and saved as {output_file}")

if __name__ == "__main__":
    main()