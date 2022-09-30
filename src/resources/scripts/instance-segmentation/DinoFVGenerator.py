# TODO: Adapt script to handle list of cropped images as json or path to directory
# Unneeded imports?
from pathlib import Path
from PIL import Image
from PIL import UnidentifiedImageError

# Kept imports:
import os
import sys
import json
import cv2
from pyvips import Image as VipsImage
from pyvips.error import Error as VipsError
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Adapted imports:
import torch
from torchvision import transforms as pth_transforms

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    TEST = '\033[97m'

# Generates feature maps using DINO and saves them as .npz.

# Simplified preprocessing
preprocess_simple = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# TODO: Adapt
class DinoFVGenerator(object):
    def __init__(self, params):
        # Dict of cropped image IDs and file paths to the cropped images to process.
        self.cropped_images = params['cropped_images']
        # Architectures: See https://github.com/facebookresearch/dino#pretrained-models-on-pytorch-hub
        # E.g.: dino_resnet50, dino_vitb8, dino_vitb16, dino_vits8, dino_vits16
        self.backbone = params['backbone']

        # Path to the directory to store temporary files.
        self.tmp_dir = params['tmp_dir']
        # Estimated available GPU memory in bytes.
        # TODO: Rausnehmen?
        self.available_bytes = params['available_bytes']

        self.max_workers = params['max_workers']

        self.fv_path = '{}/fv'.format(self.tmp_dir)

    def generate(self):
        # Check if dir(s) are there, create if missing
        self.ensure_dirs()
        model = torch.hub.load('facebookresearch/dino:main', backbone)
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        jobs = []

        # TODO: Add image processing from main()
        for i, crop in enumerate(self.cropped_images):
            jobs.append(executor.submit(self.process_image, crop))

        fv_files = []

        for job in as_completed(jobs):
            fv = job.result()
            if fv is not False:
                fv_files.extend(i)

        if len(fv_files) == 0:
            raise Exception('No feature vectors in dataset. All corrupt?')

        return {
            'fv_path': self.cropped_images_path,
            'fv_files': fv_files,
        }

    # Ensure path exists
    def ensure_dirs(self):
        if not os.path.exists(self.fv_path):
           os.makedirs(self.fv_path)

    def process_image(self, i, annotation):
        try:
            imageId = annotation[0]
            image = VipsImage.new_from_file(self.images[imageId])
            crop_paths = []

            image_file = '{}_{}_cropped.jpg'.format(imageId, i)
            image_crop, = self.generate_crop(image, annotation)
            # Standardize image sizes by resizing to specified dimensions
            image_crop = image_crop.resize(self.resize_dimension, self.resize_dimension)

            image_crop.write_to_file(os.path.join(self.cropped_images_path, image_file), strip=True, Q=100)

        except VipsError as e:
            print('Image #{} is corrupt! Skipping...'.format(imageId))

            return False, False, False

        return image_file

    # Gets the features from the given DINO model
    def getFeatures(model, path):
        features = None
        try:
            img = Image.open(path)
            features = (model(preprocess_simple(img).unsqueeze(0))).detach().numpy()[0]
            img.close()
        except FileNotFoundError:
            print(bc.FAIL+'Cannot find file: '+path+bc.ENDC)
        except UnidentifiedImageError:
            print(bc.FAIL+'Cannot open image: '+path+bc.ENDC)
        return np.reshape(features, (1, len(features)))

    # Saves the features as .npz.
    def saveFeatures(target, path, features, suffix='.dino'):
        np.savez_compressed(target+os.path.basename(path)+suffix, features=features)

with open(sys.argv[1]) as f:
    params = json.load(f)

# TODO: Crop the images, return a path to directory?
runner = DinoFVGenerator(params)
output = runner.generate()

with open(params['output_path'], 'w') as f:
    json.dump(output, f)

# TODO: Bring this into the generator, adapt to use Json params
def main():
    target = sys.argv[1]
    if not target.endswith(os.sep):
        target += os.sep
    Path(target).mkdir(parents=True, exist_ok=True)
    backbone = sys.argv[2]
    paths = sys.argv[3:]
    # TODO: Wenn schon vorhanden, einfach vorhandenes Modell benutzen? Siehe MaskRCNN-Ansatz
    model = torch.hub.load('facebookresearch/dino:main', backbone)
    executor = ProcessPoolExecutor(max_workers=os.cpu_count())
    for i, path in enumerate(paths):
        print(f'{i+1}/{len(paths)} Processing image: {path}')
        try:
            features = getFeatures(model, path)
            executor.submit(saveFeatures, target, path, features, '.simple_'+backbone)
        except Exception as e:
            print(f'{bc.FAIL}Exception during processing of image {i+1}/{len(paths)}: {e}{bc.ENDC}')
    executor.shutdown(wait=True)
