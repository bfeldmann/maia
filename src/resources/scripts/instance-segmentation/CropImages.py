import os
import sys
import json
import cv2
from pyvips import Image as VipsImage
from pyvips.error import Error as VipsError
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class CropGenerator(object):
    def __init__(self, params):
        # Dict of image IDs and file paths to the images to process.
        self.annotations = params['annotations']

        # Path to cropped images
        self.cropped_images_path = '{}/cropped_images'.format(self.tmp_dir)

        # Dict of image IDs and file paths to the images to process.
        self.images = params['images']

        self.max_workers = params['max_workers']

        # Dimension used for resizing cropped images.
        self.resize_dimension = params['resize_dimension']

        # Path to the directory to store temporary files.
        self.tmp_dir = params['tmp_dir']

    def generate(self):
        # Check if dir(s) are there, create if missing
        self.ensure_dirs()
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        jobs = []

        for i, annotation in enumerate(self.annotations):
            jobs.append(executor.submit(self.process_image, i, annotation))

        images = []

        for job in as_completed(jobs):
            i = job.result()
            if i is not False:
                images.extend(i)

        if len(images) == 0:
            raise Exception('No images in dataset. All corrupt?')

        return {
            'cropped_images_path': self.cropped_images_path,
            'cropped_images': images,
            'resize_dimension': self.resize_dimension,
        }

    # Ensure path exists
    def ensure_dirs(self):
        if not os.path.exists(self.cropped_images_path):
           os.makedirs(self.cropped_images_path)

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

            return False

        return image_file

    # Generate a crop given the image and annotation
    # annotation:    0           1       2       3       4
    #           [$imageId, $xCenter, $yCenter, $radius, $score]
    def generate_crop(self, image, annotation):
        width, height = image.width, image.height
        radius = annotation[3]

        crop_width = min(width, 2*radius)
        crop_height = min(height, 2*radius)
        current_crop_dimension = np.array([crop_width, crop_height])

        center = np.array([annotation[1], annotation[2]])
        topLeft = np.round(center - current_crop_dimension / 2).astype(np.int32)
        bottomRight = np.round(center + current_crop_dimension / 2).astype(np.int32)
        offset = [0, 0]
        if topLeft[0] < 0: offset[0] = abs(topLeft[0])
        if topLeft[1] < 0: offset[1] = abs(topLeft[1])
        if bottomRight[0] > width: offset[0] = width - bottomRight[0]
        if bottomRight[1] > height: offset[1] = height - bottomRight[1]

        topLeft += offset
        bottomRight += offset

        image_crop = image.extract_area(topLeft[0], topLeft[1], current_crop_dimension[0], current_crop_dimension[1])

        return image_crop

with open(sys.argv[1]) as f:
    params = json.load(f)

# TODO: Crop the images, return a path to directory?
runner = CropGenerator(params)
output = runner.generate()

with open(params['output_path'], 'w') as f:
    json.dump(output, f)
