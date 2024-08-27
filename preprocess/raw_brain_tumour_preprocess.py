import os
import numpy as np
import h5py
from multiprocessing import Pool
from matplotlib import pyplot as plt


def process_single_file(args):
    """
    Processes a single .mat file, extracts the image, mask, and metadata, and saves them as grayscale PNGs.

    Parameters:
    - args: A tuple containing (mat_path, output_image_dir, output_mask_dir)
    """
    mat_path, output_image_dir, output_mask_dir = args

    with h5py.File(mat_path, 'r') as mat_file:
        img = mat_file['cjdata']['image']
        label = mat_file['cjdata']['label'][0][0]
        mask = mat_file['cjdata']['tumorMask']

        # convert to numpy arrays and normalise
        img = np.array(img, dtype=np.float32)
        img = img / 127.5 - 1
        mask = np.array(mask, dtype=np.float32)
        mask = mask / 127.5 - 1

        # filenames
        base_filename = os.path.splitext(os.path.basename(mat_path))[0]
        image_save_path = os.path.join(output_image_dir, f"{base_filename}_image.png")
        mask_save_path = os.path.join(output_mask_dir, f"{base_filename}_mask.png")

        # save the image and mask using matplotlib
        plt.imsave(image_save_path, img, cmap='gray')
        plt.imsave(mask_save_path, mask, cmap='gray')

        print(f"Processed and saved: {base_filename} (Label: {label})")


def process_mat_files_parallel(input_dirs, output_image_dir, output_mask_dir, num_workers=4):
    """
    Processes .mat files in parallel, extracting images, masks, and metadata, and saving them.

    Parameters:
    - input_dirs (list): List of directories containing .mat files.
    - output_image_dir (str): Directory to save the processed images.
    - output_mask_dir (str): Directory to save the processed masks.
    - num_workers (int): Number of parallel processes to use.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # paths to all .mat files
    mat_files = []

    # all .mat file paths from the input directories
    for input_dir in input_dirs:
        for filename in os.listdir(input_dir):
            if filename.endswith('.mat'):
                mat_files.append(os.path.join(input_dir, filename))

    # the arguments for each .mat file
    args = [(mat_path, output_image_dir, output_mask_dir) for mat_path in mat_files]

    # the .mat files in parallel
    with Pool(num_workers) as pool:
        for result in pool.map(process_single_file, args):
            print(result)

