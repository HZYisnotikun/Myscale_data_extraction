import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPModel, CLIPProcessor
import os
Image.MAX_IMAGE_PIXELS = None
import random
import torch

def load_clip_model_and_processor(device, clip_path):
    model = CLIPModel.from_pretrained(clip_path).to(device)
    processor = CLIPProcessor.from_pretrained(clip_path)
    return model, processor


def batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size):
    """
        Processes a list of image file paths in batches, computes image embeddings using the CLIP model.
        Args:
            image_paths (list): A list of file paths to the images that need to be processed.
            clip_model (CLIPModel): A pre-trained CLIP model used to compute the image embeddings.
            clip_processor (CLIPProcessor): A processor to preprocess images before passing them to the CLIP model.
            device (torch.device): The device (CPU or GPU) on which the computations will be performed.
            batch_size (int): The number of images to process in each batch.

        Returns:
            tuple: A tuple containing:
                - embeddings (list of lists): A list of computed image embeddings, where each embedding
                  is a list of floats representing the image in vector space.
                - valid_paths (list of str): A list of file paths corresponding to the images
                  that were successfully processed and embedded.
        """
    random.shuffle(image_paths)
    embeddings = []  # List to store image embeddings
    valid_paths = []  # List to store valid image file paths, ensuring the lengths of embeddings and valid paths are consistent, since only successfully processed images are included.
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = []  # List to store loaded images
        current_valid_paths = []  # List to store valid paths in the current batch
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
                current_valid_paths.append(path)
            except Exception as e:
                print(f"Error processing image file: {path}, error: {e}")
                continue

        if not images:
            # Skip the batch if no valid images were found
            continue

        try:
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                batch_embeddings = clip_model.get_image_features(**inputs)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            valid_paths.extend(current_valid_paths)
        except Exception as e:
            print(f"Error processing images in batch {batch_paths}: {e}")
            continue

    return embeddings, valid_paths


