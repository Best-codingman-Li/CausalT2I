import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import functools
import gc

import hashlib
import itertools
import json
import logging

import math
import torch.nn as nn

import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import clip
import open_clip
import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo,upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPTextModelWithProjection
from maskmlp import MaskMLP, MergeMaskMLP, Filter

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    CustomDiffusionXFormersAttnProcessor,
)
from CustomAttnProcessor import CustomDiffusionAttnProcessor
#from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from CustomModelLoader import CustomModelLoader
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available, is_xformers_available#, is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_peft_version,
    is_wandb_available,
)

import wandb

from CrossAttnMap import AttentionStore, aggregate_current_attention
#from gaussian_smoothing import GaussianSmoothing


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)

def show_attention_map_during_training(cross_attention_map, obj, out_path, global_step):
    split_tensors = torch.split(cross_attention_map, 1, dim=0)

    # create a list to store the PIL images
    images = []

    # loop over the split tensors and show the PIL image of each element
    for i, tensor in enumerate(split_tensors):
        # convert the tensor to a PIL image
        image = Image.fromarray(tensor.squeeze().mul(255).clamp(0, 255).byte().cpu().numpy())

        # append the image to the list
        images.append(image)

    # combine the images into a 1 row 4 column image
    combined_image = Image.new(mode="RGB", size=(cross_attention_map.shape[1] * len(split_tensors), cross_attention_map.shape[2]))
    for i, image in enumerate(images):
        combined_image.paste(image, (i * cross_attention_map.shape[1], 0))
    combined_image.save(os.path.join(out_path, f"cross_atten_map_{global_step}_.jpg"))
    # log the combined image to wandb
    #wandb.log({f"cross_attention_maps_obj_{obj}": wandb.Image(combined_image)})


def freeze_params(params):
    for param in params:
        param.requires_grad = False


'''
def save_model_card(
    repo_id: str,
    images: list = None,
    validation_prompt: str = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n
{img_str}

Special VAE used for training: {vae_path}.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers-training",
        "diffusers",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

'''

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")



def collate_fn(examples, with_prior_preservation, concept_list):
    pri_pixel_values, pri_mask, pri_ids, batch, input_ids, input_ids_2, ori_pixel_values, pixel_values, mask, crop_top_left = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    for concept in concept_list:
        input_ids[concept["class_prompt"]] = [example[concept["class_prompt"]]["instance_prompt_ids"] for example in
                                              examples]
        input_ids_2[concept["class_prompt"]] = [example[concept["class_prompt"]]["instance_prompt_ids_2"] for example in
                                              examples]
        pixel_values[concept["class_prompt"]] = [example[concept["class_prompt"]]["instance_images"] for example in
                                                 examples]

        ori_pixel_values[concept["class_prompt"]] = [example[concept["class_prompt"]]["ori_instance_images"] for
                                                     example
                                                     in examples]
        mask[concept["class_prompt"]] = [example[concept["class_prompt"]]["mask"] for example in examples]

        crop_top_left[concept["class_prompt"]] = [example[concept["class_prompt"]]["crop_top_left"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids[concept["class_prompt"]] += [example[concept["class_prompt"]]["class_prompt_ids"] for example
                                                   in
                                                   examples]
            pixel_values[concept["class_prompt"]] += [example[concept["class_prompt"]]["class_images"] for example
                                                      in
                                                      examples]
            mask[concept["class_prompt"]] += [example[concept["class_prompt"]]["class_mask"] for example in
                                              examples]

        input_ids[concept["class_prompt"]] = torch.cat(input_ids[concept["class_prompt"]], dim=0)
        input_ids_2[concept["class_prompt"]] = torch.cat(input_ids_2[concept["class_prompt"]], dim=0)
        
        crop_top_left[concept["class_prompt"]] = torch.stack(crop_top_left[concept["class_prompt"]])

        pixel_values[concept["class_prompt"]] = torch.stack(pixel_values[concept["class_prompt"]])
        ori_pixel_values[concept["class_prompt"]] = torch.stack(ori_pixel_values[concept["class_prompt"]])
        mask[concept["class_prompt"]] = torch.stack(mask[concept["class_prompt"]])
        pixel_values[concept["class_prompt"]] = pixel_values[concept["class_prompt"]].to(
            memory_format=torch.contiguous_format).float()
        mask[concept["class_prompt"]] = mask[concept["class_prompt"]].to(
            memory_format=torch.contiguous_format).float()
        
        ori_pixel_values[concept["class_prompt"]] = ori_pixel_values[concept["class_prompt"]].to(
            memory_format=torch.contiguous_format).float()

        batch[concept["class_prompt"]] = {"ins_input_ids": input_ids[concept["class_prompt"]],
                                          "ins_input_ids_2": input_ids_2[concept["class_prompt"]],
                                          "ins_pixel_values": pixel_values[concept["class_prompt"]],
                                          "mask": mask[concept["class_prompt"]].unsqueeze(1),
                                          "ori_pixel_values": ori_pixel_values[concept["class_prompt"]],
                                          "crop_top_left":crop_top_left[concept["class_prompt"]]}

    del input_ids, pixel_values, mask, ori_pixel_values, input_ids_2, crop_top_left
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class MultiSubjectDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            tokenizer_1,
            tokenizer_2,
            size=512,
            mask_size=64,
            center_crop=False,
            with_prior_preservation=False,
            num_class_images=200,
            hflip=False,
            aug=True,
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug
        self.concepts_list = concepts_list
        '''
        self.instance_images_path = []
        self.class_images_path = []
        '''
        self.instance_images_path = {}
        self.class_images_path = {}
        self.with_prior_preservation = with_prior_preservation

        self.original_sizes = []
        self.crop_top_lefts = []

        for concept in self.concepts_list:
            self.instance_images_path[concept["class_prompt"]] = []
            self.class_images_path[concept["class_prompt"]] = []

        for concept in self.concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path[concept["class_prompt"]].extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"] + "/" + "images")
                if os.path.isdir(class_data_root):
                    # print("os.path.isdir(class_data_root)!")
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path[concept["class_prompt"]].extend(class_img_path[:num_class_images])

            random.shuffle(self.instance_images_path[concept["class_prompt"]])

        self.num_instance_images = len(self.instance_images_path[self.concepts_list[0]["class_prompt"]])
        self.num_class_images = len(self.class_images_path[self.concepts_list[0]["class_prompt"]])
        self._length = max(self.num_class_images, self.num_instance_images)
        #self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.heatmap_transforms = transforms.Compose(
            [  
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),#size of the heatmap is 256*256
                transforms.CenterCrop(256) if center_crop else transforms.RandomCrop(256),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top: top + inner, left: left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top: top + inner, left: left + inner, :] = image
            mask[
            top // factor + 1: (top + scale) // factor - 1, left // factor + 1: (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        for concept in self.concepts_list:
            example[concept["class_prompt"]] = {}
        for concept in self.concepts_list:
            instance_image, instance_prompt = self.instance_images_path[concept["class_prompt"]][
                index % self.num_instance_images]
            instance_image = Image.open(instance_image)
            #example[concept["class_prompt"]]["ori_instance_images"] = instance_image
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")

            if args.random_flip and random.random() < 0.5:
                # flip
                instance_image = self.train_flip(instance_image)

            instance_image = self.train_resize(instance_image)
            example[concept["class_prompt"]]["ori_instance_images"] = self.image_transforms(instance_image)
            # apply resize augmentation and create a valid image region mask
            random_scale = self.size
            if self.aug:
                random_scale = (
                    np.random.randint(self.size // 3, self.size + 1)
                    if np.random.uniform() < 0.66
                    else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
                )
            instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

            if args.center_crop:
                y1 = max(0, int(round((instance_image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((instance_image.width - args.resolution) / 2.0)))
                instance_image = self.train_crop(instance_image)
            else:
                y1, x1, h, w = self.train_crop.get_params(instance_image, (args.resolution, args.resolution))
                instance_image = crop(instance_image, y1, x1, h, w)
        
            crop_top_left = (y1, x1)
            example[concept["class_prompt"]]["crop_top_left"] = crop_top_left

            if random_scale < 0.6 * self.size:
                instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
            elif random_scale > self.size:
                instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

            example[concept["class_prompt"]]["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
            example[concept["class_prompt"]]["mask"] = torch.from_numpy(mask)
            example[concept["class_prompt"]]["instance_prompt_ids"] = self.tokenizer_1(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_1.model_max_length,
                return_tensors="pt",
            ).input_ids

            example[concept["class_prompt"]]["instance_prompt_ids_2"] = self.tokenizer_2(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids

            if self.with_prior_preservation:
                class_image, class_prompt = self.class_images_path[concept["class_prompt"]][
                    index % self.num_class_images]
                class_image = Image.open(class_image)
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                example[concept["class_prompt"]]["class_images"] = self.image_transforms(class_image)
                example[concept["class_prompt"]]["class_mask"] = torch.ones_like(
                    example[concept["class_prompt"]]["mask"])
                example[concept["class_prompt"]]["class_prompt_ids"] = self.tokenizer(
                    class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids

        return example


class Merge_MultiSubjectDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            tokenizer_1,
            tokenizer_2,
            size=1024,
            mask_size=64,
            center_crop=False,
            with_prior_preservation=False,
            num_class_images=200,
            hflip=False,
            aug=True,
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

        self.interpolation = Image.BILINEAR
        self.aug = aug
        self.concepts_list = concepts_list
        self.original_sizes = []
        self.crop_top_lefts = []

        self.instance_images_path = {}
        self.class_images_path = {}
        self.with_prior_preservation = with_prior_preservation
        self.num_subject_classes = len(self.concepts_list)
        for concept in self.concepts_list:
            self.instance_images_path[concept["class_prompt"]] = []
            self.class_images_path[concept["class_prompt"]] = []

            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]

            self.instance_images_path[concept["class_prompt"]].extend(inst_img_path)
            # print("self.instance_images_path[concept[class_prompt]] = ", self.instance_images_path[concept["class_prompt"]])
            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"] + "/" + "images")
                class_data_root.mkdir(parents=True, exist_ok=True)
                if os.path.isdir(class_data_root):
                    # print("os.path.isdir(class_data_root)!")
                    class_images_path = list(class_data_root.iterdir())
                    # print("ori class_images_path = ", class_images_path)

                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]

                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path[concept["class_prompt"]].extend(class_img_path[:num_class_images])

            random.shuffle(self.instance_images_path[concept["class_prompt"]])
            # print("self.instance_images_path = ", self.instance_images_path[concept["class_prompt"]])

        self.num_instance_images = min(4,len(self.instance_images_path[self.concepts_list[0]["class_prompt"]]))
        self.num_class_images = len(self.class_images_path[self.concepts_list[0]["class_prompt"]])
        self._length = max(self.num_class_images, self.num_instance_images)
        #self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top: top + inner, left: left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top: top + inner, left: left + inner, :] = image
            mask[
            top // factor + 1: (top + scale) // factor - 1, left // factor + 1: (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        original_h,  original_w = [], []
        instance_images_w_h = [1024, 512]
        instance_images = [0] * self.num_subject_classes

        class_images_w_h = [0, 0]
        class_images = [0] * self.num_subject_classes

        # combine the images of different subjects
        for i in range(self.num_subject_classes):
            if i == 0:
                instance_images[i] = Image.open(
                    self.instance_images_path[self.concepts_list[i]["class_prompt"]][index % self.num_instance_images][
                        0])
                original_h.append(instance_images[i].height)
                original_w.append(instance_images[i].width)

                instance_images[i] = instance_images[i].resize((512,1024))
                instance_images_w_h[0], instance_images_w_h[1] = instance_images[i].size
                instance_image = Image.new("RGB",
                                           (instance_images_w_h[0] * self.num_subject_classes,
                                            instance_images_w_h[1]))
                instance_image.paste(instance_images[i], (i, 0))

                instance_prompt = \
                    self.instance_images_path[self.concepts_list[i]["class_prompt"]][index % self.num_instance_images][
                        1]
            else:
                instance_images[i] = Image.open(
                    self.instance_images_path[self.concepts_list[i]["class_prompt"]][index % self.num_instance_images][
                        0])
                original_h.append(instance_images[i].height)
                original_w.append(instance_images[i].width)

                instance_images[i] = instance_images[i].resize((512, 1024))
                instance_image.paste(instance_images[i], (instance_images_w_h[0] * i, 0))

                instance_prompt = instance_prompt + ' and ' + \
                                  self.instance_images_path[self.concepts_list[i]["class_prompt"]][
                                      index % self.num_instance_images][1]

            if self.with_prior_preservation:
                class_image, class_prompt = self.class_images_path[self.concepts_list[i]["class_prompt"]][
                    index % self.num_class_images]

                class_image = Image.open(class_image)
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                example["class_images"] = self.image_transforms(class_image)
                # example[concept["class_prompt"]]["class_mask"] = torch.ones_like(example[concept["class_prompt"]]["mask"])
                example["class_prompt_ids"] = self.tokenizer(
                    class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids

                example["class_mask"] = self.tokenizer(
                    class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).attention_mask

        
        instance_image = self.train_resize(instance_image)

        if args.random_flip and random.random() < 0.5:
            # flip
            instance_image = self.train_flip(instance_image)
        if args.center_crop:
            y1 = max(0, int(round((instance_image.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((instance_image.width - args.resolution) / 2.0)))
            instance_image = self.train_crop(instance_image)
        else:
            y1, x1, h, w = self.train_crop.get_params(instance_image, (args.resolution, args.resolution))
            instance_image = crop(instance_image, y1, x1, h, w)
        
        crop_top_left = (y1, x1)
        #self.crop_top_lefts.append(crop_top_left)

        if index == 0:
            instance_image.save("./dog_sunglasses_cat_merge.jpg")
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        example["ori_instance_image"] = self.image_transforms(instance_image)
        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids_1"] = self.tokenizer_1(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids

        example["atten_mask_1"] = self.tokenizer_1(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).attention_mask

        example["instance_prompt_ids_2"] = self.tokenizer_2(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids

        example["atten_mask_2"] = self.tokenizer_2(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).attention_mask

        example['crop_top_left'] = crop_top_left
        example['original_size'] = (max(original_h), sum(original_w))

        return example


def merge_collate_fn(examples, with_prior_preservation, concept_list):
    batch = {}

    ins_input_ids = [example["instance_prompt_ids_1"] for example in examples]
    ins_input_ids_2 = [example["instance_prompt_ids_2"] for example in examples]
    ins_pixel_values = [example["instance_images"] for example in examples]
    ins_atten_mask = [example["atten_mask_1"] for example in examples]
    ins_atten_mask_2 = [example["atten_mask_2"] for example in examples]
    ins_mask = [example["mask"] for example in examples]
    ori_instance_image = [example["ori_instance_image"] for example in examples]

    crop_top_left = [example["crop_top_left"] for example in examples]
    original_size = [example["original_size"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pri_ids = [example["class_prompt_ids"] for example in examples]
        pri_pixel_values = [example["class_images"] for example in examples]
        pri_mask = [example["class_mask"] for example in examples]
        pri_ids = torch.cat(pri_ids, dim=0)
        pri_pixel_values = torch.stack(pri_pixel_values).to(
            memory_format=torch.contiguous_format).float()
        pri_mask = torch.stack(pri_mask).to(
            memory_format=torch.contiguous_format).float()

    ins_input_ids = torch.cat(ins_input_ids, dim=0)
    ins_pixel_values = torch.stack(ins_pixel_values).to(
        memory_format=torch.contiguous_format).float()
    ins_atten_mask = torch.stack(ins_atten_mask).to(
        memory_format=torch.contiguous_format).float()
    ins_mask = torch.stack(ins_mask)

    ori_instance_image = torch.stack(ori_instance_image).to(
        memory_format=torch.contiguous_format).float()

    # .unsqueeze(1)
    if with_prior_preservation:
        batch = {"ins_input_ids": ins_input_ids,
                 "ins_input_ids_2": ins_input_ids_2,
                 "ins_pixel_values": ins_pixel_values,
                 "ins_atten_mask": ins_atten_mask,
                 "ins_atten_mask_2": ins_atten_mask_2,
                 "pri_input_ids": pri_ids,
                 "pri_pixel_values": pri_pixel_values,
                 "pri_mask": pri_mask,
                 "crop_top_left": crop_top_left,
                 "original_size": original_size}
    else:
        batch = {"ins_input_ids": ins_input_ids,
                 "ins_input_ids_2": ins_input_ids_2,
                 "ins_pixel_values": ins_pixel_values,
                 "ins_atten_mask": ins_atten_mask,
                 "ins_atten_mask_2": ins_atten_mask_2,
                 "mask": ins_mask.unsqueeze(1),
                 "ori_pixel_values": ori_instance_image,
                 "crop_top_left": crop_top_left,
                 "original_size": original_size}

    return batch

def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    for modifier_token in args.modifier_token:
        learned_embeds_dict = {modifier_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

def save_new_embed(text_encoder_1, text_encoder_2, modifier_token_id, modifier_token_id_2, accelerator, args, output_dir, safe_serialization=True):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds_1 = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight[min(modifier_token_id) : max(modifier_token_id) + 1]
    learned_embeds_2 = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight[min(modifier_token_id_2) : max(modifier_token_id_2) + 1]

    text_encoder_1_name, text_encoder_2_name = "text_encoder", "text_encoder_2"
    parent_path_list = [f"{output_dir}/{text_encoder_1_name}", f"{output_dir}/{text_encoder_2_name}"]
    for path in parent_path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    for x, y in zip(modifier_token_id, args.modifier_token):
        
        learned_embeds_dict_1 = {}
        learned_embeds_dict_1[y] = learned_embeds_1[x]
        filename_1 = f"{output_dir}/{text_encoder_1_name}/{y}.bin"

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict_1, filename_1, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict_1, filename_1)
            
    for x, y in zip(modifier_token_id_2, args.modifier_token):
        learned_embeds_dict_2 = {}
        learned_embeds_dict_2[y] = learned_embeds_2[x]
        filename_2 = f"{output_dir}/{text_encoder_2_name}/{y}.bin"

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict_2, filename_2, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict_2, filename_2)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Custom Diffusion training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_clip_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained clip models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )

    parser.add_argument(
        "--clip_image_encoder",
        action="store_true",
        help="Whether to use clip to encoder image.",
    )
    parser.add_argument(
        "--gussmooth",
        action="store_true",
        help="Whether to use gussmooth.",
    )
    parser.add_argument(
        "--local_attention",
        action="store_true",
        help="Whether to use local attention loss.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--train_k",
        action="store_true",
        default=False,
        help="Whether to train the k layer of the cross attention.",
    )
    parser.add_argument(
        "--train_v",
        action="store_true",
        default=False,
        help="Whether to train the v layer of the cross attention.",
    )
    parser.add_argument(
        "--train_q",
        action="store_true",
        default=False,
        help="Whether to train the q layer of the cross attention.",
    )
    parser.add_argument(
        "--train_out",
        action="store_true",
        default=False,
        help="Whether to train the out layer of the cross attention.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--merge_subjects", action="store_true", help="Whether or not to merge multiple subjects."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--merged_class_token",
        type=str,
        default=None,
        help="A token to use as a merged_class for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "Wether to train a DoRA as proposed in- DoRA: Weight-Decomposed Low-Rank Adaptation https://arxiv.org/abs/2402.09353. "
            "Note: to use DoRA you need to install peft from main, `pip install git+https://github.com/huggingface/peft.git`"
        ),
    )

    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]
            

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )
        
        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0].to(text_encoder.device) #[1, 77, 768]
        prompt_embeds = prompt_embeds[-1][-2].to(text_encoder.device)  #[1, 77, 768]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        #print("prompt_embeds.view.shape", prompt_embeds.shape)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) #[1, 77, 2048]
    
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1) #[1, 1280]
    
    return prompt_embeds, pooled_prompt_embeds


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor

    # There might have slightly performance improvement
    # by changing model_input.cpu() to accelerator.gather(model_input)
    return {"model_input": model_input.cpu()}


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    # If passed along, set the training seed now.

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        #datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        #datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("CasualT2I", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept["class_data_dir"])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if args.real_prior:
                assert (
                        class_images_dir / "images"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                        len(list((class_images_dir / "images").iterdir())) == args.num_class_images
                ), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                        class_images_dir / "caption.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                        class_images_dir / "images.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
                concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
                args.concepts_list[i] = concept
                accelerator.wait_for_everyone()
            else:
                cur_class_images = len(list(class_images_dir.iterdir()))

                if cur_class_images < args.num_class_images:
                    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                    if args.prior_generation_precision == "fp32":
                        torch_dtype = torch.float32
                    elif args.prior_generation_precision == "fp16":
                        torch_dtype = torch.float16
                    elif args.prior_generation_precision == "bf16":
                        torch_dtype = torch.bfloat16
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = args.num_class_images - cur_class_images
                    logger.info(f"Number of class images to sample: {num_new_images}.")

                    sample_dataset = PromptDataset(args.class_prompt, num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                    sample_dataloader = accelerator.prepare(sample_dataloader)
                    pipeline.to(accelerator.device)

                    for example in tqdm(
                            sample_dataloader,
                            desc="Generating class images",
                            disable=not accelerator.is_local_main_process,
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = (
                                    class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            )
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    if args.clip_image_encoder:
        clip_trans = transforms.Resize( (224, 224), interpolation=transforms.InterpolationMode.BILINEAR )
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained=args.pretrained_clip_path) #pretrained='laion2b_s32b_b79k'

    # Load the tokenizer CLIPTokenizer
    '''
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer2",
        revision=args.revision,
        use_fast=False,
    )  
    '''
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2"
    )  
   
    '''
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    '''

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
       args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )

    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id,  modifier_token_id_2 = [], []
    initializer_token_id, initializer_token_id_2 = [], []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer_one.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer_one already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )
            
            num_added_tokens_2 = tokenizer_two.add_tokens(modifier_token)
            if num_added_tokens_2 == 0:
                raise ValueError(
                    f"The tokenizer_two already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer_one.encode(initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer_one.convert_tokens_to_ids(modifier_token))

            token_ids_2 = tokenizer_two.encode(initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids_2) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id_2.append(token_ids_2[0])
            modifier_token_id_2.append(tokenizer_two.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder_one.resize_token_embeddings(len(tokenizer_one))
        text_encoder_two.resize_token_embeddings(len(tokenizer_two))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder_one.get_input_embeddings().weight.data
        token_embeds_2 = text_encoder_two.get_input_embeddings().weight.data

        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]
        
        for x, y in zip(modifier_token_id_2, initializer_token_id_2):
            token_embeds_2[x] = token_embeds_2[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder_one.text_model.encoder.parameters(),
            text_encoder_one.text_model.final_layer_norm.parameters(),
            text_encoder_one.text_model.embeddings.position_embedding.parameters(),
            text_encoder_two.text_model.encoder.parameters(),
            text_encoder_two.text_model.final_layer_norm.parameters(),
            text_encoder_two.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
    ########################################################

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    if args.modifier_token is None:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

    if args.clip_image_encoder:
        clip_model.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if args.modifier_token is not None:#accelerator.mixed_precision != "fp16" and 
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)


    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    attention_class = (
        #CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else 
        CustomDiffusionAttnProcessor
    ) 
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            #attention_class = CustomDiffusionXFormersAttnProcessor
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    train_k = args.train_k
    train_v = args.train_v
    train_q = args.train_q
    train_out = args.train_out
    custom_diffusion_attn_procs = {}

    
    #atten_map 
    controller = AttentionStore(LOW_RESOURCE=False)
    if args.merge_subjects:
        controller.num_att_layers = 32 #16 or 24 or 32 
    else:
        controller.num_att_layers = 16
    
    st = unet.state_dict()
    

    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]

        weights = {}
        if train_k:
            weights["to_k_custom_diffusion.weight"] = st[layer_name + ".to_k.weight"]
        if train_v:
            weights["to_v_custom_diffusion.weight"] = st[layer_name + ".to_v.weight"]
        if train_q:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
        if train_out:
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
    
    
        #atten_map
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                train_k=train_k,
                train_v=train_v,
                train_q=train_q,
                train_out=train_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                controller=controller,
                place_in_unet=name.split("_")[0],
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_k=False,
                train_v=False,
                train_q=False,
                train_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                controller=controller,
                place_in_unet=name.split("_")[0],
            )
    del st
    
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
    accelerator.register_for_checkpointing(custom_diffusion_layers)

    #############################################################################
    #register CustomModelLoader as model saving hook
    loader = CustomModelLoader(unet=unet)
    #############################################################################

    def get_lora_config(rank, use_dora, target_modules):
        base_config = {
            "r": rank,
            "lora_alpha": rank,
            "init_lora_weights": "gaussian",
            "target_modules": target_modules,
        }
        if use_dora:
            if is_peft_version("<", "0.9.0"):
                raise ValueError(
                    "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                )
            else:
                base_config["use_dora"] = True

        return LoraConfig(**base_config)
    
     # now we will add new LoRA weights to the attention layers
    unet_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    unet_lora_config = get_lora_config(rank=args.rank, use_dora=args.use_dora, target_modules=unet_target_modules)
    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.modifier_token:
        text_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        text_lora_config = get_lora_config(rank=args.rank, use_dora=args.use_dora, target_modules=text_target_modules)
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    
    
    # Optimizer creation
    if args.clip_image_encoder:
        num_subject = len(args.concepts_list)
        maskmlp = [MaskMLP(1024, 1024, 1024, use_residual=True).to(accelerator.device) for _ in range(num_subject)]
        mergeMaskmlp = MergeMaskMLP(num_subject, 1024, 1024, 1024, use_residual=True).to(accelerator.device)
        map_filter = Filter(num_subject, 1024, 1024).to(accelerator.device)

        if args.modifier_token:
            if args.merge_subjects:
                params_to_optimize = (itertools.chain(text_encoder_one.get_input_embeddings().parameters(), text_encoder_two.get_input_embeddings().parameters(), custom_diffusion_layers.parameters(),
                                    mergeMaskmlp.parameters(), map_filter.parameters()))  #[x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])]
            else:
                params_to_optimize = (
                    itertools.chain(text_encoder_one.get_input_embeddings().parameters(), text_encoder_two.get_input_embeddings().parameters(), custom_diffusion_layers.parameters(),
                                    maskmlp[0].parameters()))  #
                # clip_model.parameters()
        else:
            if args.merge_subjects:
                params_to_optimize = (
                    itertools.chain(custom_diffusion_layers.parameters(), mergeMaskmlp.parameters(), map_filter.parameters()))  #
            else:
                params_to_optimize = (
                    itertools.chain(custom_diffusion_layers.parameters(), maskmlp[0].parameters()))  #

    else:
        if args.modifier_token:
            params_to_optimize = (
                itertools.chain(text_encoder_one.get_input_embeddings().parameters(), text_encoder_two.get_input_embeddings().parameters(), custom_diffusion_layers.parameters()))
        else:
            params_to_optimize = (itertools.chain(custom_diffusion_layers.parameters()))
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )# params_to_optimize

    # Dataset and DataLoaders creation:
    if args.merge_subjects:

        train_dataset = Merge_MultiSubjectDataset(
            concepts_list=args.concepts_list,
            tokenizer_1=tokenizer_one,
            tokenizer_2=tokenizer_two,
            with_prior_preservation=args.with_prior_preservation,
            size=args.resolution,
            mask_size=vae.encode(
                torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
            ).latent_dist.sample().size()[-1],
            center_crop=args.center_crop,
            num_class_images=args.num_class_images,
            hflip=args.hflip,
            aug=not args.noaug,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: merge_collate_fn(examples, args.with_prior_preservation, args.concepts_list),
            num_workers=args.dataloader_num_workers,
        )
    else:
        train_dataset = MultiSubjectDataset(
            concepts_list=args.concepts_list,
            tokenizer_1=tokenizer_one,
            tokenizer_2=tokenizer_two,
            with_prior_preservation=args.with_prior_preservation,
            size=args.resolution,
            mask_size=vae.encode(
                torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
            ).latent_dist.sample().size()[-1],
            center_crop=args.center_crop,
            num_class_images=args.num_class_images,
            hflip=args.hflip,
            aug=not args.noaug,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation, args.concepts_list),
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.

    if args.modifier_token is not None:
        if args.clip_image_encoder:
            custom_diffusion_layers, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler, clip_model, maskmlp, mergeMaskmlp, map_filter = accelerator.prepare(
                custom_diffusion_layers, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler, clip_model, maskmlp, mergeMaskmlp, map_filter
            )
        else:
            custom_diffusion_layers, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                custom_diffusion_layers, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
            )
    else:
        if args.clip_image_encoder:
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler, clip_model, maskmlp, mergeMaskmlp, map_filter = accelerator.prepare(
                custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler, clip_model, maskmlp, mergeMaskmlp, map_filter
            )
        else:
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    from_where = ["up", "down", "mid"]
    res_list = [32]
   

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    cosine_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def resize_cross_attention_map(cross_attention_map, token_index,  re_size):
        image = cross_attention_map[:, :, token_index] #counting from 0, the position of <new1> token locates at 4, should be adjusted according to the text prompt
        image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0), size=re_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)#size=(256, 256)
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        return image
        
    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    def single_subject_train_step(vae, text_encoder_1, text_encoder_2, unet, batch, class_name, maskmlp):

        if args.clip_image_encoder:
            ori_pixel_values = batch["ori_pixel_values"]
            with torch.no_grad():
                img_state = clip_model.encode_image(clip_trans(ori_pixel_values)).unsqueeze(1)
            # Predict the noise residual
            mask_clip_image_features, cf_mask_clip_image_features = maskmlp(img_state)
            
            mask_clip_image_features_resize, cf_mask_clip_image_features_resize = [], []
            for i in range(len(mask_clip_image_features)):
                mask_clip_image_features_resize.append(torch.cat(mask_clip_image_features[i], mask_clip_image_features[i], dim=-1))
                cf_mask_clip_image_features_resize.append(torch.cat(cf_mask_clip_image_features[i], cf_mask_clip_image_features[i], dim=-1))

            
        # Convert images to latent space
        latents = vae.encode(batch["ins_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)        

        elems_to_repeat_text_embeds = 1
        # time ids
        add_time_ids = torch.cat(
            [
                compute_time_ids(original_size=s, crops_coords_top_left=c)
                for s, c in zip(batch["original_size"], batch["crop_top_left"])
            ]
        )

        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[text_encoder_1, text_encoder_2],
            tokenizers=None,
            prompt=None,
            text_input_ids_list=[batch["ins_input_ids"], batch["ins_input_ids_2"]]
        )
        prompt_embeds = prompt_embeds.to(accelerator.device)
        unet_added_conditions.update(
            {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
        )
        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1) #[1, 77, 2048]
                    
        # Predict the noise residual
        model_pred = unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds_input + cf_mask_clip_image_features_resize,
                        added_cond_kwargs=unet_added_conditions
                    ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
        mask = batch["mask"]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

        total_loss = loss 

        if args.clip_image_encoder:
            mean_encoder_hidden_states = prompt_embeds_input.mean(dim=1)
            clip_loss = cosine_loss(mean_encoder_hidden_states, cf_mask_clip_image_features_resize.squeeze(0)).mean() / cosine_loss(mean_encoder_hidden_states, mask_clip_image_features_resize.squeeze(0)).mean()
            total_loss +=  0.001 * clip_loss
            return total_loss, mask_clip_image_features.squeeze(0)
        else:

            return total_loss, mask_clip_image_features.squeeze(0)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.modifier_token is not None:
            text_encoder_one.train()
            text_encoder_two.train()
        for step, batch in enumerate(train_dataloader):
            if args.merge_subjects:
                concepts_list = []
                for i, concept in enumerate(args.concepts_list):
                    concepts_list.append(concept["class_prompt"])
                with accelerator.accumulate(unet), accelerator.accumulate(mergeMaskmlp), accelerator.accumulate(text_encoder_one), accelerator.accumulate(text_encoder_two), accelerator.accumulate(map_filter):
                    if args.clip_image_encoder:
                        mergeMaskmlp.train()
                        map_filter.train()
                        mergeMaskmlp = mergeMaskmlp.to(accelerator.device)
                        map_filter = map_filter.to(accelerator.device)
                        ori_pixel_values = batch["ori_pixel_values"]
                        with torch.no_grad():
                            img_state = clip_model.encode_image(clip_trans(ori_pixel_values)).unsqueeze(1)
                            
                        single_mask_image_features, cf_single_mask_image_features, merge_image_feature, cf_merge_image_feature = mergeMaskmlp(img_state)#(1, 1, 1024)
                        
                        single_mask_image_feature_resize, cf_single_mask_image_feature_resize = [], []
                        for i in range(len(cf_single_mask_image_features)):
                            single_mask_image_feature_resize.append(torch.cat([single_mask_image_features[i], single_mask_image_features[i]], dim = -1))
                            cf_single_mask_image_feature_resize.append(torch.cat([cf_single_mask_image_features[i], cf_single_mask_image_features[i]], dim = -1))

                        merge_image_feature = torch.cat([merge_image_feature, merge_image_feature], dim = -1)
                        cf_merge_image_feature = torch.cat([cf_merge_image_feature, cf_merge_image_feature], dim = -1)

                    # Convert images to latent space
                    latents = vae.encode(batch["ins_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    merge_subject_class_id_1 = tokenize_prompt(tokenizer_one, args.merged_class_token).to(accelerator.device) #[1, 77]
                    merge_subject_class_id_2 = tokenize_prompt(tokenizer_two, args.merged_class_token).to(accelerator.device)

                    elems_to_repeat_text_embeds = 1
                    
                    # time ids
                    add_time_ids = torch.cat(
                        [
                            compute_time_ids(original_size=s, crops_coords_top_left=c)
                            for s, c in zip(batch["original_size"], batch["crop_top_left"])
                        ]
                    )

                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[merge_subject_class_id_1, merge_subject_class_id_2]
                    )
                    prompt_embeds = prompt_embeds.to(accelerator.device)

       
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1) #[1, 77, 2048]
                    
                    # Get the text embedding for conditioning
                    subject_model_pred, cf_subject_model_pred, subject_prompt_embeds = [], [], []
                    
                    for i in range(len(concepts_list)):
                        
                        subject_text_inputs_1 = tokenize_prompt(tokenizer_one, concepts_list[i]).to(accelerator.device)
                        subject_text_inputs_2 = tokenize_prompt(tokenizer_two, concepts_list[i]).to(accelerator.device)

                        subject_prompt_embed, subject_pooled_prompt_embed = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[subject_text_inputs_1, subject_text_inputs_2]
                        )
                        subject_prompt_embed = subject_prompt_embed.to(accelerator.device)

                        subject_prompt_embeds_input = subject_prompt_embed.repeat(elems_to_repeat_text_embeds, 1, 1)
                        subject_prompt_embeds.append(subject_prompt_embeds_input)

                    #Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions
                    ).sample

                    merge_image_pred = unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds_input + cf_merge_image_feature,
                        added_cond_kwargs=unet_added_conditions
                    ).sample
                    
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                    mask = batch["mask"]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    merge_loss = F.mse_loss(merge_image_pred.float(), target.float(), reduction="mean")
                    merge_loss = ((merge_loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    total_loss = merge_loss

                    if args.clip_image_encoder:
                        clip_loss = []
                        prompt_embeds_input = prompt_embeds_input.mean(dim=1)
                        for i in range(len(concepts_list)):
                            subject_prompt_embeds[i] = subject_prompt_embeds[i].mean(dim=1)
                            clip_loss.append(cosine_loss(subject_prompt_embeds[i], cf_single_mask_image_feature_resize[i].squeeze(0)).mean() / cosine_loss(subject_prompt_embeds[i], single_mask_image_feature_resize[i].squeeze(0)).mean())
                            
                        merge_clip_loss = cosine_loss(prompt_embeds_input, cf_merge_image_feature.squeeze(0)).mean() / cosine_loss(prompt_embeds_input, merge_image_feature.squeeze(0)).mean()
                        total_loss += 0.001 * (sum(clip_loss) + merge_clip_loss)#0.001
                    
                    #atten_map
                    if args.local_attention:
                        # add attention loss to loss
                        attention_loss_l = [] 
                        token_index_l = [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]]#, [9, 10, 11]

                        cross_attention_maps_l = [[] for _ in range(len(concepts_list))]
                        cross_attention_maps_res_l = [[] for _ in range(len(concepts_list))]
                        #print("controller=", controller)
                        prompts = [merge_subject_class_id_1, merge_subject_class_id_2] # it should be the batch oftext prompts, but here we use batch["input_ids"] since the function only calculate the length of the batch
                        for i in range(bsz): # 1 batch
                            for res in res_list:
                                cross_attention_map = aggregate_current_attention(prompts=prompts, #get the averaged cross attention map from different layers (in same size) of the ith text in batch
                                                                                attention_store=controller, 
                                                                                res=res, 
                                                                                from_where=from_where,
                                                                                is_cross=True,
                                                                                select=i)
                                num_tokens = merge_subject_class_id_1[i].shape[0]#include the <sot> and <eot> tokens
                                for j in range(len(concepts_list)):
                                    token_index = token_index_l[j]
                                    #cross attention image corresponding to the specific token
                                    #image = resize_cross_attention_map(cross_attention_map, token_index, (32, 32))
                                    image = sum([resize_cross_attention_map(cross_attention_map, token_index[k], (args.resolution, args.resolution)) for k in range(len(token_index))] ) / len(token_index)
                                    cross_attention_maps_res_l[j].append(image) #averaged cross attention map of size <res> x <res>
                            
                        for j in range(len(concepts_list)):
                            cross_attention_maps_l[j].append(torch.stack(cross_attention_maps_res_l[j]).mean(dim=0))
                            cross_attention_maps_res_l[j] = []
                        
                        max_indices_list, cross_attention_maps = [], []
                        
                        for j in range(len(concepts_list)):
                            cross_attention_maps_l[j] = torch.stack(cross_attention_maps_l[j]).to("cuda")#(1, 1024, 1024)unsqueeze(1)
                            #print("cross_attention_maps_l[j].shape:", cross_attention_maps_l[j].shape)
                            
                            if global_step % 100 == 0:
                                show_attention_map_during_training(cross_attention_map = cross_attention_maps_l[j], obj=j+1, out_path = args.output_dir, global_step= global_step)

                            cross_attention_maps_l[j] = cross_attention_maps_l[j].expand(1, 3, -1, -1)

                            with torch.no_grad():
                                cur_cross_attention_map = clip_model.encode_image(clip_trans(cross_attention_maps_l[j]).to(torch.float)).unsqueeze(1)
                            cur_cross_attention_map = cur_cross_attention_map.view(1, 32, 32)
                            
                            
                            #print("single_mask_image_features[j].shape:", single_mask_image_features[j].shape)
                            single_mask_image_features[j] = single_mask_image_features[j].view(1, 32, 32)
                            cur_atten_loss = torch.nn.MSELoss(reduction='none')(cur_cross_attention_map, single_mask_image_features[j])
                            cur_atten_loss = cur_atten_loss.reshape(1, 1, 1024)
                            cur_atten_loss = map_filter(cur_atten_loss, j)
                            attention_loss_l.append(cur_atten_loss)

                        attention_loss = sum(attention_loss_l) / len(concepts_list)
                        
                        total_loss += attention_loss
                        
                    print("total_loss = ", total_loss)

                    #clean controller after each sampling or training step
                    controller.cur_step = 0
                    controller.cur_att_layer = 0
                    controller.attention_store = controller.get_empty_store()
                    
                    accelerator.backward(total_loss, retain_graph=True)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if args.modifier_token is not None:
                        if accelerator.num_processes > 1:
                            grads_text_encoder_1 = text_encoder_one.module.get_input_embeddings().weight.grad
                            grads_text_encoder_2 = text_encoder_two.module.get_input_embeddings().weight.grad
                        else:
                            grads_text_encoder_1 = text_encoder_one.get_input_embeddings().weight.grad
                            grads_text_encoder_2 = text_encoder_two.get_input_embeddings().weight.grad

                        # Get the index for tokens that we want to zero the grads for
                        index_grads_to_zero_1 = torch.arange(len(tokenizer_one)) != modifier_token_id[0]
                        index_grads_to_zero_2 = torch.arange(len(tokenizer_two)) != modifier_token_id_2[0]

                        for i in range(len(modifier_token_id[1:])):
                            index_grads_to_zero_1 = index_grads_to_zero_1 & (
                                torch.arange(len(tokenizer_one)) != modifier_token_id[i]
                            )
                        for i in range(len(modifier_token_id_2[1:])):
                            index_grads_to_zero_2 = index_grads_to_zero_2 & (
                                torch.arange(len(tokenizer_two)) != modifier_token_id_2[i]
                            )
                        if grads_text_encoder_1:
                            grads_text_encoder_1.data[index_grads_to_zero_1, :] = grads_text_encoder_1.data[
                                index_grads_to_zero_1, :
                            ].fill_(0)
                        if grads_text_encoder_2:
                            grads_text_encoder_2.data[index_grads_to_zero_2, :] = grads_text_encoder_2.data[
                                index_grads_to_zero_2, :
                            ].fill_(0)

                    if accelerator.sync_gradients:
                        if args.clip_image_encoder:
                                params_to_clip = (
                                    itertools.chain(text_encoder_one.parameters(), text_encoder_two.parameters(), custom_diffusion_layers.parameters(), mergeMaskmlp.parameters(), map_filter.parameters())
                                    if args.modifier_token is not None
                                    else itertools.chain(custom_diffusion_layers.parameters(), mergeMaskmlp.parameters(), map_filter.parameters())
                                )
                        else:
                                params_to_clip = (
                                    itertools.chain(text_encoder_one.parameters(), text_encoder_two.parameters(), custom_diffusion_layers.parameters())
                                    if args.modifier_token is not None
                                    else custom_diffusion_layers.parameters()
                                )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            else:
                #single subject
                with accelerator.accumulate(unet), accelerator.accumulate(text_encoder_one), accelerator.accumulate(text_encoder_two), accelerator.accumulate(maskmlp[0]):
                    # todo: single subject training step
                    print("single subject customized training")
                    num_subject = len(args.concepts_list)
                    loss_list = []
                    mask_clip_features_list = []
                  
                    for i, concept in enumerate(args.concepts_list):
                        maskmlp[i].train()
                        current_loss, mask_clip_features = single_subject_train_step(vae, text_encoder_one, text_encoder_two, unet, batch[concept["class_prompt"]],
                                                    concept["class_prompt"], maskmlp[i])
                    
                        if args.local_attention:
                            # add attention loss to loss
                            cross_attention_maps_l = []
                            cross_attention_maps_res_l = []
                            token_index = [1, 2, 3]
                            prompts = batch[concept["class_prompt"]]["ins_input_ids"]# it should be the batch oftext prompts, but here we use batch["input_ids"] since the function only calculate the length of the batch
                            for i in range(args.train_batch_size): # 1 batch
                                for res in res_list:
                                    
                                    cross_attention_map = aggregate_current_attention(prompts=prompts, 
                                                                                    attention_store=controller, 
                                                                                    res=res, 
                                                                                    from_where=from_where,
                                                                                    is_cross=True,
                                                                                    select=i)
                                    
                                    num_tokens = batch[concept["class_prompt"]]["ins_input_ids"][i].shape[0]#include the <sot> and <eot> tokens
                                    image = sum([resize_cross_attention_map(cross_attention_map, token_index[k], (args.reslution, args.reslution)) for k in range(len(token_index))] ) / len(token_index)
                                    cross_attention_maps_res_l.append(image) #averaged cross attention map of size <res> x <res>

                                cross_attention_maps_l.append(torch.stack(cross_attention_maps_res_l).mean(dim=0))
                                cross_attention_maps_res_l = []
                                
                            cross_attention_maps_l = torch.stack(cross_attention_maps_l).to("cuda")
                            cross_attention_maps_l = cross_attention_maps_l.expand(1, 3, -1, -1)
                            
                            with torch.no_grad():
                                cur_cross_attention_map = clip_model.encode_image(clip_trans(cross_attention_maps_l)).unsqueeze(1)

                            cur_cross_attention_map = cur_cross_attention_map.view(1, 32, 32)
                            mask_clip_features = mask_clip_features.view(1, 32, 32)
                            attention_loss = torch.nn.functional.mse_loss(cur_cross_attention_map, mask_clip_features, reduction="mean")
                            
                            print("current_loss = ", current_loss)
                            print("attention_loss = ", attention_loss)
                            current_loss += 0.3 * attention_loss
                            loss_list.append(current_loss)

                        #clean controller after each sampling or training step
                        controller.cur_step = 0
                        controller.cur_att_layer = 0
                        controller.attention_store = controller.get_empty_store()
                    total_loss = sum(loss_list)
                    accelerator.backward(total_loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if args.modifier_token is not None:
                        if accelerator.num_processes > 1:
                            grads_text_encoder_1 = text_encoder_one.module.get_input_embeddings().weight.grad
                            grads_text_encoder_2 = text_encoder_two.module.get_input_embeddings().weight.grad
                        else:
                            grads_text_encoder_1 = text_encoder_one.get_input_embeddings().weight.grad
                            grads_text_encoder_2 = text_encoder_two.get_input_embeddings().weight.grad

                        # Get the index for tokens that we want to zero the grads for
                        index_grads_to_zero_1 = torch.arange(len(tokenizer_one)) != modifier_token_id[0]
                        index_grads_to_zero_2 = torch.arange(len(tokenizer_two)) != modifier_token_id_2[0]

                        for i in range(len(modifier_token_id[1:])):
                            index_grads_to_zero_1 = index_grads_to_zero_1 & (
                                    torch.arange(len(tokenizer_one)) != modifier_token_id[i]
                            )

                        for i in range(len(modifier_token_id_2[1:])): 
                            index_grads_to_zero_2 = index_grads_to_zero_2 & (
                                    torch.arange(len(tokenizer_two)) != modifier_token_id_2[i]
                            )
                        if grads_text_encoder_1:
                            grads_text_encoder_1.data[index_grads_to_zero_1, :] = grads_text_encoder_1.data[
                                                                            index_grads_to_zero_1, :
                                                                            ].fill_(0)
                        if grads_text_encoder_2:
                            grads_text_encoder_2.data[index_grads_to_zero_2, :] = grads_text_encoder_2.data[
                                                                            index_grads_to_zero_2, :
                                                                            ].fill_(0)
                    
                    if accelerator.sync_gradients:
                        if args.clip_image_encoder:
                            params_to_clip = (
                                itertools.chain(text_encoder_one.parameters(), text_encoder_two.parameters(), custom_diffusion_layers.parameters(), maskmlp[0].parameters())
                                if args.modifier_token is not None
                                else itertools.chain(custom_diffusion_layers.parameters(), maskmlp[0].parameters())
                            )
                        else:
                            params_to_clip = (
                                itertools.chain(text_encoder_one.parameters(), text_encoder_two.parameters(), custom_diffusion_layers.parameters())
                                if args.modifier_token is not None
                                else custom_diffusion_layers.parameters()
                            )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step > 449 and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:

                        save_path = os.path.join(args.output_dir, "wkwv", f"checkpoint-{global_step}") #wkwv means we finetune the W_q and W_v matrix in the cross attention layer
                        unet = unet.to(torch.float32)
                        loader.save_attn_procs(save_path, safe_serialization=not args.no_safe_serialization)
                        
                        '''
                        if args.modifier_token is not None:
                            save_new_embed(
                                text_encoder_one,
                                text_encoder_two,
                                modifier_token_id,
                                modifier_token_id_2,
                                accelerator,
                                args,
                                save_path,
                                safe_serialization=not args.no_safe_serialization,
                            )
                        '''   
                        unet = unwrap_model(unet)
                        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
                        if args.modifier_token is not None:
                            text_encoder_one = unwrap_model(text_encoder_one)
                            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
                            )
                            text_encoder_two = unwrap_model(text_encoder_two)
                            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(text_encoder_two.to(torch.float32))
                            )
                        else:
                            text_encoder_lora_layers = None
                            text_encoder_2_lora_layers = None

                        StableDiffusionXLPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_layers,
                            text_encoder_lora_layers=text_encoder_lora_layers,
                            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                        )

                        if args.clip_image_encoder:
                            if args.merge_subjects:
                                torch.save(mergeMaskmlp.state_dict(), os.path.join(save_path, "mergeMaskmlp.pt"))
                                torch.save(map_filter.state_dict(), os.path.join(save_path, "map_filter.pt"))
                            else:
                                for i in range(len(args.concepts_list)):
                                    torch.save(maskmlp[i].state_dict(), os.path.join(save_path, "maskmlp_{}.pt".format(i)))


            if global_step >= args.max_train_steps:
                break


    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
