import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
import open_clip
from PIL import Image
from CustomModelLoader import CustomModelLoader

cosine_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

# combine the images of different subjects
def merge_subject_image(num_subject_classes, instance_images):
    instance_images_w_h = [512, 512]
    #instance_images = [0] * num_subject_classes

    for i in range(num_subject_classes):
        instance_image = instance_images[i].resize((256, 512))
        #print("instance_images[i].size", instance_image.size)
        if i == 0:
            instance_images_w_h[0], instance_images_w_h[1] = instance_image.size
            new_instance_image = Image.new("RGB",
                                        (instance_images_w_h[0] * num_subject_classes,
                                        instance_images_w_h[1]))
            new_instance_image.paste(instance_image, (i, 0))

        else:
            new_instance_image.paste(instance_image, (instance_images_w_h[0] * i, 0))
    return new_instance_image


def save_generated_images(all_generated_images, image_output_path, text_prompt):
    for text_prompt, image_list in all_generated_images.items():
        for i, image in enumerate(image_list):
            image.save(os.path.join(image_output_path, f"{text_prompt}_{i}.jpg"))


def main():
    
    cosine_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
    CLIP_PATH="/home/admin/LCY/AIGC/ViT-H-14/open_clip_pytorch_model.bin"
    base_model_id = "/home/admin/LCY/AIGC/SDXL"

    
    model_ckpt = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*dog-v*sunglasses-k*lake-aspect-0.2atten/wkwv/checkpoint-1500"
    text_encoder_1_p = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*dog-v*sunglasses-k*lake-aspect-0.2atten/wkwv/checkpoint-1500/text_encoder"
    text_encoder_2_p = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*dog-v*sunglasses-k*lake-aspect-0.2atten/wkwv/checkpoint-1500/text_encoder_2"
    
    '''
    model_ckpt = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*dog-v*cat-aspect-0.2atten/wkwv/checkpoint-1000"
    text_encoder_1_p = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*dog-v*cat-aspect-0.2atten/wkwv/checkpoint-1000/text_encoder"
    text_encoder_2_p = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*dog-v*cat-aspect-0.2atten/wkwv/checkpoint-1000/text_encoder_2"
    
    
    model_ckpt = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*cat-v*wooden_pot-aspect-0.2atten/wkwv/checkpoint-800"
    text_encoder_1_p = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*cat-v*wooden_pot-aspect-0.2atten/wkwv/checkpoint-800/text_encoder"
    text_encoder_2_p = "/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*cat-v*wooden_pot-aspect-0.2atten/wkwv/checkpoint-800/text_encoder_2"
    '''

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
    #pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
    with torch.no_grad():
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained=CLIP_PATH)
        clip_model = clip_model.to("cuda")

    #register unet for loader
    if "wq" in model_ckpt:
        train_q = True
    else:
        train_q = False
    if "wk" in model_ckpt:
        train_k = True
    else:
        train_k = False
    if "wv" in model_ckpt:
        train_v = True
    else:
        train_v = False
    if "wout" in model_ckpt:
        train_out = True
    else:
        train_out = False
    
    loader = CustomModelLoader(pipe.unet)
    loader.load_attn_procs(model_ckpt, weight_name="pytorch_textual_localization_weights.bin", train_q=train_q, train_k=train_k, train_v=train_v, train_out=train_out)

    pipe.load_textual_inversion(text_encoder_1_p, weight_name="s*.bin")
    pipe.load_textual_inversion(text_encoder_1_p, weight_name="v*.bin")
    #pipe.load_textual_inversion(text_encoder_1_p, weight_name="k*.bin")
    #pipe.load_textual_inversion(text_encoder_1_p, weight_name="h*.bin")


    #pipe.load_textual_inversion(model_ckpt, weight_name="k*.bin")
    #pipe.load_textual_inversion(model_ckpt, weight_name="q*.bin")
    #pipe.load_lora_weights("/home/admin/LCY/AIGC/Textual-Localization-main/outmodel/SDXL/s*dog-v*cat-SDXL-lora-12.18/wkwv/checkpoint-600/pytorch_lora_weights.safetensors")

    random_seed = range(1,6)#torch.manual_seed(1)
    
    
    #text_prompt = "a s* dog and a v* cat swim in the pool"
    #text_prompt = "a s* teddybear with a red hat on its head in the garden with full of flowers"

    #text_prompt = "an ink painting of a s* man wearing a v* sunglasses with the k* lake as the background"
    text_prompt = "a s* dog and a v* sunglasses without any background"
    
    #text_prompt = "a s* cat is on top of a purple rug, and in the real forest"

    text_prompt_list = [text_prompt]

    #out_path = "./SDXL_results/inference_s*cat_v*wooden-pot-apect/1000-30"

    #out_path = "./SDXL_results/inference_s*dog_v*sunglasses_k*lake/1200-100"
    #out_path = "./SDXL_results/inference_s*dog_v*cat-aspect/1000-30"

    out_path = "./SDXL_results/inference_s*dog-v*sunglasses-spect/1500-30"

    #out_path = "./SDXL_results/inference_s*teddybear-v*tortoise_plushy-spect/1000-50"

    all_generated_images = {}
    for text_prompt in tqdm(text_prompt_list, desc='Text Prompt Loop'):
        all_generated_images[text_prompt] = []
        for seed in tqdm(random_seed, desc='Seed Loop'):
            generator = torch.Generator("cuda").manual_seed(seed)
            #images = pipe(prompt=text_prompt, num_images_per_prompt=10, num_inference_steps=100, guidance_scale=7.5, generator = generator).images #generate 10 images once, return a list of PIL images
            images = pipe(prompt=text_prompt, num_images_per_prompt=5, num_inference_steps=30, guidance_scale=7.5).images
            all_generated_images[text_prompt].extend(images)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_generated_images(all_generated_images, out_path, text_prompt)


if __name__ == '__main__':
    main()