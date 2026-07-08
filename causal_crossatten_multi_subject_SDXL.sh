###
 # @Author: Guosy_wxy 1579528809@qq.com
 # @Date: 2024-01-08 08:23:31
 # @LastEditors: Guosy_wxy 1579528809@qq.com
 # @LastEditTime: 2024-09-10 01:04:49
 # @FilePath: /AIGC/custom_diffusion/multi_concept_without_prior.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
export MODEL_NAME="/home/admin/LCY/AIGC/SDXL"
export CLIP_PATH="/home/admin/LCY/AIGC/ViT-H-14/open_clip_pytorch_model.bin"
export OUTPUT_DIR="./outmodel/SDXL/s*cat-v*wooden_pot-aspect-0.2atten"

#--merge_subjects  
#--merged_class_token "a sks1 cat and a sks2 dog"
#--enable_xformers_memory_efficient_attention \
#--merged_class_token "a s* cat and a v* dog" \
#--pretrained_clip_path=$CLIP_PATH  \
#--clip_image_encoder \
#--merge_subjects \
#--merged_class_token "a s* dog and a v* teddybear" \
#--gussmooth
#  --local_attention  \
#--modifier_token="s*+v*"  \
#--enable_xformers_memory_efficient_attention \
#--gradient_checkpointing \
#--mixed_precision="fp16"

CUDA_VISIBLE_DEVICES=1 accelerate launch ./causal_guidance_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --pretrained_clip_path=$CLIP_PATH  \
  --clip_image_encoder \
  --concepts_list=./causal_concept_list.json  \
  --resolution=1024  \
  --train_batch_size=1  \
  --learning_rate=5e-5  \
  --lr_warmup_steps=0  \
  --max_train_steps=1200\
  --scale_lr  \
  --hflip  \
  --no_safe_serialization  \
  --modifier_token="s*+v*"  \
  --initializer_token="s+v"  \
  --local_attention  \
  --merge_subjects  \
  --merged_class_token="a s* cat and a v* wooden pot"  \
  --train_v --train_k \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rows=1 \
  --cols=2
