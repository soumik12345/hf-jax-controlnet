python train_controlnet_flax.py \
    --wandb_entity "ml-colabs" \
    --dataset_name "fusing/fill50k" \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --revision="flax" \
    --caption_column text \
    --conditioning_image_column conditioning_image \
    --image_column image \
    --num_train_epochs 1 \
    --validation_steps 1000 \
    --checkpointing_steps 1000 \
    --output_dir "dump_dir" \
    --validation_prompt \
        "bright golden rod circle with old lace background" \
        "dark magenta circle with cyan background" \
        "orange circle with sienna background" \
        "ghost white circle with medium purple background" \
        "indian red circle with medium violet red background" \
    --validation_image \
        "../data_cache/conditioning_images/0.png" \
        "../data_cache/conditioning_images/1.png" \
        "../data_cache/conditioning_images/2.png" \
        "../data_cache/conditioning_images/3.png" \
        "../data_cache/conditioning_images/4.png"
