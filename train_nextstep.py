import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser

from nextstep.models.modeling_nextstep import NextStep
from nextstep.models.modeling_flux_vae import AutoencoderKL

from nextstep.models.tokenization_nextstep import (
    special_tokens_dict,
    DEFAULT_BOI_TOKEN,
    DEFAULT_EOI_TOKEN,
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
    DEFAULT_IMAGE_AREA_TOKEN,
)

# ⭐️ 2. Define a dataclass for your script-specific arguments
@dataclass
class ScriptArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    vae_name_or_path: str = field(metadata={"help": "Path to pretrained VAE model"})
    tokenizer_name_or_path: str = field(metadata={"help": "Path to pretrained tokenizer"})
    dataset_name: str = field(metadata={"help": "Path to the data_info.json file"})
    image_hw: List[int] = field(
        default_factory=lambda: [256, 256],
        metadata={
            "help": "Image height and width for resizing.",
            "nargs": 2, # Tells the parser to expect 2 values
            "type": int   # Tells the parser to convert each value to an integer
        }
    )

# --- Dataset class (unchanged) ---
class JsonImageDataset(Dataset):
    def __init__(self, json_path, image_hw: Tuple[int, int] = (256, 256), transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_hw),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        root = ".root/BLIP3o-60k-harmon-format"
        item = self.data[idx]
        caption_json_path = os.path.join(root, "cap_folder", item['annotation'])
        with open(caption_json_path, 'r', encoding='utf-8') as f:
            caption_data = json.load(f)
        
        caption = caption_data.get('caption') or caption_data.get('text')
        if caption is None:
            raise KeyError(f"Neither 'caption' nor 'text' key found in {caption_json_path}")
            
        image_path = os.path.join(root, "local_folder", item['image'])
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        
        return {"pixel_values": pixel_values, "caption": caption}

# --- Data Collator (unchanged) ---
@dataclass
class NextStepDataCollator:
    tokenizer: AutoTokenizer
    vae: AutoencoderKL
    patch_size: int

    def __post_init__(self):
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        
        self.boi_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_BOI_TOKEN)
        self.eoi_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_EOI_TOKEN)
        self.placeholder_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)

    def _hw2str(self, h, w):
        return f"<hw:{h},{w}>"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        captions = [f["caption"] for f in features]
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        self.vae.to(pixel_values.device)
        with torch.no_grad():
            posterior = self.vae.encode(pixel_values.to(self.vae.dtype)).latent_dist
            latents = posterior.sample()
        
        latent_hw = (latents.shape[-2] // self.patch_size, latents.shape[-1] // self.patch_size)
        num_image_tokens = latent_hw[0] * latent_hw[1]
        image_ids = [self.boi_id] + [self.placeholder_id] * num_image_tokens + [self.eoi_id]
        image_str = (
            DEFAULT_IMAGE_AREA_TOKEN + 
            self._hw2str(*latent_hw) + 
            self.tokenizer.decode(image_ids)
        )
        full_prompts = [caption + image_str for caption in captions]
        padded_batch = self.tokenizer(
            full_prompts, padding="longest", truncation=True, return_tensors="pt"
        )
        labels = padded_batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        latents = (latents - self.shift_factor) * self.scaling_factor
        latents_mask = torch.ones(len(features), dtype=torch.long)
        return {
            "input_ids": padded_batch["input_ids"],
            "attention_mask": padded_batch["attention_mask"],
            "labels": labels,
            "latents": latents,
            "latents_mask": latents_mask,
        }

# --- Main Training Script ---
def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path, use_fast=True)
    vae = AutoencoderKL.from_pretrained(script_args.vae_name_or_path)
    model = NextStep.from_pretrained(script_args.model_name_or_path)
    
    vae_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_patch_size = model.config.latent_patch_size
    down_factor = vae_factor * latent_patch_size

    image_hw = tuple(script_args.image_hw)
    latent_hw = (image_hw[0] // down_factor, image_hw[1] // down_factor)
    hw_token = f"<hw:{latent_hw[0]},{latent_hw[1]}>"
    print(f"Input image H, W: {image_hw}")
    print(f"Calculated HW token to be added: {hw_token}")

    updated_special_tokens_dict = special_tokens_dict.copy()
    updated_special_tokens_dict["additional_special_tokens"].extend([
        DEFAULT_IMAGE_AREA_TOKEN, hw_token
    ])

    num_added_tokens = tokenizer.add_special_tokens(updated_special_tokens_dict)
    if num_added_tokens > 0:
        print(f"Added {num_added_tokens} special tokens to the tokenizer.")
        model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    
    train_dataset = JsonImageDataset(json_path=script_args.dataset_name, image_hw=image_hw)

    vae.requires_grad_(False)
    vae.eval()
    
    data_collator = NextStepDataCollator(
        tokenizer=tokenizer, 
        vae=vae, 
        patch_size=model.config.latent_patch_size
    )

 
    trainer = Trainer(
        model=model,
        args=training_args, # Pass the parsed training_args here
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training with corrected argument parsing...")
    trainer.train()
    print("Training complete.")

if __name__ == '__main__':
    main()