import os
import re
import cv2
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


def main(prompt, image_dir, output_dir):
    num_inference_steps = 30
    guidance_scale = 9.0
    controlnet_name = "lllyasviel/sd-controlnet-depth"

    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if re.match(r'canonical_\d+\.png$', f)
    ])

    if not image_files:
        print(f"No canonical_*.png found in {image_dir}")
        return

    for fname in tqdm(image_files, desc="Generating images"):
        image_path = os.path.join(image_dir, fname)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cond_image = Image.fromarray(img)

        result = pipe(
            prompt=prompt,
            image=cond_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        result.save(os.path.join(output_dir, fname))
        print(f"✅ Saved: {os.path.join(output_dir, fname)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using StableDiffusion + ControlNet (depth) on all canonical_*.png images.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to guide generation.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing canonical_*.png images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images.")
    args = parser.parse_args()

    main(args.prompt, args.image_dir, args.output_dir)
