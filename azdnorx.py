import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import imageio
import os
import math
import gradio as gr

# --- Configuration ---
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
ACCEPTED_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=(torch.float16 if device == "cuda" else torch.float32)
).to(device)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image):
    orig_w, orig_h = image.size
    new_w = math.ceil(orig_w / 64) * 64
    new_h = math.ceil(orig_h / 64) * 64
    if (new_w, new_h) != (orig_w, orig_h):
        image = image.resize((new_w, new_h))
    image_np = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2)
    return tensor.to(device=device,
                     dtype=(torch.float16 if device=="cuda" else torch.float32))

# --- Video Generation ---
def generate_video(image_file, prompt, steps, guidance, fps, decode_size, easter_egg):
    ext = os.path.splitext(image_file.name)[1].lower()
    if ext not in ACCEPTED_EXT:
        raise gr.Error(f"Unsupported file type: {ext}")
    image = Image.open(image_file).convert("RGB")
    img_tensor = preprocess_image(image)

    output = pipe(
        prompt=prompt,
        image=img_tensor,
        num_inference_steps=steps,
        guidance_scale=guidance,
        decode_chunk_size=decode_size,
        generator=torch.manual_seed(42)
    )
    frames = output.frames[0]

    # Easter Egg: dreamy glow + watermark
    if easter_egg:
        enhanced = []
        for i, frame in enumerate(frames):
            img = frame.copy()
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            if i == len(frames) - 1:
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), "@axdoxz", font=font, fill=(255, 100, 180))
            enhanced.append(np.array(img))
        frames = enhanced

    os.makedirs("temp_vid", exist_ok=True)
    mp4_path = os.path.join("temp_vid", "output.mp4")
    imageio.mimsave(mp4_path, frames, fps=fps)
    return mp4_path

# --- Gradio UI ---
with gr.Blocks(title="Stable Video Diffusion Img2Vid") as demo:
    gr.Markdown("# Stable Video Diffusion: Image → Video Demo")
    with gr.Row():
        img_in = gr.File(label="Upload Image (.jpg, .png, etc.)")
        with gr.Column():
            prompt_in = gr.Textbox(label="Prompt", value="A couple romantically taking a walk, looking into the camera")
            steps_in = gr.Slider(5, 100, value=30, step=5, label="Inference Steps")
            guidance_in = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
            fps_in = gr.Slider(1, 30, value=12, step=1, label="FPS")
            decode_in = gr.Slider(1, 16, value=8, step=1, label="Chunk Size")
            egg_toggle = gr.Checkbox(label="Easter Egg Mode: Dreamy Glow + @axdoxz Watermark", value=False)
            gen_btn = gr.Button("Generate Video")
    video_out = gr.Video(label="Output Video")
    gen_btn.click(
        generate_video,
        inputs=[img_in, prompt_in, steps_in, guidance_in, fps_in, decode_in, egg_toggle],
        outputs=[video_out]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
