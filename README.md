# Eleanordoxz.Visualore
“Dockerized Gradio web UI for Stable Video Diffusion Img2Vid with optional dreamy glow .”

# 🎥 Stable Video Diffusion: Img2Vid Web UI

Turn a static image into a short cinematic loop using the power of Stable Video Diffusion — all via a simple web interface built with Gradio and Docker.

---

## 🚀 Features

- Upload any `.jpg` / `.png` image
- Customize prompt, steps, guidance, FPS, chunk size
- Optional **Easter Egg Mode**: dreamy glow + `__eleshmy_` watermark ✨
- Export as `.mp4`
- Docker-ready & GPU-compatible

---

## 🛠 Requirements

- Python 3.9+
- NVIDIA GPU (for local usage)
- Docker (optional)
- Internet (to download the model weights)

---

## 🧪 Running Locally with Docker

```bash
docker build -t stable-vid-img2vid .
docker run --gpus all -p 7860:7860 stable-vid-img2vid
Then open your browser at http://localhost:7860.


