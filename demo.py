import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# TODO: add arguments to load trained/finetuned model

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)
set_random_seed(3)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "142 bpm, amen break",
    "seconds_total": 10
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=8,
    cfg_scale=1.0,
    conditioning=conditioning,
    sample_size=sample_size,
    sampler_type="pingpong",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)