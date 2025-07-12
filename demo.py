#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torchaudio
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.interface.gradio import load_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


def seed_all(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def peak_norm_int16(x: torch.Tensor) -> torch.Tensor:
    """Normalize audio to int16 range."""
    x = x / x.abs().max()
    return (x.clamp(-1, 1) * 32767).to(torch.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio from text prompt")
    parser.add_argument("--prompt", default="Lo-Fi, guitar", help="Text prompt for audio generation")
    parser.add_argument("--output", default="output.wav", help="Output audio file path")
    parser.add_argument("--model-repo", default="stabilityai/stable-audio-open-small", help="Pretrained model repository")
    parser.add_argument("--model-config", default="", help="Path to model configuration file (if not using pretrained)")
    parser.add_argument("--model-ckpt", default="", help="Path to model checkpoint file (if not using pretrained)")
    parser.add_argument("--steps", type=int, default=8, help="Number of diffusion steps")
    parser.add_argument("--seconds-total", type=int, default=11, help="Length of generated audio in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_config: Optional[Dict[str, Any]] = None

    if args.model_repo:
        print(f"Loading pretrained model: {args.model_repo}")
        model, model_config = get_pretrained_model(args.model_repo)
        model = model.to(device)
    else:
        if not args.model_config or not args.model_ckpt:
            raise ValueError("Must provide either --model-repo or both --model-config and --model-ckpt")
        
        if not Path(args.model_config).exists():
            raise FileNotFoundError(f"Model config file not found: {args.model_config}")
        
        if not Path(args.model_ckpt).exists():
            raise FileNotFoundError(f"Model checkpoint file not found: {args.model_ckpt}")
        
        print(f"Loading model from config: {args.model_config}")
        cfg_dict = json.loads(Path(args.model_config).read_text())
        model, model_config = load_model(
            model_config=cfg_dict,
            model_ckpt_path=args.model_ckpt,
            device=str(device),  # Convert device to string
            model_half=False
        )

    if model_config is None:
        raise RuntimeError("Failed to load model configuration")
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    print(f"Model loaded - Sample rate: {sample_rate}Hz, Sample size: {sample_size}")

    # Set seed for reproducibility
    seed_all(args.seed)

    # Generate audio
    print(f"Generating audio with prompt: '{args.prompt}'")
    conditioning = {"prompt": args.prompt, "seconds_total": args.seconds_total}
    
    with torch.no_grad():
        audio = generate_diffusion_cond(
                model=model,
                steps=args.steps,
                conditioning=[conditioning],
                sample_size=sample_size,
                seed=args.seed,
                device=str(device),  # Convert device to string
            ).squeeze(0).cpu()  # Remove batch dimension and move to CPU

    # Save audio
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(str(output_path), peak_norm_int16(audio), sample_rate)
    print(f"Audio saved to: {output_path}")


if __name__ == "__main__":
    main()
