import argparse
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import partial
from enum import Enum

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Configure torch.load to disable weights_only for compatibility
torch.load = partial(torch.load, weights_only=False)


class CLAPImplementation(Enum):
    """Available CLAP implementations."""
    LAION = "laion"
    TRANSFORMERS = "transformers"


def _truncate_at_commas(text: str, n: int) -> str:
    """Return the substring containing the first *n* comma separators.
    n == 0 → no truncation.
    """
    if n <= 0:
        return text.strip()
    parts = text.split(",")
    if len(parts) <= n:
        return text.strip()  # nothing to cut
    return ",".join(parts[:n]).strip()


def load_prompts(cfg_path: Path, truncate_commas: int = 4) -> List[str]:
    """Parse dataset_cfg.json and collect prompts, optionally truncating them."""
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {cfg_path}")
    
    with cfg_path.open() as f:
        cfg = json.load(f)

    prompts: List[str] = []
    for ds in cfg["datasets"]:
        root = Path(ds["path"])
        
        if not root.exists():
            print(f"Warning: Dataset path does not exist: {root}")
            continue

        spec = importlib.util.spec_from_file_location("meta", ds["custom_metadata_module"])
        if spec is None or spec.loader is None:
            print(f"Warning: Could not load metadata module: {ds['custom_metadata_module']}")
            continue
            
        meta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(meta)

        for wav in root.rglob("*.wav"):
            info: Dict[str, str] = {"relpath": str(wav.relative_to(root))}
            try:
                metadata = meta.get_custom_metadata(info, None)
                prompt = ""
                if metadata and "prompt" in metadata:
                    prompt = metadata["prompt"]
                if prompt:
                    prompts.append(_truncate_at_commas(prompt, truncate_commas))
            except Exception as e:
                print(f"Warning: Error processing {wav}: {e}")

    if not prompts:
        raise RuntimeError("No prompts found – check dataset configuration.")
    return prompts


class CLAPEvaluator:
    """Unified CLAP evaluator supporting multiple implementations."""
    
    def __init__(self, 
                 implementation: CLAPImplementation,
                 model_path: Optional[str] = None,
                 device: str = "cuda"):
        self.implementation = implementation
        self.device = device
        self.model = None
        self.processor = None
        
        if implementation == CLAPImplementation.LAION:
            self._init_laion_clap(model_path)
        elif implementation == CLAPImplementation.TRANSFORMERS:
            self._init_transformers_clap(model_path)
        else:
            raise ValueError(f"Unsupported implementation: {implementation}")
    
    def _init_laion_clap(self, model_path: Optional[str]) -> None:
        """Initialize LAION CLAP implementation."""
        try:
            import laion_clap
        except ImportError:
            raise ImportError("laion-clap not installed. Install with: pip install laion-clap")
        
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model = self.model.eval().to(self.device)
        
        if model_path and Path(model_path).exists():
            print(f"Loading LAION CLAP checkpoint from {model_path}")
            clap_state = torch.load(model_path, map_location=self.device)
            self.model.model.load_state_dict(clap_state, strict=False)
        else:
            print("Loading default LAION CLAP checkpoint")
            self.model.load_ckpt()
    
    def _init_transformers_clap(self, model_path: Optional[str]) -> None:
        """Initialize HuggingFace Transformers CLAP implementation."""
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        
        model_name = model_path or "laion/larger_clap_music_and_speech"
        print(f"Loading Transformers CLAP model: {model_name}")
        
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
    
    def get_embeddings(self, 
                      audio: torch.Tensor, 
                      prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get audio and text embeddings using the appropriate implementation."""
        if self.implementation == CLAPImplementation.LAION:
            return self._get_laion_embeddings(audio, prompts)
        elif self.implementation == CLAPImplementation.TRANSFORMERS:
            return self._get_transformers_embeddings(audio, prompts)
        else:
            raise ValueError(f"Unsupported implementation: {self.implementation}")
    
    def _get_laion_embeddings(self, 
                            audio: torch.Tensor, 
                            prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings using LAION CLAP."""
        if self.model is None:
            raise RuntimeError("LAION CLAP model not initialized")
            
        if audio.is_cuda:
            audio = audio.cpu()
        
        with torch.no_grad():
            # LAION CLAP expects audio as numpy for get_audio_embedding_from_data
            audio_np = audio.numpy()
            audio_embedding = self.model.get_audio_embedding_from_data(x=audio_np, use_tensor=True)
            text_embedding = self.model.get_text_embedding(prompts, use_tensor=True)
            
        return audio_embedding.to(self.device), text_embedding.to(self.device)
    
    def _get_transformers_embeddings(self, 
                                   audio: torch.Tensor, 
                                   prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings using HuggingFace Transformers CLAP."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Transformers CLAP model not initialized")
            
        with torch.no_grad():
            # Process audio - need to handle batch of audio
            audio_list = [audio[i].cpu().numpy() for i in range(audio.shape[0])]
            audio_inputs = self.processor(
                audios=audio_list, 
                return_tensors="pt", 
                sampling_rate=48000
            )
            
            # Move audio inputs to device
            for key in audio_inputs:
                if isinstance(audio_inputs[key], torch.Tensor):
                    audio_inputs[key] = audio_inputs[key].to(self.device)
            
            # Process text
            text_inputs = self.processor(
                text=prompts, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move text inputs to device
            for key in text_inputs:
                if isinstance(text_inputs[key], torch.Tensor):
                    text_inputs[key] = text_inputs[key].to(self.device)
            
            # Get embeddings
            audio_embedding = self.model.get_audio_features(**audio_inputs)
            text_embedding = self.model.get_text_features(**text_inputs)
            
        return audio_embedding, text_embedding


def evaluate_clap_quality(evaluator: CLAPEvaluator, 
                         audio: torch.Tensor, 
                         prompts: List[str]) -> float:
    """Evaluate CLAP quality score for audio-text pairs."""
    audio_embedding, text_embedding = evaluator.get_embeddings(audio, prompts)
    
    scores = torch.nn.functional.cosine_similarity(
        audio_embedding, text_embedding, dim=1, eps=1e-6
    )
    avg_score = scores.mean().item()
    print(f"Batch CLAP quality score: {avg_score:.4f}")
    return scores.sum().item()


def evaluate_clap_diversity(evaluator: CLAPEvaluator,
                           prompt: str,
                           model, 
                           model_config: Dict[str, Any],
                           num_samples: int = 5,
                           **generation_kwargs) -> float:
    """Evaluate CLAP diversity by generating multiple samples with same prompt."""
    print(f"Evaluating diversity for prompt: '{prompt[:50]}...'")
    
    # Import here to avoid circular imports
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    # Generate multiple samples with different seeds
    audio_samples = []
    for i in range(num_samples):
        conditioning_dict = {"prompt": prompt, "seconds_total": generation_kwargs.get("seconds_total", 11)}
        conditioning = [conditioning_dict]  # Must be a list of dictionaries
        
        with torch.no_grad():
            audio = generate_diffusion_cond(
                model=model,
                steps=generation_kwargs.get("steps", 8),
                conditioning=conditioning,
                sample_size=sample_size,
                seed=generation_kwargs.get("seed", 42) + i,  # Different seed for each sample
                device=generation_kwargs.get("device", "cuda"),
            )
            audio_samples.append(audio.squeeze(0).mean(dim=0))  # Convert to mono and remove batch dim
    
    # Stack all samples
    audio_batch = torch.stack(audio_samples, dim=0)
    
    # Resample if necessary  
    if sample_rate != 48000:
        audio_batch = torchaudio.functional.resample(audio_batch, sample_rate, 48000)
    
    # Get embeddings for all samples
    prompts = [prompt] * num_samples
    audio_embeddings, _ = evaluator.get_embeddings(audio_batch, prompts)
    
    # Calculate pairwise cosine similarities
    similarities = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            sim = torch.nn.functional.cosine_similarity(
                audio_embeddings[i:i+1], audio_embeddings[j:j+1], dim=1
            ).item()
            similarities.append(sim)
    
    # Average similarity (lower = more diverse)
    avg_similarity = np.mean(similarities) if similarities else 0.0
    diversity_score = 1.0 - avg_similarity  # Convert to diversity score (higher = more diverse)
    
    print(f"Average pairwise similarity: {avg_similarity:.4f}, Diversity score: {diversity_score:.4f}")
    return diversity_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio generation with CLAP scores")
    parser.add_argument("--model-config", default="/home/tj147/stable-audio-tools/checkpoint/stable-audio-open-small-base/model_config.json",
                        help="Path to model configuration file")
    parser.add_argument("--model-ckpt", default="/home/tj147/stable-audio-tools/checkpoint/stable-audio-open-small-base/model.ckpt",
                        help="Path to model checkpoint file")
    parser.add_argument("--clap-implementation", choices=["laion", "transformers"], default="transformers", # laion one has bug
                        help="CLAP implementation to use")
    parser.add_argument("--clap-model", default=None,
                        help="Path to CLAP checkpoint (LAION) or model name (Transformers)")
    parser.add_argument("--dataset-config", default="./stable_audio_tools/data/local/dataset_cfg.json",
                        help="Path to dataset configuration file")
    parser.add_argument("--steps", type=int, default=8, help="Number of diffusion steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--seconds-total", type=int, default=11, help="Length of generated audio in seconds")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--evaluate-quality", action="store_true", default=False,
                        help="Evaluate quality (CLAP score)")
    parser.add_argument("--evaluate-diversity", action="store_true", default=True,
                        help="Evaluate diversity (requires multiple generations)")
    parser.add_argument("--diversity-samples", type=int, default=5,
                        help="Number of samples per prompt for diversity evaluation")
    parser.add_argument("--max-prompts", type=int, default=10,
                        help="Maximum number of prompts to evaluate")
    
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model configuration
    if not Path(args.model_config).exists():
        raise FileNotFoundError(f"Model config file not found: {args.model_config}")

    cfg_dict = json.loads(Path(args.model_config).read_text())
    
    # Load model
    from stable_audio_tools.interface.gradio import load_model
    model, model_config = load_model(
        model_config=cfg_dict,
        model_ckpt_path=args.model_ckpt,
        device=str(device),
        model_half=False,
    )
    
    if model_config is None:
        raise RuntimeError("Failed to load model configuration")

    # Initialize CLAP evaluator
    clap_impl = CLAPImplementation.LAION if args.clap_implementation == "laion" else CLAPImplementation.TRANSFORMERS
    clap_evaluator = CLAPEvaluator(
        implementation=clap_impl,
        model_path=args.clap_model,
        device=str(device)
    )

    # Load prompts
    prompts = load_prompts(Path(args.dataset_config))
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts for evaluation")

    # Evaluation
    results = {}
    
    if args.evaluate_quality:
        print("\n=== QUALITY EVALUATION ===")
        total_samples = 0
        total_similarity = 0.0
        
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating Quality"):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_size = len(batch_prompts)
            
            # Generate audio for each prompt individually
            batch_audio = []
            for prompt in batch_prompts:
                conditioning_dict = {"prompt": prompt, "seconds_total": args.seconds_total}
                conditioning = [conditioning_dict]  # Must be a list of dictionaries
                
                with torch.no_grad():
                    audio = generate_diffusion_cond(
                        model=model,
                        steps=args.steps,
                        conditioning=conditioning,
                        sample_size=sample_size,
                        seed=args.seed + i,
                        device=str(device),
                    )
                    batch_audio.append(audio.squeeze(0).mean(dim=0))  # Convert to mono, remove batch dim
            
            # Stack all audio in batch
            batch_audio = torch.stack(batch_audio, dim=0)
            
            # Resample if necessary
            if sample_rate != 48000:
                batch_audio = torchaudio.functional.resample(batch_audio, sample_rate, 48000)
            
            # Evaluate CLAP scores
            batch_similarity = evaluate_clap_quality(clap_evaluator, batch_audio, batch_prompts)
            total_similarity += batch_similarity
            total_samples += batch_size

        average_clap_score = total_similarity / total_samples
        results["quality_score"] = average_clap_score
        print(f"Average CLAP Quality Score: {average_clap_score:.4f}")
    
    if args.evaluate_diversity:
        print(f"\n=== DIVERSITY EVALUATION ===")
        diversity_scores = []
        
        # Select subset of prompts for diversity evaluation (it's more expensive)
        diversity_prompts = prompts[:min(10, len(prompts))]
        
        for prompt in tqdm(diversity_prompts, desc="Evaluating Diversity"):
            diversity_score = evaluate_clap_diversity(
                evaluator=clap_evaluator,
                prompt=prompt,
                model=model,
                model_config=model_config,
                num_samples=args.diversity_samples,
                steps=args.steps,
                seconds_total=args.seconds_total,
                seed=args.seed,
                device=str(device)
            )
            diversity_scores.append(diversity_score)
        
        average_diversity = np.mean(diversity_scores)
        results["diversity_score"] = average_diversity
        print(f"Average Diversity Score: {average_diversity:.4f}")
    
    # Print final results
    print(f"\n=== FINAL RESULTS ===")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()