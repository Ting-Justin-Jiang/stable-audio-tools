import argparse
import json
import importlib.util
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import partial
from enum import Enum

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

torch.load = partial(torch.load, weights_only=False)


class CLAPImplementation(Enum):
    """Available CLAP implementations."""
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

        audio_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        for ext in audio_exts:
            for audio_path in root.rglob(f"*{ext}"):
                info: Dict[str, str] = {"relpath": str(audio_path.relative_to(root))}
                try:
                    metadata = meta.get_custom_metadata(info, None)
                    prompt = ""
                    if metadata and "prompt" in metadata:
                        prompt = metadata["prompt"]
                    if prompt:
                        prompts.append(_truncate_at_commas(prompt, truncate_commas))
                except Exception as e:
                    print(f"Warning: Error processing {audio_path}: {e}")

    if not prompts:
        raise RuntimeError("No prompts found – check dataset configuration.")
    return prompts


class CLAPEvaluator:
    """CLAP evaluator using HuggingFace Transformers implementation."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self._init_transformers_clap(model_path)

    def _init_transformers_clap(self, model_path: Optional[str]) -> None:
        """Initialize HuggingFace Transformers CLAP implementation."""
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        
        model_name = model_path or "laion/larger_clap_music_and_speech"
        print(f"Loading Transformers CLAP model: {model_name}")
        
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = ClapProcessor.from_pretrained(model_name)

    def get_embeddings(self,
                       audio: torch.Tensor,
                       prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get audio and text embeddings (Transformers)."""
        return self._get_transformers_embeddings(audio, prompts)

    def _get_transformers_embeddings(self,
                                     audio: torch.Tensor,
                                     prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings using HuggingFace Transformers CLAP."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Transformers CLAP model not initialized")
        
        with torch.no_grad():
            audio_list = [audio[i].cpu().numpy() for i in range(audio.shape[0])]
            audio_inputs = self.processor(
                audios=audio_list,
                return_tensors="pt",
                sampling_rate=48000
            )

            for key in audio_inputs:
                if isinstance(audio_inputs[key], torch.Tensor):
                    audio_inputs[key] = audio_inputs[key].to(self.device)

            text_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True
            )

            for key in text_inputs:
                if isinstance(text_inputs[key], torch.Tensor):
                    text_inputs[key] = text_inputs[key].to(self.device)

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
                seed=generation_kwargs.get("seed", 42) + i,  # Different seed per sample
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


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for Python, NumPy, and PyTorch. Enable deterministic algorithms."""
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def save_results_json(out_dir: Path,
                      run_name: Optional[str],
                      results: Dict[str, Any],
                      meta: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = ["clap_eval"] + ([run_name] if run_name else [])
    filename = f"{'_'.join(name_parts)}_{timestamp}.json"
    payload = {"results": results, "meta": meta}
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=2))
    return path


def _slugify_filename(text: str, max_len: int = 64) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    text = text.replace(" ", "_")
    slug = "".join(ch for ch in text if ch in allowed)
    return slug[:max_len] or "sample"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generative audio with CLAP (Transformers). Computes fidelity (CLAP score) "
            "and diversity across multiple generations per prompt."
        )
    )

    # Model
    parser.add_argument(
        "--model-config",
        default="/home/tj147/stable-audio-tools/checkpoint/stable-audio-open-small-base/model_config.json",
        help="Path to model configuration JSON",
    )
    parser.add_argument(
        "--model-ckpt",
        default="/home/tj147/stable-audio-tools/checkpoint/stable-audio-open-small-base/model.ckpt",
        help="Path to model checkpoint",
    )

    # CLAP (Transformers)
    parser.add_argument(
        "--clap-model",
        default="laion/larger_clap_music_and_speech",
        help="HuggingFace model id for CLAP",
    )

    # Data / prompts
    parser.add_argument(
        "--dataset-config",
        default="./stable_audio_tools/data/local/dataset_cfg.json",
        help="Path to dataset configuration JSON (audio_dir)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=10,
        help="Maximum number of prompts to evaluate",
    )

    # Generation
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for quality evaluation")
    parser.add_argument("--seconds-total", type=int, default=11, help="Length of generated audio in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--deterministic", type=int, choices=[0, 1], default=1, help="Enable deterministic algorithms")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")

    # Metrics
    parser.add_argument("--evaluate-quality", action="store_true", default=True, help="Compute fidelity (CLAP score)")
    parser.add_argument("--evaluate-diversity", action="store_true", default=True, help="Compute diversity per prompt")
    parser.add_argument("--diversity-samples", type=int, default=10, help="Samples per prompt for diversity evaluation")

    # Output
    parser.add_argument("--out-dir", default="./eval_runs", help="Directory to write results JSON")
    parser.add_argument("--run-name", default="", help="Optional run name to include in filename")
    parser.add_argument("--quality-out-name", default="test", help="Folder name under out-dir to save per-prompt audio")

    args = parser.parse_args()

    # Setup device
    # Seed and device
    set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
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
    clap_evaluator = CLAPEvaluator(
        model_path=args.clap_model,
        device=str(device)
    )

    # Load prompts
    prompts = load_prompts(Path(args.dataset_config))
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts for evaluation")

    # Evaluation
    results: Dict[str, Any] = {}
    
    if args.evaluate_quality:
        print("\n=== QUALITY EVALUATION ===")
        total_samples = 0
        total_similarity = 0.0
        
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        
        # Prepare output directory for per-prompt audio if requested
        quality_audio_dir: Optional[Path] = None
        if args.quality_out_name:
            base_dir = Path(args.out_dir).expanduser().resolve()
            quality_audio_dir = base_dir / args.quality_out_name
            quality_audio_dir.mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating Quality"):
            batch_prompts = prompts[i: i + args.batch_size]
            batch_size = len(batch_prompts)

            batch_audio = []
            for j, prompt in enumerate(batch_prompts):
                conditioning_dict = {"prompt": prompt, "seconds_total": args.seconds_total}
                conditioning = [conditioning_dict]

                with torch.no_grad():
                    audio = generate_diffusion_cond(
                        model=model,
                        steps=args.steps,
                        conditioning=conditioning,
                        sample_size=sample_size,
                        seed=int(args.seed) + i + j,
                        device=str(device),
                    )
                    mono_audio = audio.squeeze(0).mean(dim=0)
                    batch_audio.append(mono_audio)

                    if quality_audio_dir is not None:
                        slug = _slugify_filename(prompt)
                        idx = i + j
                        wav_path = quality_audio_dir / f"{idx:05d}_{slug}.wav"
                        wav = mono_audio.detach().cpu().unsqueeze(0)
                        torchaudio.save(str(wav_path), wav, sample_rate)

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
        diversity_scores: List[float] = []
        
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
        
        average_diversity = float(np.mean(diversity_scores))
        results["diversity_score"] = average_diversity
        print(f"Average Diversity Score: {average_diversity:.4f}")
    
    # Print final results
    print(f"\n=== FINAL RESULTS ===")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    # Save results JSON with datetime
    out_dir = Path(args.out_dir).expanduser().resolve()
    meta: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "model_config_path": str(Path(args.model_config).expanduser().resolve()),
        "model_ckpt_path": str(Path(args.model_ckpt).expanduser().resolve()),
        "clap_model": args.clap_model,
        "sample_rate": model_config.get("sample_rate"),
        "sample_size": model_config.get("sample_size"),
        "seconds_total": int(args.seconds_total),
        "steps": int(args.steps),
        "max_prompts": int(args.max_prompts),
        "num_diversity_samples": int(args.diversity_samples) if args.evaluate_diversity else 0,
        "library_versions": {
            "torch": torch.__version__,
            "torchaudio": getattr(torchaudio, "__version__", "unknown"),
        },
    }
    if args.evaluate_quality and args.quality_out_name:
        meta["quality_audio_dir"] = str(Path(args.out_dir).expanduser().resolve() / args.quality_out_name)
    results_path = save_results_json(out_dir, args.run_name or None, results, meta)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()