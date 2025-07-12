"""
Dataset configuration generator for stable-audio-tools.

This script generates dataset configuration files and metadata extraction modules
for training stable-audio-tools models from local audio files.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Set, Dict, Any


def _clean_text(text: str) -> str:
    """Clean text by replacing underscores and hyphens with spaces."""
    return text.replace("_", " ").replace("-", " ").strip()


def _strip_keyword(text: str, keyword: str) -> str:
    """Remove keyword from text using case-insensitive regex."""
    pattern = re.compile(r"(?i)" + re.escape(keyword).replace(r"\-", "[- ]?"))
    return pattern.sub("", text)


def prompt_from_path(relative_path: Path, 
                     prefix: str, 
                     exclude_words: Set[str],
                     drop_first_word: bool, 
                     pack_keyword: str) -> str:
    """
    Generate prompt from file path structure.
    
    Args:
        relative_path: Path relative to dataset root
        prefix: Prefix to add to prompt
        exclude_words: Set of words to exclude from prompt
        drop_first_word: Whether to drop first word from filename
        pack_keyword: Keyword that identifies sample pack folders
    
    Returns:
        Generated prompt string
    """
    parts = relative_path.with_suffix("").parts
    tokens = []

    # Handle pack keyword in first part
    if parts and pack_keyword.lower() in parts[0].lower():
        cleaned_part = _clean_text(_strip_keyword(parts[0], pack_keyword))
        if cleaned_part and cleaned_part.lower() not in exclude_words:
            tokens.append(cleaned_part)

    # Process remaining parts
    for idx, part in enumerate(parts[1:]):
        if part.lower() == "samples":
            continue
            
        cleaned_part = _clean_text(part)
        
        # Handle filename (last part)
        if idx == len(parts[1:]) - 1 and drop_first_word:
            words = cleaned_part.split()
            cleaned_part = " ".join(words[1:]) if len(words) > 1 else ""
        
        if cleaned_part and cleaned_part.lower() not in exclude_words:
            tokens.append(cleaned_part)

    body = ", ".join(tokens)
    return f"{prefix}, {body}" if prefix and body else prefix or body


def generate_metadata_module(output_path: Path,
                      prefix: str,
                             exclude_words: Set[str],
                             drop_first_word: bool,
                             pack_keyword: str) -> None:
    """
    Generate custom metadata module for stable-audio-tools.
    
    Args:
        output_path: Path to write the metadata module
        prefix: Prefix for prompts
        exclude_words: Words to exclude from prompts
        drop_first_word: Whether to drop first word from filenames
        pack_keyword: Keyword identifying sample pack folders
    """
    code = f'''import pathlib
import re
import json
from typing import Dict, Any, Optional

# Configuration
prefix = {json.dumps(prefix)}
exclude = set({json.dumps(list(exclude_words))})
drop_first = {drop_first_word}
pack_kw = {json.dumps(pack_keyword)}

# Helper functions
_clean = lambda t: t.replace('_', ' ').replace('-', ' ').strip()
_pat = re.compile(r'(?i)' + re.escape(pack_kw).replace(r'\\-', '[- ]?'))
_strip = lambda txt: _pat.sub('', txt)


def get_custom_metadata(info: Dict[str, Any], _: Optional[Any] = None) -> Dict[str, str]:
    """Generate metadata for audio file based on path structure."""
    rel_path = pathlib.Path(info['relpath'])
    parts = rel_path.with_suffix('').parts
    tokens = []
    
    # Handle pack keyword in first part
    if parts and pack_kw.lower() in parts[0].lower():
        cleaned = _clean(_strip(parts[0]))
        if cleaned and cleaned.lower() not in exclude:
            tokens.append(cleaned)
    
    # Process remaining parts
    for i, part in enumerate(parts[1:]):
        if part.lower() == 'samples':
            continue
        
        cleaned = _clean(part)
        
        # Handle filename (last part)
        if i == len(parts[1:]) - 1 and drop_first:
            words = cleaned.split()
            cleaned = ' '.join(words[1:]) if len(words) > 1 else ''
        
        if cleaned and cleaned.lower() not in exclude:
            tokens.append(cleaned)
    
    body = ', '.join(tokens)
    prompt = f'{{prefix}}, {{body}}' if prefix and body else prefix or body
    
    return {{'prompt': prompt}}
'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)


def find_audio_files(root_dir: Path) -> List[Path]:
    """Find all .wav files in directory tree."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
    
    wav_files = list(root_dir.rglob("*.wav"))
    if not wav_files:
        raise ValueError(f"No .wav files found in {root_dir}")
    
    return wav_files


def generate_dataset_config(output_path: Path,
                           dataset_dir: Path,
                           metadata_module_path: Path) -> None:
    """Generate dataset configuration JSON file."""
    config = {
        "dataset_type": "audio_dir",
        "datasets": [{
            "id": dataset_dir.name,
            "path": str(dataset_dir),
            "custom_metadata_module": str(metadata_module_path)
        }],
        "random_crop": True
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2))


def test_prompt_generation(wav_files: List[Path], 
                          root_dir: Path,
                          prefix: str,
                          exclude_words: Set[str],
                          drop_first_word: bool,
                          pack_keyword: str,
                          max_samples: int = 10) -> None:
    """Test prompt generation on sample files."""
    print(f"Testing prompt generation on {min(len(wav_files), max_samples)} samples:")
    print("-" * 80)
    
    for wav_file in wav_files[:max_samples]:
        relative_path = wav_file.relative_to(root_dir)
        prompt = prompt_from_path(relative_path, prefix, exclude_words, drop_first_word, pack_keyword)
        print(f"{relative_path}")
        print(f"  -> '{prompt}'")
        print()


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate dataset configuration for stable-audio-tools training"
    )
    parser.add_argument(
        "--dataset-dir", 
        default="./stable_audio_tools/data/local/finetune_data",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--out-base", 
        default="./stable_audio_tools/data/local/dataset_cfg",
        help="Base path for output files (without extension)"
    )
    parser.add_argument(
        "--prefix", 
        default="Lo-Fi, instrumental loop, chill",
        help="Prefix to add to all prompts"
    )
    parser.add_argument(
        "--pack-kw", 
        default="Inst",
        help="Keyword that identifies sample pack folders"
    )
    parser.add_argument(
        "--exclude", 
        default="Cymatics",
        help="Comma-separated words to exclude from prompts"
    )
    parser.add_argument(
        "--drop-first", 
        action="store_true", 
        default=True,
        help="Drop first word from file names"
    )
    parser.add_argument(
        "--test", 
        type=int, 
        default=0,
        help="Test prompt generation on N samples and exit (0 to disable)"
    )
    
    args = parser.parse_args()

    # Setup paths
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_base = Path(args.out_base).expanduser().resolve()
    metadata_module_path = output_base.with_suffix(".metadata.py")
    config_json_path = output_base.with_suffix(".json")
    
    # Parse exclude words
    exclude_words = {word.strip().lower() for word in args.exclude.split(",") if word.strip()}
    
    # Find audio files
    try:
        wav_files = find_audio_files(dataset_dir)
        print(f"Found {len(wav_files)} .wav files in {dataset_dir}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Test mode
    if args.test > 0:
        test_prompt_generation(
            wav_files, dataset_dir, args.prefix.strip(), exclude_words, 
            args.drop_first, args.pack_kw, args.test
        )
        return
    
    # Generate files
    print("Generating metadata module...")
    generate_metadata_module(
        metadata_module_path, args.prefix.strip(), exclude_words, 
        args.drop_first, args.pack_kw
    )
    
    print("Generating dataset configuration...")
    generate_dataset_config(config_json_path, dataset_dir, metadata_module_path)
    
    print(f"Generated files:")
    print(f"  Metadata module: {metadata_module_path}")
    print(f"  Dataset config:  {config_json_path}")


if __name__ == "__main__":
    main()

