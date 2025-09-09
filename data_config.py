import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


def _deduplicate_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving the original order."""
    seen: Set[str] = set()
    unique: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _parse_tags_field(raw_value: str) -> List[str]:
    """Parse a TSV "tags" field into a list of tag strings.

    MTG-Jamendo encodes tags like:
    - "genre---pop|mood/theme---corporate|instrument---piano" (pipe-separated)
    - "genre---pop, mood/theme---corporate, instrument---piano" (comma-separated)
    - JSON list ["genre---pop", ...]

    This function is robust and attempts JSON → '|' → ',' → whitespace.
    """
    if raw_value is None:
        return []

    value = str(raw_value).strip()
    if not value:
        return []

    # Try JSON list first
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    # Try pipe-separated
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]

    # Try comma-separated
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]

    # Fallback: split on whitespace
    return [part.strip() for part in re.split(r"\s+", value) if part.strip()]


def _extract_track_id(row: Dict[str, str]) -> str:
    """Extract a track_id string from a TSV row, falling back to the path stem."""
    candidates = ["track_id", "trackid", "id"]
    for key in candidates:
        for variant in (key, key.upper(), key.capitalize()):
            if variant in row and row[variant]:
                return str(row[variant]).strip()

    # Fallback to the filename stem in the path column
    for path_key in ("path", "PATH", "Path"):
        if path_key in row and row[path_key]:
            return Path(str(row[path_key]).strip()).stem

    raise KeyError(
        "Could not infer track_id from TSV row: missing 'track_id' and 'path'."
    )


def _extract_path(row: Dict[str, str]) -> str:
    """Extract the relative path column if present; otherwise empty string."""
    for key in ("path", "PATH", "Path"):
        if key in row and row[key]:
            return str(row[key]).strip()
    return ""


def _extract_duration(row: Dict[str, str]) -> float:
    """Extract duration seconds if present; otherwise -1.0."""
    for key in ("duration", "DURATION", "Duration"):
        if key in row and row[key]:
            try:
                return float(row[key])
            except Exception:
                return -1.0
    return -1.0


def load_mtg_jamendo_tsv(tsv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load MTG-Jamendo TSV into a mapping of track_id → record.

    The record contains: { 'path': str, 'duration': float, 'tags': List[str] }.
    Unknown columns are ignored. This loader is resilient to slight header
    variations and different tag separators.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    track_map: Dict[str, Dict[str, Any]] = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", restkey="__extra__", restval="")
        if reader.fieldnames is None:
            raise ValueError(f"TSV appears to have no header: {tsv_path}")

        # Identify potential tags column names
        tag_col_candidates = ["tags", "TAGS", "Tags"]

        for row in reader:
            track_id = _extract_track_id(row)
            path_value = _extract_path(row)
            duration_value = _extract_duration(row)

            # Extract tags from the 'TAGS' column plus any extra columns (tab-separated tags)
            tokens_raw: List[str] = []
            for tag_key in tag_col_candidates:
                if tag_key in row and row[tag_key]:
                    tokens_raw.append(row[tag_key])
                    break
            extras = row.get("__extra__")
            if isinstance(extras, list):
                tokens_raw.extend([str(x) for x in extras if str(x).strip()])

            tags_list: List[str] = []
            for tok in tokens_raw:
                tags_list.extend(_parse_tags_field(tok))
            tags = [t for t in tags_list if t]

            track_map[track_id] = {
                "path": path_value,
                "duration": duration_value,
                "tags": tags,
            }

    if not track_map:
        raise ValueError(f"No rows parsed from TSV: {tsv_path}")

    return track_map


def _split_tag(tag: str) -> Tuple[str, str]:
    """Split a tag like 'category---descriptor' into (category, descriptor).

    If no '---' is present, returns ("", tag).
    """
    if "---" in tag:
        category, descriptor = tag.split("---", 1)
        return category.strip().lower(), descriptor.strip()
    return "", tag.strip()


def build_prompt_from_tags(
    tags: List[str],
    category_order: List[str],
    lowercase: bool = False,
    reverse_within_category: bool = True,
) -> str:
    """Build a prompt string from a list of MTG-Jamendo tag strings.

    - Collect the descriptor after '---'.
    - Order descriptors by category priority (instrument → mood/theme → genre by default).
    - Within each category, reverse original order if reverse_within_category is True.
    - Deduplicate descriptors while preserving final order.
    - Optionally lowercase the result.
    """
    category_to_descriptors: Dict[str, List[str]] = {}
    uncategorized: List[str] = []

    for tag in tags:
        category, descriptor = _split_tag(tag)
        if not descriptor:
            continue
        if lowercase:
            descriptor = descriptor.lower()
        if category:
            key = category.lower()
            category_to_descriptors.setdefault(key, []).append(descriptor)
        else:
            uncategorized.append(descriptor)

    ordered_descriptors: List[str] = []
    for cat in category_order:
        if cat in category_to_descriptors:
            seq = category_to_descriptors[cat]
            ordered_descriptors.extend(reversed(seq) if reverse_within_category else seq)

    # Append any remaining categories not listed in category_order, preserving their order
    for cat, values in category_to_descriptors.items():
        if cat not in category_order:
            seq = values
            ordered_descriptors.extend(reversed(seq) if reverse_within_category else seq)

    # Finally, append any uncategorized descriptors
    ordered_descriptors.extend(reversed(uncategorized) if reverse_within_category else uncategorized)

    return ", ".join(_deduplicate_preserve_order(ordered_descriptors))


def _candidate_keys_from_relpath(relpath: Path) -> List[str]:
    """Derive robust candidate keys from a relative audio path.

    Handles Jamendo low-quality filenames like '23198.low.mp3' by trying:
    - stem and stem without trailing '.low'
    - digits extracted from stem and without leading zeros
    - path without suffix and with common suffix variants (.mp3, .low.mp3)
    - jamendo-prefixed variants of the above
    """
    stem = relpath.stem
    no_low_stem = stem[:-4] if stem.endswith(".low") else stem
    digits = "".join(ch for ch in no_low_stem if ch.isdigit())
    digits_nz = digits.lstrip("0") if digits else ""

    rel_no_suffix = relpath.with_suffix("")
    rel_with_suffix = relpath
    alt_no_low = rel_no_suffix
    if str(rel_no_suffix).endswith(".low"):
        alt_no_low = Path(str(rel_no_suffix)[:-4])

    candidates: List[str] = []
    # Stem-based
    for key in (stem, no_low_stem, digits, digits_nz):
        if key and key not in candidates:
            candidates.append(key)

    # Path-based
    for key in (
        str(rel_no_suffix),
        str(rel_with_suffix),
        str(alt_no_low),
        f"{alt_no_low}.mp3",
        f"{alt_no_low}.low.mp3",
    ):
        if key and key not in candidates:
            candidates.append(key)

    # Jamendo-prefixed variants
    jamendo_variants = [f"jamendo/{k}" for k in list(candidates) if "/" in k or "." in k]
    for key in jamendo_variants:
        if key not in candidates:
            candidates.append(key)

    return candidates


def write_tag_mapping_json(mapping_path: Path, track_map: Dict[str, Dict[str, Any]]) -> None:
    """Write a robust JSON mapping including useful aliases for lookups.

    Keys include (when derivable):
    - Original track_id string
    - Digits-only and digits-without-leading-zeros forms
    - 'track_{digits}' variants
    - From PATH: stem, path without suffix, and .mp3 and .low.mp3 forms
    - Forms prefixed with 'jamendo/' for low-quality subset layouts
    - Additional convenience forms like 'stem.low', 'stem.low.mp3'
    """
    mapping: Dict[str, List[str]] = {}

    def add_key(key: str, tags: List[str]) -> None:
        k = str(key).strip()
        if k and k not in mapping:
            mapping[k] = tags

    for tid, rec in track_map.items():
        tags = rec.get("tags", [])
        if not isinstance(tags, list):
            continue
        
        # Base keys from track_id
        tid_str = str(tid).strip()
        add_key(tid_str, tags)
        digits = "".join(ch for ch in tid_str if ch.isdigit())
        if digits:
            add_key(digits, tags)
            add_key(digits.lstrip("0"), tags)
            add_key(f"track_{digits}", tags)
            add_key(f"track_{digits.lstrip('0')}", tags)

        # Keys derived from TSV 'path'
        path_str = str(rec.get("path", "")).strip()
        if path_str:
            p = Path(path_str)
            stem = p.stem
            no_suffix = str(p.with_suffix(""))
            with_mp3 = f"{no_suffix}.mp3"
            with_low_mp3 = f"{no_suffix}.low.mp3"

            for key in (stem, no_suffix, with_mp3, with_low_mp3):
                add_key(key, tags)

            # jamendo-prefixed forms (common in local layouts)
            for key in (
                f"jamendo/{no_suffix}",
                f"jamendo/{with_mp3}",
                f"jamendo/{with_low_mp3}",
            ):
                add_key(key, tags)

            # '.low' variations
            for key in (
                f"{stem}.low",
                f"{stem}.low.mp3",
                f"{no_suffix}.low",
                f"jamendo/{stem}.low",
                f"jamendo/{stem}.low.mp3",
                f"jamendo/{no_suffix}.low",
            ):
                add_key(key, tags)

            stem_digits = "".join(ch for ch in stem if ch.isdigit())
            if stem_digits:
                add_key(stem_digits, tags)
                add_key(stem_digits.lstrip("0"), tags)
                add_key(f"track_{stem_digits}", tags)
                add_key(f"track_{stem_digits.lstrip('0')}", tags)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False))


def generate_custom_metadata_module(
    output_path: Path,
    mapping_json_path: Path,
    category_order: List[str],
    prompt_prefix: str,
    lowercase: bool,
) -> None:
    """Generate a custom metadata module consumed by stable-audio-tools.

    The module loads the tag mapping JSON once, derives a robust track key from
    the audio filename (handling Jamendo low-quality names like *.low.mp3),
    constructs the prompt according to the category order, and returns
    {'prompt': ...}.
    """
    code = (
        "import json\n"
        "import pathlib\n"
        "from typing import Dict, Any, Optional\n\n"
        f"MAPPING_JSON = {json.dumps(str(mapping_json_path))}\n"
        f"CATEGORY_ORDER = {json.dumps([c.lower() for c in category_order])}\n"
        f"PROMPT_PREFIX = {json.dumps(prompt_prefix)}\n"
        f"LOWERCASE = {json.dumps(lowercase)}\n\n"
        "_TAG_MAP: Optional[Dict[str, list]] = None\n\n\n"
        "def _load_mapping() -> Dict[str, list]:\n"
        "    global _TAG_MAP\n"
        "    if _TAG_MAP is None:\n"
        "        with open(MAPPING_JSON, \"r\", encoding=\"utf-8\") as f:\n"
        "            _TAG_MAP = json.load(f)\n"
        "    return _TAG_MAP\n\n\n"
        "def _split_tag(tag: str):\n"
        "    if \"---\" in tag:\n"
        "        category, descriptor = tag.split(\"---\", 1)\n"
        "        return category.strip().lower(), descriptor.strip()\n"
        "    return \"\", tag.strip()\n\n\n"
        "def _dedup(items):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for it in items:\n"
        "        if it not in seen:\n"
        "            seen.add(it)\n"
        "            out.append(it)\n"
        "    return out\n\n\n"
        "def _build_prompt(tags):\n"
        "    category_to_desc = {}\n"
        "    other = []\n"
        "    for t in tags:\n"
        "        cat, desc = _split_tag(t)\n"
        "        if not desc:\n"
        "            continue\n"
        "        if LOWERCASE:\n"
        "            desc = desc.lower()\n"
        "        if cat:\n"
        "            key = cat.lower()\n"
        "            category_to_desc.setdefault(key, []).append(desc)\n"
        "        else:\n"
        "            other.append(desc)\n\n"
        "    ordered = []\n"
        "    for cat in CATEGORY_ORDER:\n"
        "        if cat in category_to_desc:\n"
        "            ordered.extend(category_to_desc[cat])\n"
        "    for cat, vals in category_to_desc.items():\n"
        "        if cat not in CATEGORY_ORDER:\n"
        "            ordered.extend(vals)\n"
        "    ordered.extend(other)\n\n"
        "    prompt_body = \", \".join(_dedup(ordered))\n"
        "    if PROMPT_PREFIX and prompt_body:\n"
        "        return f\"{PROMPT_PREFIX}, {prompt_body}\"\n"
        "    return PROMPT_PREFIX or prompt_body\n\n\n"
        "def _candidate_keys_from_relpath(relpath: pathlib.Path):\n"
        "    stem = relpath.stem\n"
        "    no_low_stem = stem[:-4] if stem.endswith(\".low\") else stem\n"
        "    digits = ''.join(ch for ch in no_low_stem if ch.isdigit())\n"
        "    digits_nz = digits.lstrip('0') if digits else ''\n"
        "    rel_no_suffix = relpath.with_suffix(\"\")\n"
        "    rel_with_suffix = relpath\n"
        "    alt_no_low = rel_no_suffix\n"
        "    if str(rel_no_suffix).endswith(\".low\"):\n"
        "        alt_no_low = pathlib.Path(str(rel_no_suffix)[:-4])\n"
        "    candidates = []\n"
        "    for key in (stem, no_low_stem, digits, digits_nz):\n"
        "        if key and key not in candidates:\n"
        "            candidates.append(key)\n"
        "    for key in (\n"
        "        str(rel_no_suffix),\n"
        "        str(rel_with_suffix),\n"
        "        str(alt_no_low),\n"
        "        f\"{alt_no_low}.mp3\",\n"
        "        f\"{alt_no_low}.low.mp3\",\n"
        "    ):\n"
        "        if key and key not in candidates:\n"
        "            candidates.append(key)\n"
        "    jamendo_variants = [f\"jamendo/{k}\" for k in list(candidates) if '/' in k or '.' in k]\n"
        "    for key in jamendo_variants:\n"
        "        if key not in candidates:\n"
        "            candidates.append(key)\n"
        "    return candidates\n\n\n"
        "def get_custom_metadata(info: Dict[str, Any], _: Optional[Any] = None) -> Dict[str, str]:\n"
        "    relpath = pathlib.Path(info.get(\"relpath\") or info.get(\"path\") or \"\")\n"
        "    mapping = _load_mapping()\n"
        "    tags = None\n"
        "    for key in _candidate_keys_from_relpath(relpath):\n"
        "        tags = mapping.get(key)\n"
        "        if tags is not None:\n"
        "            break\n"
        "    if tags is None and relpath.stem.isdigit():\n"
        "        tags = mapping.get(str(int(relpath.stem)))\n"
        "    prompt = _build_prompt(tags or [])\n"
        "    return {\"prompt\": prompt}\n"
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)


def generate_dataset_config(
    output_path: Path,
    dataset_dir: Path,
    metadata_module_path: Path,
    random_crop: bool,
) -> None:
    """Write a dataset config JSON for stable-audio-tools."""
    config = {
        "dataset_type": "audio_dir",
        "datasets": [
            {
            "id": dataset_dir.name,
            "path": str(dataset_dir),
                "custom_metadata_module": str(metadata_module_path),
            }
        ],
        "random_crop": bool(random_crop),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2))


def discover_audio_files(dataset_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """Discover audio files under dataset_dir matching the given extensions."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    exts = tuple(ext.lower() for ext in extensions)
    results: List[Path] = []
    for ext in exts:
        results.extend(dataset_dir.rglob(f"*{ext}"))
        results.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    if not results:
        raise ValueError(
            f"No audio files with extensions {extensions} were found in {dataset_dir}"
        )
    return results


def preview_prompts(
    sample_files: List[Path],
    dataset_root: Path,
    mapping_json: Path,
    category_order: List[str],
    lowercase: bool,
                          prefix: str,
    limit: int,
) -> None:
    """Preview prompts for a subset of files to validate mapping and prompt logic."""
    mapping: Dict[str, List[str]] = json.loads(mapping_json.read_text())

    def build_prompt(tags: List[str]) -> str:
        return build_prompt_from_tags(tags, category_order, lowercase, reverse_within_category=False)

    print(f"Previewing prompts for {min(limit, len(sample_files))} files:")
    print("-" * 80)
    for path in sample_files[:limit]:
        rel = path.relative_to(dataset_root)
        tags: List[str] = []
        for key in _candidate_keys_from_relpath(rel):
            tags = mapping.get(key) or []
            if tags:
                break
        if not tags and rel.stem.isdigit():
            tags = mapping.get(str(int(rel.stem))) or []
        body = build_prompt(tags)
        prompt = f"{prefix}, {body}" if prefix and body else prefix or body
        print(f"{rel}")
        print(f"  -> '{prompt}'")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dataset config and custom metadata for MTG-Jamendo (audio_dir).\n"
            "Prompts are constructed from tags after '---' in category order"
            " (default: instrument, mood/theme, genre)."
        )
    )
    parser.add_argument(
        "--jamendo-tsv",
        default="./data/raw_30s.tsv",
        help=(
            "Path to MTG-Jamendo TSV (e.g., autotagging_moodtheme.tsv or autotagging.tsv)"
        ),
    )
    parser.add_argument(
        "--dataset-dir", 
        default="./data/jamendo",
        help=(
            "Directory containing the Jamendo audio files (unpacked). The loader will"
            " recursively scan this directory for audio."
        ),
    )
    parser.add_argument(
        "--out-base", 
        default="./stable_audio_tools/data/jamendo/jamendo",
        help=(
            "Base path for output artifacts (without extension). Will write .json,"
            " .metadata.py and .tags.json next to this base."
        ),
    )
    parser.add_argument(
        "--category-order",
        default="instrument,mood/theme,genre",
        help=(
            "Comma-separated category order for prompt construction."
            " Example: 'instrument,mood/theme,genre'"
        ),
    )
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="Optional prefix to prepend to every prompt (e.g., 'Jamendo').",
    )
    parser.add_argument(
        "--lowercase-tags",
        action="store_true",
        help="Lowercase tag descriptors in prompts.",
    )
    parser.add_argument(
        "--audio-exts",
        default=".mp3,.wav,.flac",
        help="Comma-separated list of audio file extensions to scan.",
    )
    parser.add_argument(
        "--random-crop",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to enable random cropping in dataset config (1=yes, 0=no).",
    )
    parser.add_argument(
        "--test", 
        type=int, 
        default=0,
        help="Preview prompts for N files and exit (0 to disable).",
    )
    
    args = parser.parse_args()

    # Resolve paths
    tsv_path = Path(args.jamendo_tsv).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_base = Path(args.out_base).expanduser().resolve()
    config_json_path = out_base.with_suffix(".json")
    metadata_module_path = out_base.with_suffix(".metadata.py")
    tag_mapping_json_path = out_base.with_suffix(".tags.json")

    # Load TSV → track map
    track_map = load_mtg_jamendo_tsv(tsv_path)
    print(f"Loaded {len(track_map)} track metadata entries from {tsv_path}")

    # Write compact tag mapping
    write_tag_mapping_json(tag_mapping_json_path, track_map)
    print(f"Wrote tag mapping JSON: {tag_mapping_json_path}")

    # Generate custom metadata module
    category_order = [c.strip().lower() for c in args.category_order.split(",") if c.strip()]
    generate_custom_metadata_module(
        output_path=metadata_module_path,
        mapping_json_path=tag_mapping_json_path,
        category_order=category_order,
        prompt_prefix=str(args.prompt_prefix).strip(),
        lowercase=bool(args.lowercase_tags),
    )
    print(f"Wrote custom metadata module: {metadata_module_path}")

    # Write dataset config JSON
    generate_dataset_config(
        output_path=config_json_path,
        dataset_dir=dataset_dir,
        metadata_module_path=metadata_module_path,
        random_crop=bool(args.random_crop),
    )
    print(f"Wrote dataset config JSON: {config_json_path}")

    # Optionally preview prompts for a few files
    if args.test > 0:
        exts = tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in args.audio_exts.split(",") if ext.strip()
        )
        try:
            audio_files = discover_audio_files(dataset_dir, exts)
        except Exception as e:
            print(f"Warning: could not discover audio files for preview: {e}", file=sys.stderr)
            return

        print(f"Discovered {len(audio_files)} audio files under {dataset_dir} matching {exts}")
        preview_prompts(
            sample_files=sorted(audio_files)[: args.test],
            dataset_root=dataset_dir,
            mapping_json=tag_mapping_json_path,
            category_order=category_order,
            lowercase=bool(args.lowercase_tags),
            prefix=str(args.prompt_prefix).strip(),
            limit=int(args.test),
        )


if __name__ == "__main__":
    main()

