import argparse, json, re, sys
from pathlib import Path


def _clean(t: str) -> str:
    return re.sub(r"\d+", "", t).replace("_", " ").replace("-", " ").strip()


def _strip_kw(text: str, kw: str) -> str:
    pat = re.compile(r"(?i)" + re.escape(kw).replace(r"\-", "[- ]?"))
    return pat.sub("", text)


def prompt_from_rel(rel: Path, prefix: str, exclude: set,
                    drop_first: bool, pack_kw: str) -> str:
    parts = rel.with_suffix("").parts
    toks = []

    if parts and pack_kw.lower() in parts[0].lower():
        p = _clean(_strip_kw(parts[0], pack_kw))
        if p and p.lower() not in exclude:
            toks.append(p)

    for idx, part in enumerate(parts[1:]):
        if part.lower() == "samples":
            continue
        p = _clean(part)
        if idx == len(parts[1:]) - 1 and drop_first:          # file name
            w = p.split()
            p = " ".join(w[1:]) if len(w) > 1 else ""
        if p and p.lower() not in exclude:
            toks.append(p)

    body = ", ".join(toks)
    return f"{prefix}, {body}" if prefix and body else prefix or body


def build_metadata_py(path: Path, prefix: str, exclude: set,
                      drop_first: bool, pack_kw: str) -> None:
    """
    This function generates a data config for training/finetuning.
    """
    code = (
        "import pathlib, re, json\n"
        f"prefix={json.dumps(prefix)}\n"
        f"exclude=set({json.dumps(list(exclude))})\n"
        f"drop_first={json.dumps(drop_first)}\n"
        f"pack_kw={json.dumps(pack_kw)}\n"
        "_clean=lambda t:re.sub(r'\\d+','',t).replace('_',' ').replace('-',' ').strip()\n"
        "_pat=re.compile(r'(?i)'+re.escape(pack_kw).replace(r'\\-','[- ]?'))\n"
        "_strip=lambda txt:_pat.sub('',txt)\n"
        "def get_custom_metadata(info,_):\n"
        " p=pathlib.Path(info['relpath']);parts=p.with_suffix('').parts;t=[]\n"
        " if parts and pack_kw.lower() in parts[0].lower():\n"
        "  q=_clean(_strip(parts[0]));\n"
        "  if q and q.lower() not in exclude:t.append(q)\n"
        " for i,part in enumerate(parts[1:]):\n"
        "  if part.lower()=='samples':continue\n"
        "  q=_clean(part)\n"
        "  if i==len(parts[1:])-1 and drop_first:\n"
        "   w=q.split();q=' '.join(w[1:]) if len(w)>1 else ''\n"
        "  if q and q.lower() not in exclude:t.append(q)\n"
        " body=', '.join(t)\n"
        " return {'prompt': f'{prefix}, {body}' if prefix and body else prefix or body}\n"
    )
    path.write_text(code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./stable_audio_tools/data/local/finetune_data")
    ap.add_argument("--out_base", default="./stable_audio_tools/data/local/dataset_cfg")
    ap.add_argument("--prefix", default="8-bit, Game Sound")
    ap.add_argument("--pack_kw", default="8-bit", help="keyword that marks a sample-pack folder")
    ap.add_argument("--exclude", default="", help="comma-separated words to drop")
    ap.add_argument("--drop_first", action="store_true", default=True, help="drop first word of file names")
    ap.add_argument("--test", type=int, default=0, help="print N prompts then exit")
    args = ap.parse_args()

    root = Path(args.dataset_dir).expanduser().resolve()
    wavs = list(root.rglob("*.wav"))
    if not wavs:
        sys.exit("no .wav files found")

    exclude = {w.strip().lower() for w in args.exclude.split(",") if w.strip()}

    if args.test:
        for p in wavs[:args.test]:
            print(p.relative_to(root), "->",
                  prompt_from_rel(p.relative_to(root), args.prefix,
                                  exclude, args.drop_first, args.pack_kw))
        return

    out_base = Path(args.out_base).expanduser().resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    meta_py  = out_base.with_suffix(".metadata.py")
    cfg_json = out_base.with_suffix(".json")

    build_metadata_py(meta_py, args.prefix.strip(),
                      exclude, args.drop_first, args.pack_kw)

    cfg = {
        "dataset_type": "audio_dir",
        "datasets": [{
            "id": root.stem,
            "path": str(root),
            "custom_metadata_module": str(meta_py)
        }],
        "random_crop": True
    }
    cfg_json.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

