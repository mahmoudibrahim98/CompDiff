#!/usr/bin/env python
"""Generate demographically-conditioned medical images from the released CompDiff models.

Downloads a trained CompDiff checkpoint from the Hugging Face Hub and runs the
turnkey ``CompDiffPipeline`` that ships with it. No training code or local
checkpoint is required.

Models:
    chest  -> mahmoudibra98/compdiff-chest-xray
    fundus -> mahmoudibra98/compdiff-fundus

Examples:
    python generate.py --modality chest \\
        --prompt "Cardiomegaly with small bilateral pleural effusions" \\
        --sex female --race White --age 67 --num_images 4 --output_dir generated

    python generate.py --modality fundus \\
        --prompt "glaucoma, severe vision loss, abnormal cup-disc ratio, myopia" \\
        --sex male --race Asian --age 55 --seed 0
"""

import argparse
import importlib.util
import os
import sys

REPO_IDS = {
    "chest": "mahmoudibra98/compdiff-chest-xray",
    "fundus": "mahmoudibra98/compdiff-fundus",
}


def _maybe_int(value):
    """Let the user pass sex/race as an index ("1") or a label ("female")."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate images from the released CompDiff models (Hugging Face).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modality", choices=REPO_IDS, required=True,
                   help="Which released model to use.")
    p.add_argument("--prompt", required=True,
                   help="Clinical findings only. Do NOT include sex/race — they are "
                        "conditioned through the HCN. Age is prepended automatically.")
    p.add_argument("--sex", required=True,
                   help="0/male or 1/female.")
    p.add_argument("--race", required=True,
                   help="Index or label. chest: 0=White,1=Black,2=Asian,3=Hispanic. "
                        "fundus: 0=White,1=Black,2=Asian.")
    p.add_argument("--age", type=int, default=None,
                   help="Age in years, prepended to the prompt (text-conditioned).")
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=75)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--output_dir", default="generated")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return p.parse_args()


def load_pipeline(local_dir, device, dtype):
    """Import CompDiffPipeline from the downloaded HF repo and build it."""
    module_path = os.path.join(local_dir, "compdiff_pipeline.py")
    spec = importlib.util.spec_from_file_location("compdiff_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compdiff_pipeline"] = module
    # hcn_v7.py is imported by the pipeline relative to its own directory, so make
    # the snapshot importable too.
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    spec.loader.exec_module(module)
    return module.CompDiffPipeline.from_pretrained(local_dir, device=device, dtype=dtype)


def main():
    args = parse_args()

    import torch
    from huggingface_hub import snapshot_download

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    repo_id = REPO_IDS[args.modality]
    print(f"Downloading {repo_id} from the Hugging Face Hub ...")
    local_dir = snapshot_download(repo_id)

    print(f"Loading CompDiffPipeline on {args.device} ({args.dtype}) ...")
    pipe = load_pipeline(local_dir, args.device, dtype)

    print(f"Generating {args.num_images} image(s) ...")
    images = pipe.generate(
        prompt=args.prompt,
        sex=_maybe_int(args.sex),
        race=_maybe_int(args.race),
        age=args.age,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        resolution=args.resolution,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(args.output_dir, f"{args.modality}_{i:03d}.png")
        img.save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
