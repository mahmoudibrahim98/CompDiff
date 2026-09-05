#!/usr/bin/env python
"""Generate demographically-conditioned medical images from the released CompDiff models.

Downloads a trained CompDiff checkpoint from the Hugging Face Hub and runs the
turnkey ``CompDiffPipeline`` that ships with it. No training code or local
checkpoint is required.

Models:
    chest  -> mahmoudibra98/compdiff-chest-xray
              Current release: sex, race and age all go through the conditioner
              (age is a continuous value in years, not a bin).
    fundus -> mahmoudibra98/compdiff-fundus
              Current release: sex and race go through the conditioner, age is
              prepended to the prompt.

Every release keeps the same ``generate(prompt, sex, race, age, ...)`` interface,
so this script works unchanged across them. Pass ``--revision v1`` to use the
first (July 2026) chest release.

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

RACE_LABELS = {
    "chest": "0=White, 1=Black/African American, 2=Asian, 3=Hispanic/Latino",
    "fundus": "0=White, 1=Black/African American, 2=Asian (no Hispanic/Latino class)",
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
                   help="Clinical findings only. Do NOT include age, sex or race in the "
                        "text; they are passed separately and conditioned by the model.")
    p.add_argument("--sex", required=True,
                   help="0/male or 1/female.")
    p.add_argument("--race", required=True,
                   help=f"Index or label. chest: {RACE_LABELS['chest']}. fundus: {RACE_LABELS['fundus']}.")
    p.add_argument("--age", type=float, required=True,
                   help="Age in years (e.g. 67). Chest: conditioned continuously through the "
                        "conditioner. Fundus: prepended to the prompt as '<age> years old.'.")
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=75)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--output_dir", default="generated")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--revision", default=None,
                   help="Hub revision (branch, tag or commit) of the model, e.g. 'v1' for "
                        "the first chest release. Default: latest on main.")
    return p.parse_args()


def load_pipeline(local_dir, device, dtype):
    """Import CompDiffPipeline from the downloaded HF repo and build it."""
    module_path = os.path.join(local_dir, "compdiff_pipeline.py")
    spec = importlib.util.spec_from_file_location("compdiff_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compdiff_pipeline"] = module
    # The pipeline imports its conditioner module (compdiff2.py, or hcn_v7.py in
    # the first release) relative to its own directory; make the snapshot importable.
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
    rev = f" (revision {args.revision})" if args.revision else ""
    print(f"Downloading {repo_id}{rev} from the Hugging Face Hub ...")
    local_dir = snapshot_download(repo_id, revision=args.revision)

    print(f"Loading CompDiffPipeline on {args.device} ({args.dtype}) ...")
    pipe = load_pipeline(local_dir, args.device, dtype)

    # The first release conditioned age through the prompt and takes an int.
    age = args.age if getattr(pipe.hcn, "encode_age", False) else int(round(args.age))

    print(f"Generating {args.num_images} image(s) ...")
    images = pipe.generate(
        prompt=args.prompt,
        sex=_maybe_int(args.sex),
        race=_maybe_int(args.race),
        age=age,
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
