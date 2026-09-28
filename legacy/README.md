# Legacy material

Kept for provenance; not needed to use or retrain CompDiff.

- `submit_jobs_summarized/`: the SLURM scripts used on our cluster for earlier runs
  (run names v0 = prompt-conditioned baseline, v4 = flat demographic encoder, v7 = the first
  sex x race HCN). They contain cluster-specific absolute paths and will not run elsewhere as is.
- `configs/downstream_eval/`: downstream-classifier configs from the first (conference) version
  of the study, comparing v0 and v7 synthetic corpora. They reference absolute data paths on our
  cluster. The current manuscript compares CompDiff with RoentGen-v2 under a different recipe.

For current training and generation, use `configs/` and the instructions in the top-level README.
