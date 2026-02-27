# Chest images directory skeleton (MIMIC-CXR layout)

This folder shows the **directory layout** of downloaded MIMIC-CXR chest images. It matches the structure used by the [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.1.0/) (and the JPG version, e.g. `mimic-cxr-jpg`).

## Layout

```
chest_images_skeleton/
└── files/
    └── p{XX}/                    # patient prefix (p10, p11, ... p19)
        └── p{subject_id}/        # patient folder (e.g. p10000032)
            └── s{study_id}/       # study folder (e.g. s53189527)
                └── {dicom_id}.jpg # one or more chest X-ray images per study
```

- **files/** — Root of the image tree (same as in the PhysioNet release).
- **p10 … p19** — Top-level folders; the first three characters of the patient ID determine which folder (e.g. `p10000032` → `p10/`).
- **p{subject_id}** — One folder per patient (e.g. `p10000032`).
- **s{study_id}** — One folder per radiology study (e.g. `s53189527`). A study can have multiple images (e.g. PA and lateral).
- **{dicom_id}.jpg** — Image file named by the DICOM identifier (e.g. `2a2277a9-b0ded155-c0de8eb9-c124d10e-82c5caab.jpg`).

## Contents of this skeleton

- The **directory structure** above is created with the same paths as in `real_chest/split_data_demo.csv`.
- Image files are **minimal placeholder JPEGs** (1×1 pixel) so that:
  - The layout is visible and matches the real dataset.
  - You can run the chest WebDataset script with:
    - `--source_dir chest_images_skeleton`
    - `--split_csv real_chest/split_data_demo.csv`
  - Replace this skeleton with your full MIMIC-CXR download (or point `--source_dir` at your real `files/` parent) for real training data.

## Full dataset

After obtaining access and downloading MIMIC-CXR from PhysioNet, you will have a root directory that contains **files/** (and optionally metadata CSVs). Use that root as `--source_dir` when building the WebDataset; the CSV `image` column uses paths like `files/p10/p10000032/s53189527/xxx.jpg` relative to that root.
