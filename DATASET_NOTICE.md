# FRAUD1 dataset: credit, licence and provenance

The folder `FRAUD1_Dataset/` is **not our work**. It is an unmodified copy of:

**FRAUD1 — Face Replay Attack UQ Dataset, Version 1**
Daniel F. Smith, Arnold Wiliem and Brian C. Lovell
School of Information Technology and Electrical Engineering, The University of Queensland, Australia

## Required citation

Any published work that uses this dataset must cite:

> D. F. Smith, A. Wiliem, and B. C. Lovell, "Face Recognition on Consumer Devices: Reflections on Replay Attacks,"
> *IEEE Transactions on Information Forensics and Security*, vol. 10, no. 4, pp. 736–745, April 2015.
> DOI: 10.1109/TIFS.2015.2398819

```bibtex
@article{Smith_IEEE_2015,
  author  = {Smith, Daniel F. and Wiliem, Arnold and Lovell, Brian C.},
  title   = {Face Recognition on Consumer Devices: Reflections on Replay Attacks},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {10},
  number  = {4},
  pages   = {736--745},
  year    = {2015},
  doi     = {10.1109/TIFS.2015.2398819}
}
```

## Licence

The governing text is `FRAUD1_Dataset/FRAUD1-LICENCE.txt`, which is included unchanged and must stay with every copy.
In summary, it grants permission to **use, copy and distribute** the material for **non-commercial research purposes**
(academic research, teaching, scientific publications, personal experimentation), on these conditions:

- cite the publication above in any published work that uses the dataset;
- if you alter the material or make a derivative work, mark it prominently as not the original;
- no commercial use or commercial distribution, including licensing or selling it;
- the material is provided as is, without warranty;
- keep the original licence notice in all copies and derivatives;
- all other rights are reserved by the authors.

The summary above is for convenience only. If it differs from the licence file, the licence file prevails.

## Why this copy exists

The dataset was published by its creators in February 2015. The original download link
(`http://dropbox.eait.uq.edu.au/e3blovel/FRAUD1_Dataset.zip`) no longer works. We obtained the dataset directly from
its authors on 14 October 2026 and mirror it here, unaltered, so that the timing-jitter calibration in our paper can
be reproduced.

- SHA-256 of the archive we received: `eada19636a52087b57cb6652d182ee877741dfa62238f3535ba90b138096eeb8`
- Nothing inside `FRAUD1_Dataset/` has been added, removed or edited. This notice sits outside that folder on purpose.

## What the dataset contains

Video recordings of ten live subjects, and of printed-face and white-paper objects, taken while a screen shows short
colour flashes. The human-subject data were collected by the creators under University of Queensland School of ITEE
ethics approval EC201303SMI.A1. The recordings show identifiable faces, so please treat them with care and use them
only for the research purposes the licence allows.

## How we used it

Only to estimate timing jitter (the irregularity of detected flash onsets) from the live-subject videos, with
`fraud1_jitter.py`. The dataset contains no injection or deepfake attacks, and no attack-success figure in our paper
is derived from it.

## Questions about the dataset

Questions about the data itself, or requests beyond what the licence allows, should go to its creators, not to us.
