# Anytime-Valid Stopping Rules for Randomized-Illumination Face Liveness under Persistent Attackers

Code, result tables, figures and calibration data for the paper of the same name by
Jitvan Chadha, Saravanan K. N. and Ramamurthy B. (Department of Computer Science, CHRIST (Deemed to be University),
Bangalore, India).

The paper studies how to decide, from a few noisy rounds of a screen-flash challenge, whether a face is live or is a
real-time face swap that needs extra time to respond. Sections IV and V analyze the rules in theory and simulation.
The revised paper (WcAISR-2027) adds a hardware validation: a screen-flash testbed, a real-time face-swap injection
attack, a measured latency floor, and every decision rule applied to the recorded sessions. Its code, data and results
are in [hardware_validation/](hardware_validation/README.md).

## The dataset in this repository is not ours

`FRAUD1_Dataset/` is an unmodified copy of the **FRAUD1 (Face Replay Attack UQ Dataset, Version 1)** by
Daniel F. Smith, Arnold Wiliem and Brian C. Lovell, The University of Queensland. If you use it, you must cite:

> D. F. Smith, A. Wiliem, and B. C. Lovell, "Face Recognition on Consumer Devices: Reflections on Replay Attacks,"
> *IEEE Transactions on Information Forensics and Security*, vol. 10, no. 4, pp. 736–745, April 2015.
> DOI: 10.1109/TIFS.2015.2398819

It is licensed for **non-commercial research only**. The licence is `FRAUD1_Dataset/FRAUD1-LICENCE.txt` and must stay
with every copy. See [DATASET_NOTICE.md](DATASET_NOTICE.md) for the full credit, the licence summary and how we
obtained the data. The recordings show identifiable faces, so please use them only as the licence allows.

We used FRAUD1 for one thing: estimating timing jitter (11.756 ms) from its live-subject videos. It contains no
injection or deepfake attacks, and no attack-success figure in the paper comes from it.

## Requirements

Python 3.10 or newer, and:

```
pip install -r requirements.txt
```

The two `verify_*.pl` scripts need only core Perl. Results in this repository were produced with Python 3.14,
NumPy 2.5, SciPy 1.18 and pandas 3.x. All simulations use the fixed seed 20261011.

## What is where

| Path | Contents |
|---|---|
| `sim.py` | Stress benchmark (noise 25 ms): headline table, persistence sweep, floor sweep |
| `sim_extensions.py` | Optional stopping (peeking) and adaptive attackers |
| `sim_wp3.py` | Live-offset sweep with fixed and plug-in weights, wrong floor, wrong noise level |
| `sim_wp4.py` | Efficiency grid against the fixed-round test |
| `sim_phase1.py` | False rejects against the round cap, plug-in efficiency |
| `sim_redteam.py` | Bernstein form under non-Gaussian noise, stopping-time tails, plug-in weights at the calibrated jitter |
| `sim_guessing.py` | Challenge-guessing attacker for four palette sizes |
| `calibrated_run/` | The scripts that produced `runs/real_v2`: jitter estimation from FRAUD1, then the simulations at that jitter |
| `fraud1_jitter.py` | Jitter estimator (same file as `calibrated_run/fraud1_jitter.py`) |
| `make_paper_diagrams.py` | Draws the four figures of the paper from the CSV files |
| `verify_T3_T4.pl`, `verify_T3_T4_drift.pl` | Independent numerical checks of the theory (expected rounds, fixed-round inflation, SPRT drift root) |
| `*.csv` in the root | Outputs of the stress-benchmark scripts |
| `runs/real_v2/` | Calibrated run: jitter estimate, per-video values (`calib_videos.csv`), manifest, and simulation outputs |
| `redteam_out/` | Outputs of `sim_redteam.py` and `sim_guessing.py` |
| `calibration_ablation/` | Jitter estimates under three other detector and region-of-interest settings |
| `PaperDiagrams/` | The four figures (PNG at 600 dpi and vector PDF) |
| `FRAUD1_Dataset/` | The third-party dataset described above |

## Reproducing the paper

Run every command from the repository root. Sample sizes match the paper, so some runs take several minutes.

| Paper item | Command | Output |
|---|---|---|
| Table I, Fig. 1 (persistence) | `python sim.py --n 2000000 --out . --stages 1,2` | `e1_headline.csv`, `e2_persistence.csv` |
| Sample efficiency | `python sim.py --n 2000000 --out . --stages 3` and `python sim_wp4.py` | `e3_delta_sweep.csv`, `efficiency_analysis.csv` |
| Stopping-time tails | `python sim_redteam.py` | `redteam_out/r3_round_tails.csv` |
| Table II (live offset) | `python sim_wp3.py` and `python sim_phase1.py --tasks 1c,1e` | `e4_prime_live_offset.csv`, `e_bpcer_vs_tmax.csv`, `efficiency_plugin.csv` |
| Table III, Fig. 2 (misspecification) | `python sim_wp3.py` | `e5_prime_wrong_floor.csv`, `e12_wrong_sigma.csv` |
| Optional stopping, adaptive attackers | `python sim_extensions.py --n 500000 --stages 7,8` | `e7_peeking.csv`, `e8_adaptive_attacker.csv` |
| Table IV (heavy-tailed noise) | `python sim_redteam.py` | `redteam_out/r1_bernstein.csv` |
| Table V (challenge guessing) | `python sim_guessing.py` | `redteam_out/r5_guessing.csv` |
| Jitter estimate from FRAUD1 | `python fraud1_jitter.py FRAUD1_Dataset/FRAUD1-Live_Persons --roi auto --cycles 4 --detector peaks --out calib.json` | `calib.json`, compare with `runs/real_v2/calib.json` |
| Table VI, Figs. 3 and 4 (calibrated regime, noise sweep, phase diagram) | `python calibrated_run/sim.py --n 2000000 --sigma 11.756 --delta 50 --tau-live 10 --out my_run` | compare with `runs/real_v2/` |
| Plug-in weights at the calibrated jitter | `python sim_redteam.py` | `redteam_out/r4_plugin_calibrated.csv` |
| Theory checks | `perl verify_T3_T4.pl` and `perl verify_T3_T4_drift.pl` | printed to the terminal |
| All four figures | `python make_paper_diagrams.py` | `PaperDiagrams/` |

`calibrated_run/run_calibrated_sim.py` chains the jitter estimate and the simulation and writes a manifest; run it
with `--help` for its options.

## Notes for anyone reading the code

- There are two files called `sim.py`. The one in the root covers the stress benchmark. The one in `calibrated_run/`
  adds a noise sweep and a sweep over the latency floor, and is the one that produced `runs/real_v2`.
- `sim_wp3.py` also contains an early heavy-tail experiment and an early guessing experiment. Both are superseded.
  The early heavy-tail run used a betting weight under which the Bernstein form rejects every session, so its "zero
  false accepts" meant nothing. Use `sim_redteam.py` and `sim_guessing.py` for those two results.
- `sim_extensions.py` has four stages. The paper uses stages 7 and 8 only.
- The script hashes recorded in `runs/real_v2/RUN_MANIFEST.json` were taken on the original working copy and can
  differ from the files here by line endings. Reproduce the outputs to check a script, not its hash.
- The jitter estimator needs sub-frame timing from 30 frames-per-second video and is reliable only up to roughly
  15 to 20 ms of jitter.

## Limits of this work

- Simulation only. No lag was measured on hardware.
- The latency floor of a face-swap pipeline is assumed. The guarantees hold only for attackers at or above it.
- Attacks that answer the flash without re-synthesizing the face are outside the timing model.

## Licence

- `FRAUD1_Dataset/`: the FRAUD1 licence (non-commercial research, attribution required). Not ours to relicense.
- Our code, tables and figures: licence to be added by the authors before this repository is made public.

## Citation

If you use the code or results, please cite the paper (reference to be added on publication) and, if you use the
dataset, the FRAUD1 paper above.
