# Hardware validation (Section VI of the revised paper)

Testbed code, per-round lag data and analysis outputs behind Section VI, Fig. 2 and Tables V–VIII of the revised paper
(WcAISR-2027).

## What was recorded

- 99 sessions of 20 flash rounds (1,980 rounds), one subject (the first author), recorded on one laptop
  (AMD Ryzen 7 7435HS, NVIDIA RTX 3050 Laptop GPU, built-in USB 2.0 webcam, Windows 11).
- 59 live sessions in six conditions (normal and bright light at 30 fps; normal, dim and low-ambient light at 15 fps;
  720p at 30 fps), and three injection attacks through a virtual camera: 10 passthrough, 20 real-time face-swap and
  10 tint sessions.
- **No image or video of any face was stored.** For every frame, only the mean colour of a face box was kept, and the
  released data are the per-round lags derived from those colours. The face-swap target was a generated face of no real
  person; that image is not released.

## Files

| Path | Contents |
|---|---|
| `flash_capture.py` | Verifier: randomized full-screen colour flashes and synchronized webcam capture (stores face-box mean colour per frame only) |
| `attack_vcam.py` | Attack pipelines into a virtual camera: `passthrough`, `swap` (YuNet + ArcFace + inswapper-128, ONNX Runtime DirectML), `tint` |
| `extract_lags.py` | Per-round lag extraction (half-height crossing of the face-box colour step) and quality control |
| `analyze_real.py` | All analyses: baseline and noise, assumption checks, latency floor, decision rules on recorded and resampled sessions, network-delay emulation, overhead |
| `fraud1_revision.py` | FRAUD1 per-camera jitter and the colour-consistency check (needs `../FRAUD1_Dataset`) |
| `make_fig_hardware.py` | Fig. 2 |
| `RUNSHEET.md` | Recording protocol |
| `data/lags.csv` | One row per recorded round (all 99 sessions) |
| `data/attack_pipeline_logs/` | Per-frame processing time of each attack pipeline, and the ArcFace identity checks of the swap |
| `data/rtt_trace_wifi.txt` | 400 recorded round-trip times (ms) used for the network-delay emulation |
| `results/measured/` | Analysis at the measured floor (83 ms), first two rounds of each session treated as warm-up |
| `results/delta50/` | Analysis at the design floor of 50 ms (Tables V and VII) |
| `results/allrounds/` | Sensitivity check with warm-up rounds included |
| `results/fraud1/` | FRAUD1 jitter by camera and colour-check results |

## Reproducing the numbers

From this folder, with the requirements of the main README installed:

```
python analyze_real.py data/lags.csv --out results/measured --rtt data/rtt_trace_wifi.txt --attack-logs "data/attack_pipeline_logs/attack_*.csv"
python analyze_real.py data/lags.csv --out results/delta50 --delta 50 --rtt data/rtt_trace_wifi.txt --attack-logs "data/attack_pipeline_logs/attack_*.csv"
python analyze_real.py data/lags.csv --out results/allrounds --skip-rounds 0 --rtt data/rtt_trace_wifi.txt
python fraud1_revision.py ../FRAUD1_Dataset --out results/fraud1
python make_fig_hardware.py data/lags.csv ../PaperDiagrams/fig2_hardware_lag.png
```

`analyze_real.py` and `fraud1_revision.py` import `sim.py` and `fraud1_jitter.py` from the repository root.
The resampling uses a fixed seed (20261125), so the tables reproduce exactly.

## Recording new sessions

Recording needs a webcam, OBS Studio (for the virtual camera), `opencv-python`, `pyvirtualcam`, `onnxruntime-directml`,
`onnx` and `mss`. The YuNet detector (`face_detection_yunet_2023mar.onnx` from opencv_zoo, saved as `yunet.onnx`), the
InsightFace `buffalo_l` models and `inswapper_128.onnx` are not redistributed here; place them in a folder
`attack_models` next to the repository folder. The inswapper model is licensed for non-commercial research only.
See `RUNSHEET.md` for the protocol.

## Limits

One subject, one device and one face-swap pipeline. The measured floor belongs to this stack, and the attacker shared
the verifier's GPU, which may have inflated it. Replication on other devices, pipelines and people is welcome.
