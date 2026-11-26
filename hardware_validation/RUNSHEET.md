# Recording run sheet (hardware validation, WcAISR-2027 revision)

Each session: 2 s warm-up, then 30 rounds of one random colour flash (600 ms on, 600–1000 ms black), about 45 s.
Only the mean colour of the face box is stored per frame, never an image. Press **Esc** to abort a session.
Sit about 40–50 cm from the screen, face the camera, keep still, screen brightness at maximum.

Python: `..\..\.venv\Scripts\python.exe` (run from this `testbed` folder). All sessions use the DirectShow backend:
camera 0 = built-in webcam, camera 1 = OBS Virtual Camera.

| # | Block | What you do | Command (append `--sessions N`) | N | Time |
|---|---|---|---|---|---|
| 1 | Live, normal light, 30 fps | usual room light | `flash_capture.py --camera 0 --label live --condition normal` | 20 | 15 min |
| 2 | Live, bright | all lights on plus a lamp toward your face | `... --label live --condition bright` | 10 | 8 min |
| 3 | Live, dim | room lights off, only the laptop screen | `... --label live --condition dim` | 10 | 8 min |
| 4 | Live, 15 fps | normal light | `... --label live --condition normal --fps 15` | 10 | 8 min |
| 5 | Live, 720p | normal light | `... --label live --condition normal --width 1280 --height 720` | 5 | 4 min |
| 6 | Replay | play a pre-recorded video of your face on your phone, held 10–15 cm in front of the webcam | `... --label replay --condition normal` | 10 | 8 min |
| 7 | Injection, passthrough | start `attack_vcam.py --mode passthrough` in a second terminal; sit as usual | `flash_capture.py --camera 1 --camera-name obsvcam --label passthrough --condition normal` | 10 | 8 min |
| 8 | Injection, face swap | start `attack_vcam.py --mode swap --target <photo>`; sit as usual | `... --camera 1 --camera-name obsvcam --label faceswap --condition normal` | 20 | 15 min |
| 9 | Injection, tint (Class C) | start `attack_vcam.py --mode tint --target <photo>` | `... --camera 1 --camera-name obsvcam --label tint --condition normal` | 10 | 8 min |

**Lighting note (from the pilot):** "normal" means the room lights ON, as in the pilot of 25 Nov (camera holds 30 fps).
With lights off, auto-exposure drops this webcam to about 15 fps, so the dim block will also be a 15 fps block;
that coupling is reported as a finding. Check the printed "fps measured" after each session.

Total about 80 minutes; blocks can be done on different occasions. Blocks 1, 7 and 8 are the core;
if time is short, do those first.

Attack logs: run `attack_vcam.py` with `--log sessions/attack_<mode>_<n>.csv` so the per-frame cost is kept.

After recording:
```
..\..\.venv\Scripts\python.exe extract_lags.py sessions --out ..\revision_out\lags.csv
..\..\.venv\Scripts\python.exe analyze_real.py ..\revision_out\lags.csv --out ..\revision_out\real --rtt ..\revision_out\rtt_wifi_1.1.1.1.txt --attack-logs "sessions\attack_*.csv"
```
