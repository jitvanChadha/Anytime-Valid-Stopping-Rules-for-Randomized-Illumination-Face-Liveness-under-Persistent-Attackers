#!/usr/bin/env python3
"""
flash_capture.py -- screen-flash challenge-response capture for the hardware validation.

The laptop screen shows black, then flashes a randomly chosen colour at a random onset time.
A capture thread reads the camera (the real webcam for live / replay sessions, or the virtual
camera fed by the face-swap pipeline for injection sessions) and stamps every frame with the
same monotonic clock (time.perf_counter_ns, QueryPerformanceCounter on Windows) that stamps the
flash. Only the mean BGR colour of the face region of every frame is stored -- no images.

  python flash_capture.py --camera 0 --label live --condition normal --sessions 5
  python flash_capture.py --list-cameras

Output per session (in --out):
  <id>_frames.csv  t_ns, b, g, r, full_b, full_g, full_r   (ROI mean and whole-frame mean)
  <id>_rounds.csv  round, t_flash_ns, t_off_ns, color_idx, color_hex
  <id>_meta.json   camera, fps requested/actual, resolution, label, condition, ROI, palette, notes
Lags are extracted offline by analyze_real.py.
"""
import argparse, json, os, random, secrets, sys, threading, time
from datetime import datetime

import cv2
import numpy as np

PALETTE = [  # BGR for OpenCV; saturated, mutually distinguishable colours
    ("#FF0000", (0, 0, 255)), ("#00FF00", (0, 255, 0)), ("#0000FF", (255, 0, 0)),
    ("#FFFF00", (0, 255, 255)), ("#FF00FF", (255, 0, 255)), ("#00FFFF", (255, 255, 0)),
    ("#FFFFFF", (255, 255, 255)), ("#FF8000", (0, 128, 255)),
]

def keep_display_on(on=True):
    """Stop Windows from blanking the display or locking during capture (SetThreadExecutionState)."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)

def now_ns():
    return time.perf_counter_ns()

class Grabber(threading.Thread):
    def __init__(self, cap, roi):
        super().__init__(daemon=True)
        self.cap, self.roi = cap, roi
        self.rows, self.running, self.last = [], True, None

    def run(self):
        x, y, w, h = self.roi
        while self.running:
            ok, fr = self.cap.read()
            t = now_ns()
            if not ok:
                time.sleep(0.001); continue
            self.last = fr
            rb, rg, rr = fr[y:y + h, x:x + w].reshape(-1, 3).mean(0)
            fb, fg, frr = fr.reshape(-1, 3).mean(0)
            self.rows.append((t, rb, rg, rr, fb, fg, frr))

def find_roi(cap, n_try=60):
    """Face box from the YuNet detector on the first frames; falls back to the frame centre."""
    model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "attack_models", "yunet.onnx")
    det = None
    fr = None
    for _ in range(n_try):
        ok, fr = cap.read()
        if not ok:
            continue
        if det is None:
            det = cv2.FaceDetectorYN.create(model, "", (fr.shape[1], fr.shape[0]), 0.7)
        _, res = det.detect(fr)
        faces = [] if res is None else [r[:4] for r in res if r[2] >= 80]
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # inner part of the box: cheeks, nose and forehead, little hair or background
            return (int(x + 0.2 * w), int(y + 0.15 * h), int(0.6 * w), int(0.7 * h)), True
    H, W = fr.shape[:2] if fr is not None else (480, 640)
    return (int(0.3 * W), int(0.25 * H), int(0.4 * W), int(0.5 * H)), False

def open_camera(idx, fps, width, height, backend):
    api = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}[backend]
    def make():
        c = cv2.VideoCapture(idx, api)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, width)       # format first, then the first read
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            c.set(cv2.CAP_PROP_FPS, fps)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c
    cap = make()
    for _ in range(40):                       # a virtual camera appears only once its feeder runs
        if cap.isOpened() and cap.read()[0]:
            break
        time.sleep(0.25); cap.release(); cap = make()
    if not cap.isOpened():
        sys.exit(f"cannot open camera {idx} with backend {backend}")
    return cap

def measured_fps(rows):
    t = np.array([r[0] for r in rows], dtype=np.int64)
    if len(t) < 10:
        return float("nan")
    return 1e9 / float(np.median(np.diff(t)))

def hud(img, args, rounds_left):
    """Small dim progress text, top right: session k/N, time left in session and in batch."""
    per_round = (args.on_ms + 0.5 * (args.gap_min + args.gap_max)) / 1000.0
    sess_len = args.rounds * per_round + args.warmup + 3.6
    left_sess = rounds_left * per_round
    left_all = left_sess + (args.sessions - args._sess) * sess_len
    fmt = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
    txt = f"Session {args._sess}/{args.sessions}   this session {fmt(left_sess)}   total {fmt(left_all)}"
    out = img.copy()
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(out, txt, (img.shape[1] - w - 30, 30 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 110, 110), 2, cv2.LINE_AA)
    return out

def run_session(args, sid, win):
    cap = open_camera(args.camera, args.fps, args.width, args.height, args.backend)
    if args.manual_exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if args.backend != "msmf" else 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.manual_exposure)
    black0 = np.zeros((args.screen_h, args.screen_w, 3), np.uint8)
    cv2.imshow(win, hud(black0, args, args.rounds)); cv2.waitKey(1)
    roi, face_found = find_roi(cap)
    g = Grabber(cap, roi); g.start()
    end_warm = now_ns() + int(args.warmup * 1e9)
    while now_ns() < end_warm:
        cv2.imshow(win, hud(black0, args, args.rounds))
        if cv2.waitKey(15) & 0xFF == 27:
            g.running = False; cap.release(); return None
    warm_fps = measured_fps(g.rows)
    if args.fps and warm_fps < 0.8 * args.fps and not args.allow_low_fps:
        g.running = False; g.join(1.0); cap.release()
        return {"low_fps": warm_fps}
    rng = secrets.SystemRandom()          # A3: unpredictable challenges
    rounds = []
    for k in range(1, args.rounds + 1):
        gap = rng.uniform(args.gap_min, args.gap_max)
        t_end = now_ns() + int(gap * 1e6)
        black = hud(black0, args, args.rounds - k + 1)
        cv2.imshow(win, black)            # progress text refreshed once per round, in the gap
        while now_ns() < t_end:
            if cv2.waitKey(1) & 0xFF == 27:
                g.running = False; cap.release(); return None
        ci = rng.randrange(len(PALETTE))
        img = np.empty_like(black0); img[:] = PALETTE[ci][1]
        img = hud(img, args, args.rounds - k + 1)
        cv2.imshow(win, img); cv2.waitKey(1)
        t_flash = now_ns()                 # after the present call returned
        t_end = t_flash + int(args.on_ms * 1e6)
        while now_ns() < t_end:
            cv2.waitKey(1)
        black = hud(black0, args, args.rounds - k)
        cv2.imshow(win, black); cv2.waitKey(1)
        t_off = now_ns()
        rounds.append((k, t_flash, t_off, ci, PALETTE[ci][0]))
    t_end = now_ns() + int(0.6e9)
    while now_ns() < t_end:
        cv2.waitKey(1)
    g.running = False; g.join(1.0)
    fps_meas = measured_fps(g.rows)
    iv = np.diff(np.array([r[0] for r in g.rows], dtype=np.int64)) / 1e6
    slow = (iv > 1.6 * 1000.0 / (args.fps or fps_meas)).mean() if len(iv) > 20 else 0.0
    fps_switch = bool(0.05 < slow < 0.95)       # part of the session ran at a slower frame rate
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_rep = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    base = os.path.join(args.out, sid)
    with open(base + "_frames.csv", "w") as f:
        f.write("t_ns,b,g,r,full_b,full_g,full_r\n")
        for r in g.rows:
            f.write(f"{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},{r[6]:.3f}\n")
    with open(base + "_rounds.csv", "w") as f:
        f.write("round,t_flash_ns,t_off_ns,color_idx,color_hex\n")
        for r in rounds:
            f.write(",".join(map(str, r)) + "\n")
    meta = dict(session_id=sid, label=args.label, condition=args.condition, camera=args.camera,
                camera_name=args.camera_name, backend=args.backend, fps_requested=args.fps,
                fps_reported=fps_rep, fps_measured=fps_meas, width=W, height=H, roi=roi,
                face_found=face_found, rounds=args.rounds, on_ms=args.on_ms,
                gap_ms=[args.gap_min, args.gap_max], palette=[p[0] for p in PALETTE],
                manual_exposure=args.manual_exposure, operator=args.operator, notes=args.notes, fps_switch=fps_switch,
                started=datetime.now().isoformat(timespec="seconds"), n_frames=len(g.rows))
    with open(base + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta

def list_cameras(backend):
    api = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}[backend]
    for i in range(6):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            ok, fr = cap.read()
            print(i, "ok" if ok else "opened, no frame", fr.shape if ok else "",
                  "fps", cap.get(cv2.CAP_PROP_FPS))
        cap.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--camera-name", default="builtin")
    ap.add_argument("--backend", default="dshow", choices=["msmf", "dshow", "any"])
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--label", default="live", help="live | replay | faceswap | ...")
    ap.add_argument("--condition", default="normal", help="free text, e.g. bright / normal / dim")
    ap.add_argument("--operator", default="A")
    ap.add_argument("--notes", default="")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--on-ms", type=float, default=600.0)
    ap.add_argument("--gap-min", type=float, default=600.0)
    ap.add_argument("--gap-max", type=float, default=1000.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--manual-exposure", type=float, default=None)
    ap.add_argument("--allow-low-fps", action="store_true", help="record even if the camera runs below 80%% of --fps")
    ap.add_argument("--screen-w", type=int, default=1920)
    ap.add_argument("--screen-h", type=int, default=1080)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions"))
    args = ap.parse_args()
    if args.list_cameras:
        list_cameras(args.backend); return
    os.makedirs(args.out, exist_ok=True)
    keep_display_on(True)
    win = "flash"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for s in range(args.sessions):
        args._sess = s + 1
        sid =f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}_{args.condition}_{args.camera_name}_{int(args.fps)}fps"
        meta = run_session(args, sid, win)
        if meta is None:
            print("aborted"); break
        if "low_fps" in meta:
            print(f"STOPPED: camera runs at {meta['low_fps']:.1f} fps, below the requested {args.fps:.0f}. "
                  "Add light (auto-exposure lowers the frame rate in dim light) and restart the batch. Nothing was saved for this session.")
            break
        print(f"[{s + 1}/{args.sessions}] {sid}: {meta['n_frames']} frames, "
              f"{meta['fps_measured']:.1f} fps measured, face {'found' if meta['face_found'] else 'NOT found'}")
        if meta.get("fps_switch"):
            print("STOPPED: the frame rate changed during this session (light level changed). "
                  "Its files are saved with fps_switch=true and will be excluded. Fix the light, then restart.")
            break
        time.sleep(1.0)
    cv2.destroyAllWindows()
    keep_display_on(False)

if __name__ == "__main__":
    main()
