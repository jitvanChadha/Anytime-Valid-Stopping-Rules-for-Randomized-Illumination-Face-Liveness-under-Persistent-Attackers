#!/usr/bin/env python3
"""
attack_vcam.py -- injection-attack pipelines for the hardware validation (run by the attacker side).

Reads the real webcam, transforms every frame, and writes the result to a virtual camera
(OBS Virtual Camera via pyvirtualcam). The verifier (flash_capture.py) then reads the virtual
camera instead of the real one. The attacker sits in front of the verifier's screen, so the flash
lands on the attacker's real face and has to pass through the pipeline before the verifier sees it.

Modes
  passthrough  frames are copied unchanged          -> transport delay of the injection path alone
  swap         real-time face swap (YuNet detector, ArcFace embedding, inswapper_128) on the GPU
               via onnxruntime DirectML             -> transport + synthesis delay  (Class A/B attacker)
  tint         a still image of the target face, tinted with the colour currently on the screen
               (read from the screen buffer)         -> responds without re-synthesis (Class C attacker)

Per-frame processing times are written to <log>.csv (computational overhead, R5).

  python attack_vcam.py --mode swap --target target.jpg --models ../../attack_models
  python attack_vcam.py --mode passthrough
"""
import argparse, os, sys, time
import cv2
import numpy as np

ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

def template(size):
    if size % 112 == 0:
        return ARCFACE_112 * (size / 112.0)
    t = ARCFACE_112 * (size / 128.0)
    t[:, 0] += 8.0 * (size / 128.0)
    return t

def align(img, kps, size):
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), template(size), method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0), M

class Detector:
    def __init__(self, path, w, h):
        self.d = cv2.FaceDetectorYN.create(path, "", (w, h), 0.6, 0.3, 5000)
        self.wh = (w, h)

    def __call__(self, img):
        h, w = img.shape[:2]
        if (w, h) != self.wh:
            self.d.setInputSize((w, h)); self.wh = (w, h)
        _, faces = self.d.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        return f[4:14].reshape(5, 2)

class Swapper:
    def __init__(self, models, providers):
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper
        rec = os.path.join(models, "buffalo_l", "w600k_r50.onnx")
        if not os.path.exists(rec):
            rec = os.path.join(models, "buffalo_l", "buffalo_l", "w600k_r50.onnx")
        swp = os.path.join(models, "inswapper_128.onnx")
        so = ort.SessionOptions()
        self.rec = ort.InferenceSession(rec, so, providers=providers)
        self.swp = ort.InferenceSession(swp, so, providers=providers)
        self.emap = numpy_helper.to_array(onnx.load(swp).graph.initializer[-1])
        self.in_names = [i.name for i in self.swp.get_inputs()]
        self.latent = None
        print("providers:", self.swp.get_providers())

    def embed(self, img, kps):
        a, _ = align(img, kps, 112)
        x = (cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        emb = self.rec.run(None, {self.rec.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
        return emb / np.linalg.norm(emb)

    def set_source(self, img, kps):
        emb = self.embed(img, kps)
        self.src_emb = emb
        lat = emb.reshape(1, -1) @ self.emap
        self.latent = (lat / np.linalg.norm(lat)).astype(np.float32)

    def __call__(self, img, kps):
        a, M = align(img, kps, 128)
        x = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        feeds = {self.in_names[0]: x, self.in_names[1]: self.latent}
        y = self.swp.run(None, feeds)[0][0].transpose(1, 2, 0)
        y = cv2.cvtColor(np.clip(y * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        warped = cv2.warpAffine(y, IM, (w, h), borderValue=0)
        mask = np.zeros((128, 128), np.float32)
        cv2.rectangle(mask, (10, 10), (118, 118), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 6)
        m = cv2.warpAffine(mask, IM, (w, h))[..., None]
        return (warped * m + img * (1 - m)).astype(np.uint8)

def screen_colour():
    import mss
    with mss.mss() as s:
        mon = s.monitors[1]
        box = {"left": mon["left"] + mon["width"] // 2 - 20, "top": mon["top"] + mon["height"] // 2 - 20,
               "width": 40, "height": 40}
        return np.array(s.grab(box))[..., :3].reshape(-1, 3).mean(0)  # BGR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["passthrough", "swap", "tint"], required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--target", help="image of the (consenting) person to impersonate; swap and tint modes")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "attack_models"))
    ap.add_argument("--provider", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--tint-gain", type=float, default=0.35)
    ap.add_argument("--log", default=None)
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until Ctrl+C)")
    a = ap.parse_args()
    import pyvirtualcam

    cap = cv2.VideoCapture(a.camera, cv2.CAP_MSMF)
    cap.set(3, a.width); cap.set(4, a.height); cap.set(cv2.CAP_PROP_FPS, a.fps); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, fr = cap.read()
    for _ in range(50):                       # the camera can take a few seconds to start streaming
        if ok:
            break
        time.sleep(0.2); cap.release()
        cap = cv2.VideoCapture(a.camera, cv2.CAP_MSMF)
        cap.set(3, a.width); cap.set(4, a.height); cap.set(cv2.CAP_PROP_FPS, a.fps); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, fr = cap.read()
    if not ok:
        sys.exit("cannot read camera")
    H, W = fr.shape[:2]
    det = sw = still = None
    models = os.path.abspath(a.models)
    if a.mode in ("swap", "tint"):
        if not a.target:
            sys.exit("--target is required for swap and tint")
        tgt = cv2.imread(a.target)
        if tgt is None:
            sys.exit(f"cannot read {a.target}")
    if a.mode == "swap":
        det = Detector(os.path.join(models, "yunet.onnx"), W, H)
        prov = ["DmlExecutionProvider", "CPUExecutionProvider"] if a.provider == "dml" else ["CPUExecutionProvider"]
        sw = Swapper(models, prov)
        k = Detector(os.path.join(models, "yunet.onnx"), tgt.shape[1], tgt.shape[0])(tgt)
        if k is None:
            sys.exit("no face found in the target image")
        sw.set_source(tgt, k)
    if a.mode == "tint":
        still = cv2.resize(tgt, (W, H)).astype(np.float32)
        dark = screen_colour()
    log = open(a.log or f"attack_{a.mode}_{time.strftime('%Y%m%d_%H%M%S')}.csv", "w")
    log.write("t_grab_ns,t_sent_ns,proc_ms,face,sim_target,sim_self\n")
    t_stop = time.perf_counter() + a.seconds if a.seconds else None
    n = 0
    with pyvirtualcam.Camera(width=W, height=H, fps=a.fps, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs") as vc:
        print(f"virtual camera: {vc.device} {W}x{H}@{a.fps}  mode={a.mode}  (Ctrl+C to stop)")
        try:
            while t_stop is None or time.perf_counter() < t_stop:
                ok, fr = cap.read()
                tg = time.perf_counter_ns()
                if not ok:
                    continue
                face = 1
                sim_t = sim_s = ""
                if a.mode == "swap":
                    kps = det(fr)
                    if kps is None:
                        face = 0; out = fr
                    else:
                        out = sw(fr, kps)
                elif a.mode == "tint":
                    c = screen_colour()
                    out = np.clip(still + a.tint_gain * (c - dark), 0, 255).astype(np.uint8)
                else:
                    out = fr
                ts_proc = time.perf_counter_ns()
                vc.send(out)
                ts = time.perf_counter_ns()
                if a.mode == "swap" and face and n % 15 == 0:      # identity check, outside the timed path
                    ko = det(out)
                    if ko is not None:
                        e_out = sw.embed(out, ko)
                        sim_t = f"{float(e_out @ sw.src_emb):.3f}"
                        sim_s = f"{float(e_out @ sw.embed(fr, kps)):.3f}"
                log.write(f"{tg},{ts},{(ts_proc - tg) / 1e6:.3f},{face},{sim_t},{sim_s}\n")
                n += 1
                if n % 300 == 0:
                    log.flush()
        except KeyboardInterrupt:
            pass
    log.close()
    cap.release()
    print(f"{n} frames")

if __name__ == "__main__":
    main()
