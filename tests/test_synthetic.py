#!/usr/bin/env python3
"""End-to-end test on a synthetic rotation-only panorama.

Renders 8 views (2 rows) of a random texture with known camera rotations, runs COLMAP feature
extraction and matching, converts with colmap_to_pto.py and checks every control point against
the true homography. If Hugin's autooptimiser is on PATH, the optimised yaw/pitch are checked too.

Needs: colmap, numpy, opencv-python.   Usage: python tests/test_synthetic.py [workdir]
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W, H, F = 800, 600, 700.0                  # views: pinhole, ~59 deg horizontal FOV
SW, SH, SF = 4000, 3000, 1500.0            # source "camera" that the views are rendered from
YAWS, PITCHES = (-30, -12, 6, 24), (-8, 8)


def K(w, h, f):
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])


def rot(yaw, pitch):  # world -> camera; yaw > 0 turns the camera right, pitch > 0 turns it up
    y, p = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    return rx @ ry


def texture(rng):
    img = (rng.random((SH // 8, SW // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (SW, SH), interpolation=cv2.INTER_CUBIC)
    for _ in range(3000):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y, r = int(rng.integers(0, SW)), int(rng.integers(0, SH)), int(rng.integers(4, 40))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(rng.integers(4, 40))), c, -1)
    return img


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="c2p_test_"))
    imgs = work / "imgs"
    shutil.rmtree(work / "imgs", ignore_errors=True)
    imgs.mkdir(parents=True)
    rng = np.random.default_rng(1)
    src = texture(rng)
    Kv, Ks = K(W, H, F), K(SW, SH, SF)
    views = []
    for pitch in PITCHES:
        for yaw in YAWS:
            R = rot(yaw, pitch)
            M = Ks @ R.T @ np.linalg.inv(Kv)  # view pixel -> source pixel
            name = f"view_{len(views):02d}.png"
            cv2.imwrite(str(imgs / name), cv2.warpPerspective(src, M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))
            views.append((name, yaw, pitch, R))

    db = work / "pano.db"
    db.unlink(missing_ok=True)
    run = lambda *c: subprocess.run([str(x) for x in c], check=True, capture_output=True, text=True)
    run("colmap", "feature_extractor", "--database_path", db, "--image_path", imgs,
        "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL")
    run("colmap", "exhaustive_matcher", "--database_path", db)
    hfov = np.degrees(2 * np.arctan(W / 2 / F))
    out = work / "out.pto"
    r = run(sys.executable, ROOT / "colmap_to_pto.py", db, out, "--image-dir", imgs, "--hfov", hfov)
    print(r.stdout.strip(), r.stderr.strip())

    # every control point must agree with the true homography between the two views
    pto = out.read_text().splitlines()
    cps = [ln for ln in pto if ln.startswith("c ")]
    num = lambda k, ln: float(re.search(rf"\s{k}(-?[\d.]+)", ln).group(1))
    errs = []
    for ln in cps:
        i, j = int(num("n", ln)), int(num("N", ln))
        Hij = Kv @ views[j][3] @ views[i][3].T @ np.linalg.inv(Kv)
        p = Hij @ [num("x", ln), num("y", ln), 1]
        errs.append(np.hypot(p[0] / p[2] - num("X", ln), p[1] / p[2] - num("Y", ln)))
    errs = np.array(errs)
    print(f"{len(cps)} control points, error vs. true homography: median {np.median(errs):.3f} px, "
          f"95% {np.percentile(errs, 95):.3f} px, max {errs.max():.3f} px")
    assert len(cps) >= 10 * len(YAWS), "too few control points"
    assert np.median(errs) < 0.3 and np.percentile(errs, 95) < 1.0, "control points do not fit the geometry"

    if shutil.which("autooptimiser"):
        opt = work / "opt.pto"
        run("autooptimiser", "-p", "-o", opt, out)  # positions only (FOV is known)
        # Hugin anchors image 0 at y = p = r = 0, so compare anchor-free quantities:
        # the angles between all pairs of optical axes (Hugin: yaw = longitude, pitch = latitude)
        yp = np.radians([[num("y", ln), num("p", ln)] for ln in opt.read_text().splitlines() if ln.startswith("i ")])
        axes = np.stack([np.cos(yp[:, 1]) * np.sin(yp[:, 0]), np.sin(yp[:, 1]), np.cos(yp[:, 1]) * np.cos(yp[:, 0])], 1)
        true_axes = np.stack([v[3].T @ [0, 0, 1] for v in views])
        ang = lambda a: np.degrees(np.arccos(np.clip(a @ a.T, -1, 1)))
        err = np.abs(ang(axes) - ang(true_axes)).max()
        print(f"Hugin autooptimiser: max error of the angles between views {err:.3f} deg")
        assert err < 0.05, "optimised camera directions differ from the truth"
    else:
        print("autooptimiser not found: Hugin check skipped")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
