#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Thomas Wagner
#
# This program is free software; you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version. This program is distributed WITHOUT ANY WARRANTY;
# see the file LICENSE for details.
"""Turn COLMAP feature matches into Hugin control points.

Hugin's cpfind is slow on large image sets and its matches can be imprecise. COLMAP's GPU SIFT
and matcher are fast and give sub-pixel keypoints, and its geometric verification (which fits a
homography for rotation-only image pairs) removes most wrong matches. Only COLMAP's first two
stages are used. Its mapper is not, because it needs parallax and a panorama has none.

Workflow:
  colmap feature_extractor --database_path pano.db --image_path imgs --ImageReader.single_camera 1
  colmap exhaustive_matcher --database_path pano.db
  pto_gen -o base.pto imgs/*.jpg                         # Hugin: reads lens data from EXIF
  colmap_to_pto.py pano.db out.pto --base base.pto
  autooptimiser -a -m -l -s -o opt.pto out.pto
  pano_modify --canvas=AUTO --crop=AUTO -o final.pto opt.pto
  hugin_executor --stitching --prefix=pano final.pto

Without --base a minimal project is written (needs --image-dir; FOV from --hfov or COLMAP's prior).
"""
import argparse
import os
import re
import sqlite3
import sys
from collections import defaultdict

import numpy as np

PAIR_ID_FACTOR = 2147483647  # COLMAP: pair_id = id1 * 2147483647 + id2 with id1 < id2
# two_view_geometries.config values that mean "no usable geometry" (COLMAP TwoViewGeometry enum)
BAD_CONFIGS = {0: "undefined", 1: "degenerate", 7: "watermark"}
LENS_TYPES = {"rectilinear": 0, "circular-fisheye": 2, "fullframe-fisheye": 3, "equirectangular": 4}
# lens parameters shared by all images of the same camera (Hugin "=N" links)
LINKED = ["v", "a", "b", "c", "d", "e", "g", "t", "Ra", "Rb", "Rc", "Rd", "Re",
          "Va", "Vb", "Vc", "Vd", "Vx", "Vy"]


def read_database(path):
    """Images, cameras, keypoints and verified matches from a COLMAP database (3.x and 4.x)."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    images = {i: (name, cam) for i, name, cam in db.execute("SELECT image_id, name, camera_id FROM images")}
    cameras = {}
    for cid, w, h, params, prior in db.execute(
            "SELECT camera_id, width, height, params, prior_focal_length FROM cameras"):
        f = float(np.frombuffer(params, np.float64)[0]) if params else None  # first param = focal in every model
        cameras[cid] = dict(width=w, height=h, focal=f, focal_from_exif=bool(prior))
    keypoints = {}
    for i, rows, cols, data in db.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        if rows:  # cols = 2, 4 or 6 depending on version; x, y come first
            keypoints[i] = np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2]
    matches = {}
    for pair_id, rows, data, config in db.execute(
            "SELECT pair_id, rows, data, config FROM two_view_geometries"):
        if not rows or config in BAD_CONFIGS:
            continue
        id2 = pair_id % PAIR_ID_FACTOR
        id1 = (pair_id - id2) // PAIR_ID_FACTOR
        matches[(id1, id2)] = np.frombuffer(data, np.uint32).reshape(rows, 2)
    db.close()
    if not keypoints:
        sys.exit("no keypoints in the database: run 'colmap feature_extractor' first")
    if not matches:
        sys.exit("no verified matches in the database: run a COLMAP matcher first")
    return images, cameras, keypoints, matches


def thin(p1, max_points, grid, width, height, rng):
    """Indices of at most max_points matches, spread over the overlap: one per grid cell first."""
    order = rng.permutation(len(p1))
    if max_points <= 0 or len(p1) <= max_points:
        return order
    cx = np.clip((p1[order, 0] / width * grid).astype(int), 0, grid - 1)
    cy = np.clip((p1[order, 1] / height * grid).astype(int), 0, grid - 1)
    _, first = np.unique(cy * grid + cx, return_index=True)  # first (random) match in each occupied cell
    picked = order[np.sort(first)]
    if len(picked) > max_points:
        picked = picked[np.round(np.linspace(0, len(picked) - 1, max_points)).astype(int)]
    elif len(picked) < max_points:  # fill up with random other matches
        rest = np.setdiff1d(order, picked, assume_unique=True)
        picked = np.concatenate([picked, rng.permutation(rest)[:max_points - len(picked)]])
    return picked


def parse_base(path):
    """The lines of an existing .pto and, per image line, (index, file name, width, height)."""
    lines = open(path, encoding="utf-8").read().splitlines()
    entries = []
    for line in lines:
        if line.startswith("i "):
            name = re.search(r'\bn"([^"]*)"', line)
            w = re.search(r"\sw(\d+)", line)
            h = re.search(r"\sh(\d+)", line)
            if not (name and w and h):
                sys.exit(f"cannot parse image line in {path}: {line}")
            entries.append((len(entries), name.group(1), int(w.group(1)), int(h.group(1))))
    if not entries:
        sys.exit(f"{path} has no image lines ('i ...'); create it with Hugin's pto_gen")
    return lines, entries


def image_line(w, h, lens, hfov, name, link_to):
    fields = {"w": w, "h": h, "f": lens, "v": f"{hfov:.4f}", "Ra": 0, "Rb": 0, "Rc": 0, "Rd": 0, "Re": 0,
              "Eev": 0, "Er": 1, "Eb": 1, "r": 0, "p": 0, "y": 0, "TrX": 0, "TrY": 0, "TrZ": 0, "Tpy": 0,
              "Tpp": 0, "j": 0, "a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "g": 0, "t": 0,
              "Va": 1, "Vb": 0, "Vc": 0, "Vd": 0, "Vx": 0, "Vy": 0, "Vm": 5}
    if link_to is not None:
        for k in LINKED:
            fields[k] = f"={link_to}"
    return "i " + " ".join(f"{k}{v}" for k, v in fields.items()) + f' n"{name}"'


def new_project(images, cameras, args):
    """Minimal .pto: one image line per COLMAP image, lens parameters linked per COLMAP camera."""
    if not args.image_dir:
        sys.exit("without --base, --image-dir is needed (the --image_path given to colmap feature_extractor)")
    lens = LENS_TYPES[args.lens]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    lines = ["# hugin project file (control points from COLMAP, written by colmap_to_pto.py)",
             'p f2 w3000 h1500 v360 R0 n"TIFF_m c:LZW r:CROP"', "m i0", ""]
    entries, first_of_camera = [], {}
    for idx, (iid, (name, cid)) in enumerate(sorted(images.items(), key=lambda kv: kv[1][0])):
        cam = cameras[cid]
        if args.hfov:
            hfov = args.hfov
        else:
            if not cam["focal_from_exif"] and cid not in first_of_camera:
                print(f"WARNING: camera {cid} has no EXIF focal length; COLMAP guessed one. "
                      "Pass --hfov or use --base with a pto_gen project.", file=sys.stderr)
            x = cam["width"] / (2 * cam["focal"])
            hfov = np.degrees(2 * np.arctan(x) if lens == 0 else 2 * x)  # fisheye: equidistant
        path = os.path.relpath(os.path.join(os.path.abspath(args.image_dir), name), out_dir)
        lines.append(image_line(cam["width"], cam["height"], lens, hfov, path.replace(os.sep, "/"),
                                first_of_camera.get(cid)))
        first_of_camera.setdefault(cid, idx)
        entries.append((idx, name, cam["width"], cam["height"]))
    lines.append("")
    return lines, entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("database", help="COLMAP database after feature extraction and matching")
    ap.add_argument("out", help="output .pto")
    ap.add_argument("--base", help="existing .pto (e.g. from pto_gen) to add control points to")
    ap.add_argument("--replace-cps", action="store_true", help="drop control points already in --base")
    ap.add_argument("--image-dir", help="without --base: the folder COLMAP read the images from")
    ap.add_argument("--hfov", type=float, help="without --base: horizontal field of view in degrees")
    ap.add_argument("--lens", choices=LENS_TYPES, default="rectilinear", help="without --base: lens type")
    ap.add_argument("--max-per-pair", type=int, default=40, help="control points per image pair (0 = all)")
    ap.add_argument("--min-matches", type=int, default=20,
                    help="ignore pairs with fewer verified matches (weak pairs are often wrong)")
    ap.add_argument("--grid", type=int, default=8, help="grid cells per image side used to spread the points")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images, cameras, keypoints, matches = read_database(args.database)
    if args.base:
        lines, entries = parse_base(args.base)
        if args.replace_cps:
            lines = [ln for ln in lines if not ln.startswith("c ")]
    else:
        lines, entries = new_project(images, cameras, args)

    # COLMAP image -> .pto image index, matched by file name (COLMAP stores paths relative to --image_path)
    by_base = defaultdict(list)
    for idx, name, w, h in entries:
        by_base[os.path.basename(name)].append((idx, w, h))
    pto_of = {}
    for iid, (name, cid) in images.items():
        hits = by_base.get(os.path.basename(name), [])
        if len(hits) > 1:
            sys.exit(f"file name {os.path.basename(name)} occurs more than once in the project")
        if not hits:
            print(f"WARNING: {name} is not in the project, its matches are skipped", file=sys.stderr)
            continue
        idx, w, h = hits[0]
        cam = cameras[cid]
        if (w, h) != (cam["width"], cam["height"]):
            hint = (" (swapped: EXIF rotation is applied by one program but not the other; "
                    "save the images without rotation tag)") if (w, h) == (cam["height"], cam["width"]) else ""
            sys.exit(f"{name}: size {w}x{h} in the project but {cam['width']}x{cam['height']} in COLMAP{hint}")
        pto_of[iid] = (idx, w, h)

    rng = np.random.default_rng(args.seed)
    cps, used_pairs, skipped = [], 0, 0
    parent = list(range(len(entries)))  # union-find: is the panorama connected?

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (id1, id2), m in sorted(matches.items()):
        if id1 not in pto_of or id2 not in pto_of:
            continue
        if len(m) < args.min_matches:
            skipped += 1
            continue
        (n1, w1, h1), (n2, _, _) = pto_of[id1], pto_of[id2]
        # COLMAP puts the centre of the first pixel at (0.5, 0.5), Hugin at (0, 0)
        p1 = keypoints[id1][m[:, 0]] - 0.5
        p2 = keypoints[id2][m[:, 1]] - 0.5
        for k in thin(p1, args.max_per_pair, args.grid, w1, h1, rng):
            cps.append(f"c n{n1} N{n2} x{p1[k, 0]:.3f} y{p1[k, 1]:.3f} X{p2[k, 0]:.3f} Y{p2[k, 1]:.3f} t0")
        used_pairs += 1
        parent[root(n1)] = root(n2)

    while lines and not lines[-1].strip():
        lines.pop()
    lines += ["", "# control points from COLMAP"] + cps + [""]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    groups = defaultdict(list)
    for idx, name, _, _ in entries:
        groups[root(idx)].append(os.path.basename(name))
    print(f"{len(entries)} images, {used_pairs} pairs ({skipped} below --min-matches), "
          f"{len(cps)} control points -> {args.out}")
    if len(groups) > 1:
        print(f"WARNING: the images fall into {len(groups)} unconnected groups:", file=sys.stderr)
        for g in sorted(groups.values(), key=len, reverse=True):
            print(f"  {len(g)}: {', '.join(sorted(g)[:6])}{' ...' if len(g) > 6 else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
