# colmap-to-pto

Use [COLMAP](https://colmap.github.io/)'s feature matching as the control-point finder for
[Hugin](https://hugin.sourceforge.io/) panoramas.

Hugin's `cpfind` can be slow on many images, and its matches can be imprecise. COLMAP extracts SIFT
features on the GPU with sub-pixel accuracy, matches quickly, and checks each image pair
geometrically: for rotation-only pairs it fits a homography and keeps only the matches that fit
it. `colmap_to_pto.py` reads those verified matches from COLMAP's database and writes them as
control points into a Hugin project. Hugin then optimises, blends and renders as usual.

Only COLMAP's feature extraction and matching are used. Its 3D reconstruction (`mapper`) needs
camera translation (parallax), which a well-shot panorama does not have.

## Requirements

- COLMAP (tested with 4.2; the database layout of 3.x is also supported)
- Python ≥ 3.8 with numpy
- Hugin (`pto_gen`, `autooptimiser`, `pano_modify`, `hugin_executor`)

## Usage

```sh
colmap feature_extractor --database_path pano.db --image_path imgs \
    --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL
colmap exhaustive_matcher --database_path pano.db    # many images in shooting order: sequential_matcher

pto_gen -o base.pto imgs/*.jpg                      # Hugin reads lens and FOV from EXIF
python colmap_to_pto.py pano.db cp.pto --base base.pto

autooptimiser -a -m -l -s -o opt.pto cp.pto         # positions, lens, photometrics, horizon, output size
pano_modify --canvas=AUTO --crop=AUTO -o final.pto opt.pto
hugin_executor --stitching --prefix=pano final.pto  # or open final.pto in Hugin and check it first
```

Notes:
- `--ImageReader.single_camera 1` assumes that all images come from the same camera and lens at the
  same zoom. Leave it out for mixed images.
- Without `--base`, a minimal project is written. It needs `--image-dir` (the folder passed to
  COLMAP) and the field of view, from `--hfov DEG` or COLMAP's focal length (EXIF, otherwise a
  guess). Use `--lens fullframe-fisheye` or `circular-fisheye` for fisheye lenses.
- `--max-per-pair` (default 40) keeps that many control points per image pair, spread over the
  overlap on a `--grid` × `--grid` raster. Thousands of points only slow the optimiser down.
- `--min-matches` (default 20) skips pairs with few verified matches. Such pairs are often
  repetitive structures matched at the wrong place.
- `--replace-cps` removes control points that are already in `--base`.
- Images are matched to the project by file name. Width and height must agree: if one program
  applies the EXIF rotation and the other does not, the script stops. Save the images without a
  rotation tag in that case.
- The script warns when the images fall into unconnected groups. Then add images with more overlap,
  or run `colmap exhaustive_matcher` if a sequential matcher was used.
- `cpclean` and `celeste_standalone` from Hugin can still be applied to the result.

## Test

```sh
python tests/test_synthetic.py
```

The test renders 8 views (2 rows) of a random texture with known rotations, runs COLMAP, and
checks every control point against the true homography (median error < 0.3 px). If
`autooptimiser` is on `PATH`, it also checks the optimised camera directions (< 0.05°).
It needs `colmap`, numpy and opencv-python.
