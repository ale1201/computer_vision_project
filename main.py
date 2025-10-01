#!/usr/bin/env python3
import argparse, subprocess, sys, re, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

SRC_SEG = HERE / "segmentation_masks.py"
SRC_REC = HERE / "recolor.py"
SRC_MET = HERE / "metrics.py"

def _patch_assignment(txt: str, var: str, value_literal: str) -> str:
    """
    Replace a top-level assignment like:
        VAR = "anything"
    with:
        VAR = <value_literal>
    Uses a function replacement so backslashes are not treated as regex escapes.
    If not found, inserts the assignment at the top of the file.
    """
    pattern = re.compile(rf'^{var}\s*=\s*["\'].*?["\']', flags=re.MULTILINE)
    def repl(_m):
        return f'{var} = {value_literal}'
    new_txt, n = pattern.subn(repl, txt)
    if n == 0:
        # Insert after shebang / encoding if present, else at very top
        lines = txt.splitlines()
        insert_idx = 0
        while insert_idx < len(lines) and (
            lines[insert_idx].startswith("#!") or
            lines[insert_idx].lstrip().startswith("# -*- coding")
        ):
            insert_idx += 1
        lines.insert(insert_idx, f'{var} = {value_literal}')
        new_txt = "\n".join(lines)
    return new_txt

def patch_text(txt: str, project_dir: Path, target_hex: str,
               is_seg: bool, is_recolor: bool, is_metrics: bool) -> str:
    # 1) Ensure PROJECT_DIR is set to the provided path (use raw-string literal in source)
    txt = _patch_assignment(txt, "PROJECT_DIR", f'r"{str(project_dir)}"')

    # 2) Replace target_hex where applicable (recolor + metrics)
    if is_recolor or is_metrics:
        # replacement via function to avoid backslash issues (not needed here, but consistent)
        pattern = re.compile(r'^target_hex\s*=\s*["\']#[0-9A-Fa-f]{6}["\']', flags=re.MULTILINE)
        def repl(_m):
            return f'target_hex = "{target_hex}"'
        txt, _ = pattern.subn(repl, txt)

    # 3) Strip any notebook-style '!pip' lines (mostly relevant for the segmentation export)
    if is_seg:
        lines = []
        for line in txt.splitlines():
            if line.lstrip().startswith("!pip"):
                continue
            lines.append(line)
        txt = "\n".join(lines)

    return txt

def run_script(src: Path, tmp_dir: Path, project_dir: Path, target_hex: str,
               is_seg=False, is_recolor=False, is_metrics=False, verbose=False):
    code = src.read_text(encoding="utf-8")
    patched = patch_text(code, project_dir, target_hex, is_seg, is_recolor, is_metrics)
    out_path = tmp_dir / src.name
    out_path.write_text(patched, encoding="utf-8")

    cmd = [sys.executable, str(out_path)]
    print(f"\n▶ Running: {src.name}")
    if verbose:
        print(f"    cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(
        description="Run full pipeline: Segmentation → Recoloring → Metrics (all stages run)"
    )
    ap.add_argument(
        "--project_dir",
        default=None,
        help="Path to project root (defaults to current working directory)"
    )
    ap.add_argument(
        "--target_hex",
        default="#D32F2F",
        help='Target color for recoloring, e.g., "#D32F2F"'
    )
    ap.add_argument("--verbose", action="store_true", help="Print subprocess commands")
    args = ap.parse_args()

    # Use current working directory if none provided
    project_dir = Path(args.project_dir).resolve() if args.project_dir else Path.cwd()
    if args.project_dir is None:
        print(f"[INFO] --project_dir not provided; using current working directory: {project_dir}")
    if not project_dir.exists():
        raise SystemExit(f"[ERROR] project_dir not found: {project_dir}")

    # Friendly checks
    weights = project_dir / "sam_vit_h_4b8939.pth"
    if not weights.exists():
        print(f"[WARN] SAM weights not found at {weights}. Segmentation will fail without them.")

    images_dir = project_dir / "data" / "images"
    if not images_dir.exists():
        print(f"[WARN] {images_dir} does not exist. Create it and add images before running.")

    # Temp workspace for patched copies
    with tempfile.TemporaryDirectory(prefix="runpatched_") as td:
        tmp_dir = Path(td)

        # Always run all three stages in order
        run_script(SRC_SEG, tmp_dir, project_dir, args.target_hex, is_seg=True,  verbose=args.verbose)
        run_script(SRC_REC, tmp_dir, project_dir, args.target_hex, is_recolor=True, verbose=args.verbose)
        run_script(SRC_MET, tmp_dir, project_dir, args.target_hex, is_metrics=True, verbose=args.verbose)

    print("\n✅ Pipeline finished.")

if __name__ == "__main__":
    main()
