#!/usr/bin/env python3
"""Configure DeCoMix-HDR dataset paths and local train/test configs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_CHECKPOINT = "./pretrained_models/decomix_hdr_net_g_400000.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create DeCoMix-HDR dataset symlinks, verify LMDB layout, and write local configs."
    )
    parser.add_argument("--hdrtv4k-root", required=True, type=Path, help="Path to prepared HDRTV4K/HDRTVDM root.")
    parser.add_argument("--hdrtv1k-root", required=True, type=Path, help="Path to prepared HDRTV1K root.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path written to the local test config. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--absolute-config-paths",
        action="store_true",
        help="Write absolute LMDB paths into generated local configs instead of relying on datasets/ symlinks.",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Do not create datasets/HDRTV4K and datasets/HDRTV1K symlinks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing dataset symlinks. Non-empty real directories are never removed.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify dataset/checkpoint layout; do not create symlinks or write configs.",
    )
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Fail if the checkpoint path does not exist.",
    )
    parser.add_argument(
        "--skip-local-configs",
        action="store_true",
        help="Do not write options/local/*.yml config copies.",
    )
    return parser.parse_args()


def resolve_existing_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def checkpoint_path(repo_root: Path, checkpoint: str) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def check_file(path: Path, errors: list[str], warnings: list[str], required: bool = True) -> None:
    if path.is_file():
        print(f"[OK] {path}")
    elif required:
        errors.append(f"Missing required file: {path}")
    else:
        warnings.append(f"Missing optional file: {path}")


def check_negative_pair(root: Path, errors: list[str]) -> None:
    luma_full = root / "negatives_luma_fullrank.txt"
    luma_fallback = root / "negatives_luma.txt"
    chroma_full = root / "negatives_chroma_fullrank.txt"
    chroma_fallback = root / "negatives_chroma.txt"

    if luma_full.is_file():
        print(f"[OK] {luma_full}")
    elif luma_fallback.is_file():
        print(f"[OK] {luma_fallback} (fallback)")
    else:
        errors.append(f"Missing luma negatives: {luma_full} or {luma_fallback}")

    if chroma_full.is_file():
        print(f"[OK] {chroma_full}")
    elif chroma_fallback.is_file():
        print(f"[OK] {chroma_fallback} (fallback)")
    else:
        errors.append(f"Missing chroma negatives: {chroma_full} or {chroma_fallback}")


def verify_layout(hdrtv4k_root: Path, hdrtv1k_root: Path, ckpt: Path, require_checkpoint: bool) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    lmdb_dirs = [
        hdrtv4k_root / "trainwtiny.lmdb",
        hdrtv4k_root / "test" / "test.lmdb",
        hdrtv4k_root / "test" / "sample" / "test_sample.lmdb",
        hdrtv1k_root / "test_set_ori.lmdb",
    ]

    for lmdb_dir in lmdb_dirs:
        if not lmdb_dir.is_dir():
            errors.append(f"Missing LMDB directory: {lmdb_dir}")
            continue
        print(f"[OK] {lmdb_dir}")
        check_file(lmdb_dir / "data.mdb", errors, warnings)
        check_file(lmdb_dir / "lock.mdb", errors, warnings)
        check_file(lmdb_dir / "names.pkl", errors, warnings)

    train_lmdb = hdrtv4k_root / "trainwtiny.lmdb"
    if train_lmdb.is_dir():
        check_negative_pair(train_lmdb, errors)

    check_file(ckpt, errors, warnings, required=require_checkpoint)

    for warning in warnings:
        print(f"[WARN] {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)


def ensure_symlink(link_path: Path, target: Path, force: bool) -> None:
    link_parent = link_path.parent
    link_parent.mkdir(parents=True, exist_ok=True)

    if link_path.is_symlink():
        current = link_path.resolve()
        if current == target:
            print(f"[OK] {link_path} -> {target}")
            return
        if not force:
            raise FileExistsError(f"Existing symlink points to {current}; use --force to replace it: {link_path}")
        link_path.unlink()
    elif link_path.exists():
        if not force:
            raise FileExistsError(f"Path already exists; use --force only if it is an empty directory: {link_path}")
        if link_path.is_dir() and not any(link_path.iterdir()):
            link_path.rmdir()
        else:
            raise FileExistsError(f"Refusing to remove non-empty or non-directory path: {link_path}")

    os.symlink(target, link_path, target_is_directory=True)
    print(f"[OK] created {link_path} -> {target}")


def patch_config(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_local_configs(repo_root: Path, hdrtv4k_root: Path, hdrtv1k_root: Path, checkpoint: str, absolute_paths: bool) -> None:
    train_src = repo_root / "options" / "train" / "train_decomix_hdr.yml"
    test_src = repo_root / "options" / "test" / "test_decomix_hdr_fullval.yml"
    out_dir = repo_root / "options" / "local"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_text = train_src.read_text()
    test_text = test_src.read_text()

    replacements = {
        "./experiments/decomix_hdr_main/models/net_g_400000.pth": checkpoint,
        DEFAULT_CHECKPOINT: checkpoint,
    }
    if absolute_paths:
        replacements.update(
            {
                "./datasets/HDRTV4K/trainwtiny.lmdb": str(hdrtv4k_root / "trainwtiny.lmdb"),
                "./datasets/HDRTV4K/test/test.lmdb": str(hdrtv4k_root / "test" / "test.lmdb"),
                "./datasets/HDRTV4K/test/sample/test_sample.lmdb": str(
                    hdrtv4k_root / "test" / "sample" / "test_sample.lmdb"
                ),
                "./datasets/HDRTV1K/test_set_ori.lmdb": str(hdrtv1k_root / "test_set_ori.lmdb"),
            }
        )

    train_out = out_dir / "train_decomix_hdr_local.yml"
    test_out = out_dir / "test_decomix_hdr_fullval_local.yml"
    train_out.write_text(patch_config(train_text, replacements))
    test_out.write_text(patch_config(test_text, replacements))
    print(f"[OK] wrote {train_out}")
    print(f"[OK] wrote {test_out}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    hdrtv4k_root = resolve_existing_dir(args.hdrtv4k_root, "HDRTV4K root")
    hdrtv1k_root = resolve_existing_dir(args.hdrtv1k_root, "HDRTV1K root")
    ckpt = checkpoint_path(repo_root, args.checkpoint)

    verify_layout(hdrtv4k_root, hdrtv1k_root, ckpt, args.require_checkpoint)

    if args.check_only:
        print("[OK] check-only verification passed")
        return

    if not args.no_symlink:
        ensure_symlink(repo_root / "datasets" / "HDRTV4K", hdrtv4k_root, args.force)
        ensure_symlink(repo_root / "datasets" / "HDRTV1K", hdrtv1k_root, args.force)

    if not args.skip_local_configs:
        write_local_configs(repo_root, hdrtv4k_root, hdrtv1k_root, args.checkpoint, args.absolute_config_paths)

    print("\nNext commands:")
    print("  CUDA_VISIBLE_DEVICES=0 python train.py -opt options/local/train_decomix_hdr_local.yml")
    print("  CUDA_VISIBLE_DEVICES=0 python test.py -opt options/local/test_decomix_hdr_fullval_local.yml")


if __name__ == "__main__":
    main()
