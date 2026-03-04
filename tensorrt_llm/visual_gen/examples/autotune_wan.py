#!/usr/bin/env python3
"""Custom autotune: loop over attn/linear/compile choices, report runtime.
Supports shmoo modes for linear, attention, compile, and frame counts."""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VISUAL_GEN_DIR = SCRIPT_DIR.parent

# All available choices
ALL_ATTN_CHOICES = ["default", "sage-attn", "cudnn-attn", "trtllm-attn", "flash-attn3", "flash-attn3-fp8"]
# after transformer_engine: ["te", "te-fp8"]
# after sparse-videogen: ["sparse-videogen", "sparse-videogen2"]

ALL_LINEAR_CHOICES = ["default", "trtllm-fp8-blockwise", "trtllm-fp8-per-tensor", "torch-ao-fp8"]
# after transformer_engine: ["te-fp8-blockwise", "te-fp8-per-tensor"]

ALL_COMPILE_MODES = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]

ALL_FRAME_COUNTS = [9, 121]

NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

def extract_runtime(output: str) -> str | None:
    m = re.search(r"Inference completed in ([0-9.]+) seconds", output)
    if m:
        return m.group(1)
    m = re.search(r"Time taken:\s*([0-9.]+)\s*s", output)
    if m:
        return m.group(1)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Wan autotune with shmoo support")
    parser.add_argument(
        "--shmoo",
        type=str,
        choices=["linear", "attention", "compile", "frames", "all"],
        default="all",
        help="What to shmoo: linear, attention, compile, frames, or all (default: all)"
    )
    parser.add_argument(
        "--base-attn",
        type=str,
        choices=ALL_ATTN_CHOICES,
        default="default",
        help="Fixed attention type (when not shmooing attn)"
    )
    parser.add_argument(
        "--base-linear",
        type=str,
        choices=ALL_LINEAR_CHOICES,
        default="default",
        help="Fixed linear type (when not shmooing linear)"
    )
    parser.add_argument(
        "--base-compile-mode",
        type=str,
        choices=ALL_COMPILE_MODES,
        default="max-autotune-no-cudagraphs",
        help="Fixed compile mode (when not shmooing compile)"
    )
    parser.add_argument(
        "--base-cuda-graph",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help="Base CUDA graph setting: auto (smart based on compile mode), true, or false (default: auto)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=9,
        help="Fixed frame count (when not shmooing frames). Default: 9"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per run in seconds (default: 600)"
    )
    return parser.parse_args()


def should_enable_cuda_graph(compile_mode, base_cuda_graph_setting):
    """Determine if CUDA graph should be enabled based on compile mode.

    Logic:
    - max-autotune-no-cudagraphs: ALWAYS enable CUDA graph wrapper (designed for this)
    - max-autotune: NEVER enable wrapper (already has cudagraphs built-in)
    - Other modes: Follow base_cuda_graph_setting
    """
    if base_cuda_graph_setting == "true":
        return True
    elif base_cuda_graph_setting == "false":
        return False
    else:  # "auto"
        if compile_mode == "max-autotune-no-cudagraphs":
            return True  # Always enable for this mode
        elif compile_mode == "max-autotune":
            return False  # Don't double-wrap
        else:
            return False  # Default: off for default/reduce-overhead


def build_base_args(num_frames):
    """Build base arguments for wan_t2v.py"""
    return [
        "examples/wan_t2v.py",
        "--model_path", "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--fps", "24",
        "--num_frames", str(num_frames),
        "--height", "704",
        "--width", "1280",
        "--prompt", "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage",
        "--negative_prompt", NEG_PROMPT,
    ]


def run_config(attn, linear, compile_mode, cuda_graph, num_frames, log_dir, timeout):
    """Run a single configuration and return runtime"""
    base_args = build_base_args(num_frames)
    cmd = ["python"] + base_args + [
        "--attn_type", attn,
        "--linear_type", linear,
        "--torch_compile_mode", compile_mode,
    ]

    if cuda_graph:
        cmd.append("--enable_cuda_graph")

    try:
        proc = subprocess.run(
            cmd,
            cwd=VISUAL_GEN_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout + proc.stderr
    except (subprocess.TimeoutExpired, Exception) as e:
        out = str(e)

    # Save log
    log_name = f"attn_{attn}_linear_{linear}_compile_{compile_mode}_cg_{cuda_graph}_frames_{num_frames}.log"
    log_name = log_name.replace("-", "_")
    (log_dir / log_name).write_text(out, encoding="utf-8")

    runtime = extract_runtime(out)
    return runtime if runtime else "FAIL"


def main():
    args = parse_args()

    # Create log directory
    log_dir = VISUAL_GEN_DIR / f"auto_tune_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to vary based on shmoo mode
    if args.shmoo == "linear":
        attn_choices = [args.base_attn]
        linear_choices = ALL_LINEAR_CHOICES
        compile_choices = [args.base_compile_mode]
        frame_choices = [args.frames]
        shmoo_desc = f"Linear shmoo (base_attn={args.base_attn}, base_compile={args.base_compile_mode}, frames={args.frames})"
    elif args.shmoo == "attention":
        attn_choices = ALL_ATTN_CHOICES
        linear_choices = [args.base_linear]
        compile_choices = [args.base_compile_mode]
        frame_choices = [args.frames]
        shmoo_desc = f"Attention shmoo (base_linear={args.base_linear}, base_compile={args.base_compile_mode}, frames={args.frames})"
    elif args.shmoo == "compile":
        attn_choices = [args.base_attn]
        linear_choices = [args.base_linear]
        compile_choices = ALL_COMPILE_MODES
        frame_choices = [args.frames]
        shmoo_desc = f"Compile shmoo (base_attn={args.base_attn}, base_linear={args.base_linear}, frames={args.frames})"
    elif args.shmoo == "frames":
        attn_choices = [args.base_attn]
        linear_choices = [args.base_linear]
        compile_choices = [args.base_compile_mode]
        frame_choices = ALL_FRAME_COUNTS
        shmoo_desc = f"Frame count shmoo (base_attn={args.base_attn}, base_linear={args.base_linear}, base_compile={args.base_compile_mode})"
    else:  # all
        attn_choices = ALL_ATTN_CHOICES
        linear_choices = ALL_LINEAR_CHOICES
        compile_choices = ALL_COMPILE_MODES
        frame_choices = ALL_FRAME_COUNTS
        shmoo_desc = "Full shmoo (all combinations)"

    # Calculate total
    total = len(attn_choices) * len(linear_choices) * len(compile_choices) * len(frame_choices)

    print("=" * 70)
    print(f"Wan Autotune: {shmoo_desc}")
    print("=" * 70)
    print(f"Attention choices: {attn_choices}")
    print(f"Linear choices: {linear_choices}")
    print(f"Compile choices: {compile_choices}")
    print(f"Frame choices: {frame_choices}")
    print(f"CUDA graph mode: {args.base_cuda_graph}")
    print(f"Total runs: {total}")
    print(f"Logs: {log_dir}")
    print("=" * 70)
    print()

    results = {}
    idx = 0

    for frames in frame_choices:
        for attn in attn_choices:
            for linear in linear_choices:
                for compile_mode in compile_choices:
                    idx += 1

                    # Determine CUDA graph setting for this compile mode
                    cuda_graph = should_enable_cuda_graph(compile_mode, args.base_cuda_graph)
                    config = (frames, attn, linear, compile_mode, cuda_graph)

                    print(f"[{idx}/{total}] frames={frames} attn={attn} linear={linear} "
                          f"compile={compile_mode} cuda_graph={cuda_graph} ...")

                    runtime = run_config(attn, linear, compile_mode, cuda_graph, frames, log_dir, args.timeout)
                    results[config] = runtime

                    if runtime != "FAIL":
                        print(f"  -> {runtime}s")
                    else:
                        print(f"  -> FAIL")

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("frames,attn,linear,compile_mode,cuda_graph,runtime_s")

    valid_results = []
    for config, runtime in results.items():
        frames, attn, linear, compile_mode, cuda_graph = config
        val = runtime if runtime != "FAIL" else "-"
        print(f"{frames},{attn},{linear},{compile_mode},{cuda_graph},{val}")

        if runtime != "FAIL":
            try:
                valid_results.append((config, float(runtime)))
            except (TypeError, ValueError):
                pass

    # Print best configuration
    if valid_results:
        print()
        print("=" * 70)
        # Best overall
        best = min(valid_results, key=lambda x: x[1])
        frames, attn, linear, compile_mode, cuda_graph = best[0]
        print(f"Best overall: frames={frames}, attn={attn}, linear={linear}, "
              f"compile={compile_mode}, cuda_graph={cuda_graph}, runtime={best[1]}s")

        # Best per frame count
        for frame_count in frame_choices:
            frame_results = [(c, r) for c, r in valid_results if c[4] == frame_count]
            if frame_results:
                best_frame = min(frame_results, key=lambda x: x[1])
                frames, attn, linear, compile_mode, cuda_graph = best_frame[0]
                print(f"Best for {frames} frames: attn={attn}, linear={linear}, "
                      f"compile={compile_mode}, cuda_graph={cuda_graph}, runtime={best_frame[1]}s")
        print("=" * 70)

if __name__ == "__main__":
    main()
    sys.exit(0)
