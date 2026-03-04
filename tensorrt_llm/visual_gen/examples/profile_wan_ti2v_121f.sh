#!/bin/bash
# Profile Wan TI2V 121 frames with different configurations
# Usage: ./profile_wan_ti2v_121f.sh [1|2|3|all]
#   1: Baseline (default settings)
#   2: max-autotune-no-cudagraphs + CUDA graph
#   3: max-autotune-no-cudagraphs + CUDA graph + flash-attn3-fp8 + torch-ao-fp8
#   all: Run all three profiles (default)

COMMON_ARGS=(
  --model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers
  --fps 24
  --num_frames 121
  --height 704
  --width 1280
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
  --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

run_profile_1() {
  echo "========================================"
  echo "Running Profile 1: Baseline (default settings)"
  echo "========================================"
  nsys profile \
    -f true \
    -o 0227_wan_ti2v_121f_baseline \
    --gpu-metrics-devices=all \
    --cuda-graph-trace=node \
    python examples/wan_t2v.py \
    "${COMMON_ARGS[@]}"
  echo "Output: 0227_wan_ti2v_121f_baseline.nsys-rep"
  echo ""
}

run_profile_2() {
  echo "========================================"
  echo "Running Profile 2: max-autotune-no-cudagraphs + CUDA graph"
  echo "========================================"
  nsys profile \
    -f true \
    -o 0227_wan_ti2v_121f_tc-max-autotune-no-cg_cudagraph \
    --gpu-metrics-devices=all \
    --cuda-graph-trace=node \
    python examples/wan_t2v.py \
    "${COMMON_ARGS[@]}" \
    --torch_compile_mode max-autotune-no-cudagraphs \
    --enable_cuda_graph \
    --attn_type default \
    --linear_type default
  echo "Output: 0227_wan_ti2v_121f_tc-max-autotune-no-cg_cudagraph.nsys-rep"
  echo ""
}

run_profile_3() {
  echo "========================================"
  echo "Running Profile 3: max-autotune-no-cudagraphs + CUDA graph + flash-attn3-fp8 + torch-ao-fp8"
  echo "========================================"
  nsys profile \
    -f true \
    -o 0227_wan_ti2v_121f_tc-max-autotune-no-cg_cudagraph_flashattn3fp8_torchao-fp8 \
    --gpu-metrics-devices=all \
    --cuda-graph-trace=node \
    python examples/wan_t2v.py \
    "${COMMON_ARGS[@]}" \
    --torch_compile_mode max-autotune-no-cudagraphs \
    --enable_cuda_graph \
    --attn_type flash-attn3-fp8 \
    --linear_type torch-ao-fp8
  echo "Output: 0227_wan_ti2v_121f_tc-max-autotune-no-cg_cudagraph_flashattn3fp8_torchao-fp8.nsys-rep"
  echo ""
}

# Main script
MODE="${1:-all}"

case "$MODE" in
  1)
    run_profile_1
    ;;
  2)
    run_profile_2
    ;;
  3)
    run_profile_3
    ;;
  all)
    run_profile_1
    run_profile_2
    run_profile_3
    ;;
  *)
    echo "Usage: $0 [1|2|3|all]"
    echo "  1: Baseline (default settings)"
    echo "  2: max-autotune-no-cudagraphs + CUDA graph"
    echo "  3: max-autotune-no-cudagraphs + CUDA graph + flash-attn3-fp8 + torch-ao-fp8"
    echo "  all: Run all three profiles (default)"
    exit 1
    ;;
esac

echo "========================================"
echo "All requested profiles completed!"
echo "========================================"
