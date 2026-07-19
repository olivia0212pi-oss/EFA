from __future__ import annotations


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        message = "PyTorch is not installed. Run: pip install -r requirements-gpu.txt"
        raise SystemExit(message) from exc

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Run this project on an NVIDIA GPU instance.")

    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}")
    print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    if properties.total_memory < 23 * 1024**3:
        print("WARNING: less than 23 GiB VRAM; use quantization or a larger GPU.")
    if not torch.cuda.is_bf16_supported():
        print("WARNING: BF16 is unavailable; change model.dtype to float16.")


if __name__ == "__main__":
    main()
