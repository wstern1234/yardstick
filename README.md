# 🤖📏 Yardstick Benchmarks

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![GPU Available](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Latest Benchmark](https://img.shields.io/badge/latest_benchmark-passed-brightgreen.svg)](#logs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**High-performance AI model benchmarking with TensorRT**

Yardstick provides a modular, flexible system for benchmarking deep learning models (currently **ResNet50** and **BERT**) on NVIDIA GPUs using TensorRT. It supports automated engine building, FP16 acceleration, sequence-length customization for transformers, and logging for performance analysis. This repository demonstrates end-to-end handling of AI models from ONNX conversion to high-throughput GPU inference benchmarking.

---

## Table of Contents

* [Features](#features)
* [Benchmarks](#benchmarks)
* [Setup](#setup)
* [Usage](#usage)
* [Implementation Notes](#implementation-notes)
* [Challenges & Solutions](#challenges--solutions)
* [Project Highlights](#project-highlights)

---

## Features

* **Flexible Benchmark Runner**:

  * `benchmark_runner.py` allows you to select and run benchmarks interactively or with CLI flags.
  * Supports custom Python executables, dry-run mode, and log parsing.

* **ResNet50 and BERT Benchmarks**:

  * Handles large transformer models with variable input shapes.
  * Optional **FP16 precision** for faster inference.
  * Sequence-length customization for transformer inputs.

* **Engine Management**:

  * Automatic ONNX-to-TensorRT engine conversion.
  * Lazy building of engines to save time during experimentation.

* **Logging and Results Parsing**:

  * Structured logs stored per benchmark run.
  * Auto-parsing of average inference time and sample output shapes.

* **Large Model Management**:

  * Supports downloading large models (e.g., `bert.onnx` > 500 MB) from Hugging Face directly, including integration with **Xet** for faster chunked downloads.

---

## Benchmarks

Currently included:

| Benchmark  | Description                                  | Flags                           |
| ---------- | -------------------------------------------- | ------------------------------- |
| `resnet50` | Image classification model (ONNX → TensorRT) | `--lazy`, `--fp16`              |
| `bert`     | Transformer-based NLP model (BERT-base)      | `--lazy`, `--fp16`, `--seq_len` |

> Additional benchmarks can be added by creating a `*_benchmark.py` in the `benchmarks/` folder and registering supported flags in `benchmark_runner.py`.

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/wstern1234/yardstick.git
cd yardstick
```

2. **Create a Python virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download models**

* ResNet50 is included directly (91 MB).
* BERT is too large for GitHub; the benchmark will fetch it automatically via Xet:

```bash
python benchmarks/bert_benchmark.py --lazy
```

---

## Usage

### List available benchmarks

```bash
python benchmark_runner.py --list
```

### Run a benchmark interactively

```bash
python benchmark_runner.py --interactive
```

Follow prompts to select a benchmark and optional flags.

### Run a benchmark with flags

```bash
python benchmark_runner.py --benchmark bert --flags "--fp16 --seq_len=128"
```

### Parse an existing log

```bash
python benchmark_runner.py --parse-only logs/bert_20251003_023015.log
```

### Dry-run mode

Prints the exact command without executing:

```bash
python benchmark_runner.py --benchmark resnet50 --flags "--fp16" --dry-run
```

---

## Implementation Notes

* **Dynamic ONNX Inputs**: Handles `-1` dynamic dimensions for batch size and sequence length.
* **FP16 Conversion**: TensorRT engine supports optional FP16 precision; warnings are surfaced for subnormal values and potential layernorm overflows.
* **Large File Handling**: Xet integration allows download of models >500 MB efficiently, avoiding GitHub LFS limitations.
* **Logging**: Each benchmark run logs all stdout and stderr to `logs/<benchmark>_<timestamp>.log` with parsing for key metrics.

---

## Challenges & Solutions

* **Dynamic shapes in transformers**: Original BERT ONNX model had `-1` dimensions. Solved by setting concrete shapes per run using TensorRT's `set_binding_shape`.
* **FP16 stability warnings**: Warnings from layernorm in FP16 were handled by optional FP32 execution of sensitive layers for numerical stability.
* **Large file storage**: BERT ONNX is too large for GitHub. Implemented on-the-fly download via Hugging Face + Xet for reproducibility.
* **Cross-platform script execution**: `benchmark_runner.py` allows consistent benchmark invocation across environments, abstracting Python executable and flags.

---

## Project Highlights

* Demonstrates **AI model deployment and benchmarking skills** using GPU acceleration and TensorRT.
* Showcases **software engineering best practices**: modularity, logging, CLI usability, and error handling.
* Handles **real-world AI/robotics model challenges**: large model size, numerical precision, dynamic input shapes, and reproducible performance testing.
* Illustrates ability to **bridge AI research and engineering**, valuable for frontier AI & robotics development.

---

## Future Work

* Add additional benchmarks for other transformer and vision models.
* Extend FP16 auto-casting with mixed-precision safeguards.
* Web dashboard for visualizing benchmark performance across models and hardware.