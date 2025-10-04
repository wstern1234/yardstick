# benchmarks/bert_benchmark.py (robust, dynamic-safe)
import os
import argparse
import tensorrt as trt
import numpy as np
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--lazy", action="store_true", help="Enable CUDA lazy loading")
parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference")
parser.add_argument("--seq_len", type=int, default=128, help="Fixed sequence length for BERT inputs")
args = parser.parse_args()

if args.lazy:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import pycuda.driver as cuda
import pycuda.autoinit

# Auto-download model if not present
def ensure_bert_onnx(model_path="models/bert.onnx"):
    if os.path.exists(model_path):
        return
    print(f"{model_path} not found. Downloading from Hugging Face (Xet-aware)...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Run `pip install huggingface_hub[hf_xet]`.")

    # You can pass `cache_dir` or other parameters if needed
    local = hf_hub_download(
        repo_id="google-bert/bert-base-uncased",
        filename="model.onnx",
        local_dir=".",             # put in current directory
        local_dir_use_symlinks=False,
    )
    # Move or rename from cache location to target path if needed
    if os.path.abspath(local) != os.path.abspath(model_path):
        os.replace(local, model_path)
    print("Download complete:", model_path)

# Call it early in your script
ensure_bert_onnx("models/bert.onnx")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, batch_size=1, seq_len=128, use_fp16=True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MB

        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Optimization profile
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            shape = tuple(input_tensor.shape)

            # Replace -1 with batch_size or seq_len
            min_shape = tuple((batch_size if j==0 else seq_len if s==-1 else s) for j, s in enumerate(shape))
            opt_shape = min_shape
            max_shape = min_shape
            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)

        config.add_optimization_profile(profile)
        serialized_engine = builder.build_serialized_network(network, config)

        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

def dtype_for_binding(engine, idx):
    name = engine.get_tensor_name(idx)
    trt_dtype = engine.get_tensor_dtype(name)
    return trt.nptype(trt_dtype)

def size_from_shape(shape):
    prod = 1
    for s in shape:
        prod *= s
    return prod

def benchmark(engine, iterations=20, batch_size=1, seq_len=128):
    context = engine.create_execution_context()

    # Get binding info
    num_bindings = engine.num_bindings
    tensor_names = [engine.get_tensor_name(i) for i in range(num_bindings)]
    is_input = [engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tensor_names]
    input_indices = [i for i, b in enumerate(is_input) if b]

    # Set concrete shapes
    # Set concrete shapes using the recommended API
    for i in input_indices:
        name = tensor_names[i]
        shape = tuple(context.get_tensor_shape(name))  # <-- new API
        # Replace -1 with batch_size or seq_len
        concrete = tuple((batch_size if j == 0 else seq_len if s == -1 else s) for j, s in enumerate(shape))
        context.set_input_shape(name, concrete)  # <-- new API

    concrete_shapes = [tuple(context.get_tensor_shape(name)) for name in tensor_names]

    # Allocate host/device buffers
    host_buffers = [None] * num_bindings
    device_buffers = [None] * num_bindings
    for i in range(num_bindings):
        shape = concrete_shapes[i]
        dtype = dtype_for_binding(engine, i)
        el_count = size_from_shape(shape)
        if is_input[i]:
            if dtype == np.int32:
                host = np.random.randint(0, 1000, size=shape, dtype=np.int32)
            else:
                host = np.random.random(shape).astype(dtype)
        else:
            host = np.empty(shape, dtype=dtype)
        device = cuda.mem_alloc(host.nbytes)
        host_buffers[i] = host
        device_buffers[i] = device

    # Copy inputs to device
    for i in input_indices:
        cuda.memcpy_htod(device_buffers[i], host_buffers[i])

    # Bindings for TRT
    bindings = [int(device_buffers[i]) for i in range(num_bindings)]

    # Warmup
    for _ in range(3):
        context.execute_v2(bindings)

    # Timed runs
    start = time.time()
    for _ in range(iterations):
        context.execute_v2(bindings)
    end = time.time()

    avg_ms = (end - start) / iterations * 1000.0
    print(f"Average inference time over {iterations} iters: {avg_ms:.2f} ms")

    # Copy first output back to host for sanity check
    out_idx = next((i for i in range(num_bindings) if not is_input[i]), None)
    if out_idx is not None:
        cuda.memcpy_dtoh(host_buffers[out_idx], device_buffers[out_idx])
        print("Sample output shape:", host_buffers[out_idx].shape, "dtype:", host_buffers[out_idx].dtype)

if __name__ == "__main__":
    model_path = os.path.join("models", "bert.onnx")
    if not os.path.exists(model_path):
        print("Model not found at", model_path)
        sys.exit(1)

    print("Building engine (may take a bit)...")
    engine = build_engine(model_path, batch_size=1, seq_len=args.seq_len, use_fp16=args.fp16)
    if engine is None:
        print("Failed to build engine")
        sys.exit(1)

    print("Running benchmark...")
    benchmark(engine, iterations=20, batch_size=1, seq_len=args.seq_len)
