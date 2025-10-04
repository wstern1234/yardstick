import os
import argparse
import tensorrt as trt
import numpy as np
import time
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--lazy", action="store_true", help="Enable CUDA lazy loading")
parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode if supported")
args = parser.parse_args()

if args.lazy:
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, fixed_input_shape=(1, 3, 224, 224), use_fp16=True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        config = builder.create_builder_config()
        # recommended way to set workspace in modern TRT
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MB

        # only enable FP16 if platform supports it and user asked for it
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Create an optimization profile for any dynamic inputs
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            shape = tuple(input_tensor.shape)  # may contain -1
            # Build min/opt/max shapes, replace -1 with values from fixed_input_shape where reasonable
            # If model has multiple inputs, try to use fixed_input_shape for the first input and copy dims for others
            def replace_dynamic(s, fallback):
                return tuple((fallback[j] if dim == -1 else dim) for j, dim in enumerate(s))

            if i == 0 and len(shape) == len(fixed_input_shape):
                min_shape = replace_dynamic(shape, fixed_input_shape)
                opt_shape = min_shape
                max_shape = min_shape
            else:
                # fallback is to replace all -1 with 1 for min, opt, and 2 for max where unknown
                min_shape = tuple((1 if d == -1 else d) for d in shape)
                opt_shape = tuple((max(1, d) if d == -1 else d) for d in shape)
                max_shape = tuple((2 if d == -1 else d) for d in shape)

            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)

        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        with trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

def dtype_for_binding(engine, idx):
    # returns numpy dtype for binding index
    name = engine.get_tensor_name(idx)
    trt_dtype = engine.get_tensor_dtype(name)
    return trt.nptype(trt_dtype)

def size_from_shape(shape):
    # compute element count from shape tuple-like
    prod = 1
    for s in shape:
        prod *= s
    return prod

def benchmark(engine, iterations=20, batch_size=1):
    context = engine.create_execution_context()

    # Determine bindings and which are inputs
    num_bindings = engine.num_bindings
    tensor_names = [engine.get_tensor_name(i) for i in range(num_bindings)]
    is_input = [engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tensor_names]
    # choose first input index for setting concrete shape
    input_indices = [i for i, b in enumerate(is_input) if b]
    if not input_indices:
        raise RuntimeError("No input bindings found")

    # For simplicity set concrete shape for all input bindings using context.set_binding_shape
    for i in input_indices:
        # if input shape is dynamic, we must set it
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        # shape may contain -1, so create a concrete shape, prefer (batch_size, 3, 224, 224) for first input
        if any([s == -1 for s in shape]):
            if i == input_indices[0] and len(shape) == 4:
                concrete = (batch_size, 3, 224, 224)
            else:
                # fallback, replace -1 with 1
                concrete = tuple((batch_size if j == 0 else (1 if s == -1 else s)) for j, s in enumerate(shape))
            context.set_binding_shape(i, concrete)
        else:
            # already concrete, still set to ensure context consistency (no-op)
            context.set_input_shape(name, tuple(shape))

    # After setting shapes, query concrete shapes for all bindings
    concrete_shapes = [tuple(context.get_tensor_shape(name)) for name in tensor_names]
    # Prepare host/device buffers for every binding in engine order
    host_buffers = [None] * num_bindings
    device_buffers = [None] * num_bindings

    for i in range(num_bindings):
        shape = concrete_shapes[i]
        dtype = dtype_for_binding(engine, i)
        el_count = size_from_shape(shape)
        # create host array, for inputs we populate random data, for outputs we empty-allocate
        if is_input[i]:
            if dtype == np.int32:
                host = np.random.randint(0, 255, size=shape, dtype=np.int32)
            else:
                host = np.random.random(shape).astype(dtype)
        else:
            host = np.empty(shape, dtype=dtype)

        # allocate device memory of required bytes
        device = cuda.mem_alloc(host.nbytes)

        host_buffers[i] = host
        device_buffers[i] = device

    # copy inputs to device
    for i in range(num_bindings):
        if is_input[i]:
            cuda.memcpy_htod(device_buffers[i], host_buffers[i])

    # Build bindings list (device pointers as ints) in engine binding order
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

    # Copy first output back to host for quick sanity check (find first output index)
    out_idx = next((i for i in range(num_bindings) if not is_input[i]), None)
    if out_idx is not None:
        cuda.memcpy_dtoh(host_buffers[out_idx], device_buffers[out_idx])
        print("Sample output shape:", host_buffers[out_idx].shape, "dtype:", host_buffers[out_idx].dtype)

    # memory safety, free device buffers explicitly
    for buf in device_buffers:
        buf.free()

    # release TRT objects deterministically
    del context
    del engine

if __name__ == "__main__":
    model_path = os.path.join("models", "resnet50.onnx")
    if not os.path.exists(model_path):
        print("Model not found at", model_path)
        sys.exit(1)

    print("Building engine (may take a bit)...")
    engine = build_engine(
        model_path,
        fixed_input_shape=(1,3,224,224),
        use_fp16=args.fp16
    )
    if engine is None:
        print("Failed to build engine")
        sys.exit(1)

    print("Running benchmark...")
    benchmark(engine, iterations=20, batch_size=1)
