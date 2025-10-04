#!/usr/bin/env python3
"""
Simple runner to pick and execute benchmark scripts in the benchmarks/ folder.

Usage examples:
  - List available benchmarks: python benchmark_runner.py --list
  - Run resnet50 interactively: python benchmark_runner.py --interactive
  - Run bert non-interactive with flags: python benchmark_runner.py --benchmark bert --flags "--fp16 --seq_len=128"

This script does not import TensorRT or CUDA — it just shells out to the chosen benchmark
script using the same Python interpreter by default.
"""
import os
import sys
import argparse
import subprocess
import shlex
from textwrap import indent


def find_benchmarks(dir_path="benchmarks"):
    if not os.path.isdir(dir_path):
        return {}
    results = {}
    for fn in os.listdir(dir_path):
        if fn.endswith("_benchmark.py") and os.path.isfile(os.path.join(dir_path, fn)):
            name = fn[: -len("_benchmark.py")]
            results[name] = os.path.join(dir_path, fn)
    return dict(sorted(results.items()))


# Known flags per benchmark (lightweight validation / hinting)
KNOWN_FLAGS = {
    "resnet50": ["--lazy", "--fp16"],
    "bert": ["--lazy", "--fp16", "--seq_len"],
}


def validate_flags(bench_name, tokens):
    """Basic validation of flags we will forward to the benchmark.

    tokens is a list as returned by shlex.split; this accepts '--seq_len 128' or '--seq_len=128'.
    """
    allowed = KNOWN_FLAGS.get(bench_name, [])
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            return False, f"Unexpected token: {t}"

        if "=" in t:
            key, val = t.split("=", 1)
        else:
            key = t
            val = None

        if key not in allowed:
            return False, f"Flag {key} not in allowed flags for '{bench_name}': {allowed}"

        # if the flag expects a value (seq_len), ensure a value exists
        if key == "--seq_len":
            if val is None:
                # next token must be a number
                i += 1
                if i >= len(tokens):
                    return False, "--seq_len requires a value"
                try:
                    int(tokens[i])
                except Exception:
                    return False, "--seq_len requires an integer value"
        i += 1

    return True, ""


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Benchmark runner: choose and run a benchmark script")
    parser.add_argument("--list", action="store_true", help="List available benchmarks and exit")
    parser.add_argument("--interactive", action="store_true", help="Interactively choose a benchmark and flags")
    parser.add_argument("--benchmark", type=str, help="Benchmark to run (name of the script without _benchmark.py)")
    parser.add_argument(
        "--flags", nargs=argparse.REMAINDER, default=[], help="Flags to forward to the benchmark script"
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use for running the benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print the command that would be run and exit")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to write benchmark logs into")
    parser.add_argument("--parse-only", type=str, help="Parse an existing log file and print the condensed results")

    args = parser.parse_args(argv)

    benches = find_benchmarks()

    def parse_log_file(log_path):
        import re
        avg_re = re.compile(r"Average inference time over\s*(\d+)\s*iters:\s*([0-9.]+)\s*ms")
        sample_re = re.compile(r"Sample output shape:\s*(\S.*)\s+dtype:\s*(\S+)")

        avg_line = None
        sample_line = None
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if avg_line is None:
                        m = avg_re.search(line)
                        if m:
                            avg_line = f"Average inference time over {m.group(1)} iters: {m.group(2)} ms"
                    if sample_line is None:
                        m2 = sample_re.search(line)
                        if m2:
                            sample_line = f"Sample output shape: {m2.group(1)} dtype: {m2.group(2)}"
                    if avg_line and sample_line:
                        break
        except FileNotFoundError:
            print(f"Log file not found: {log_path}")
            return 2

        # Print only the condensed results (or indicate missing parts)
        if avg_line:
            print(avg_line)
        else:
            print("Average inference time: not found in log")

        if sample_line:
            print(sample_line)
        else:
            print("Sample output shape/dtype: not found in log")

        return 0

    # If parse-only was requested, do it and exit
    if args.parse_only:
        sys.exit(parse_log_file(args.parse_only))

    if args.list:
        if not benches:
            print("No benchmarks found in 'benchmarks/'")
            return 0
        print("Available benchmarks:")
        for name, path in benches.items():
            hints = KNOWN_FLAGS.get(name, [])
            print(f"  {name}\t-> {path}\n    supported flags: {hints}")
        return 0

    bench_name = args.benchmark
    flags_input = " ".join(args.flags).strip()

    if args.interactive or not bench_name:
        # Interactive selection
        if not benches:
            print("No benchmarks found in 'benchmarks/'")
            return 1
        print("Select a benchmark to run:")
        for i, name in enumerate(benches.keys(), start=1):
            print(f"  {i}) {name}")

        choice = None
        while choice is None:
            resp = input("Enter number or name (or 'q' to quit): ").strip()
            if resp.lower() in ("q", "quit", "exit"):
                print("Canceled")
                return 0
            if resp.isdigit():
                idx = int(resp) - 1
                keys = list(benches.keys())
                if 0 <= idx < len(keys):
                    choice = keys[idx]
                    break
                else:
                    print("Invalid number")
                    continue
            if resp in benches:
                choice = resp
                break
            print("Invalid selection")

        bench_name = choice

        # Ask for flags
        hint = KNOWN_FLAGS.get(bench_name, [])
        if hint:
            print(f"Supported flags for {bench_name}: {hint}")
        flags_input = input("Enter optional flags to forward (space separated, include '--'): ").strip()

    if bench_name not in benches:
        print(f"Benchmark '{bench_name}' not found. Use --list to see available benchmarks.")
        return 2

    # Parse flags
    try:
        tokens = shlex.split(flags_input)
    except Exception as e:
        print("Failed to parse flags:", e)
        return 3

    # Normalize tokens: allow users to write 'lazy' or 'fp16' instead of '--lazy' or '--fp16'
    def normalize_tokens(tokens, bench_name):
        allowed = KNOWN_FLAGS.get(bench_name, [])
        allowed_short = set(k.lstrip("-") for k in allowed)
        out = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("--"):
                out.append(t)
            elif t.startswith("-"):
                # single dash -> convert to double dash
                out.append("--" + t.lstrip("-"))
            elif "=" in t:
                key, val = t.split("=", 1)
                if not key.startswith("--"):
                    key = "--" + key
                out.append(f"{key}={val}")
            elif t.isdigit():
                # numeric token: treat as a value
                out.append(t)
            elif t in allowed_short:
                out.append("--" + t)
            else:
                # Heuristic: if next token looks like a value, treat this as a flag name
                if i + 1 < len(tokens) and (tokens[i + 1].isdigit() or "=" in tokens[i + 1]):
                    out.append("--" + t)
                else:
                    # Default to prefixing so users can type 'fp16' friendly
                    out.append("--" + t)
            i += 1
        return out

    norm_tokens = normalize_tokens(tokens, bench_name)

    # Only forward tokens that belong to the benchmark for validation
    bench_tokens = [t for t in norm_tokens if t in KNOWN_FLAGS.get(bench_name, []) or any(t.startswith(k+'=') for k in KNOWN_FLAGS.get(bench_name, []))]
    ok, msg = validate_flags(bench_name, bench_tokens)
    if not ok:
        print("Flag validation failed:", msg)
        print("If you want to bypass validation, re-run with the exact flags using --flags '...' or edit this runner.")
        return 4

    script_path = benches[bench_name]
    cmd = [args.python, script_path] + tokens

    print("Running:")
    print(indent(" ".join(shlex.quote(x) for x in cmd), "  "))

    # Ensure log dir exists
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{bench_name}_{timestamp}.log")

    if args.dry_run:
        print("Dry run: command not executed.")
        print(f"Log would be written to: {log_path}")
        return 0

    # Run benchmark and capture output to log file
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            # echo the command at top of the log
            lf.write("# Command: " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            try:
                ret = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                print("Interrupted by user")
                return 130

        print(f"Benchmark finished with exit code {ret}. Log written to: {log_path}")

        # Parse the log and print condensed results
        parse_rc = parse_log_file(log_path)
        if parse_rc != 0:
            print("Parsing log failed with code", parse_rc)

        return ret
    except FileNotFoundError:
        print(f"Failed to execute {args.python}: executable not found")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
