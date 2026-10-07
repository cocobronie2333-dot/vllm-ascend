# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare HOST_UVA FP8 lookup source snapshots on one NPU.

Run from /workspace with --candidate and --baseline pointing at npu.py files.
The optional --uncapped snapshot should precede the eight-program limit.
Use --id-batches to rotate inputs without adding copies to the timed graph;
this expands the working set but does not guarantee cold caches.
"""

import argparse
import importlib.util
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(functions, iterations):
    for index in range(5):
        functions[index % len(functions)]()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for index in range(iterations):
            functions[index % len(functions)]()
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    return graph


def measure(graph, iterations):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--uncapped")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 17, 128, 384])
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--width", type=int, choices=[256], default=256)
    parser.add_argument("--table-rows", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--id-batches", type=int, default=1, help="Rotate preallocated ID batches within each graph")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--profile", help="Optional profiler output directory; collected after timing")
    args = parser.parse_args()
    if args.id_batches < 1 or args.iterations < args.id_batches or args.iterations % args.id_batches:
        parser.error("--iterations must be a positive multiple of --id-batches")
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    modules = {
        name: load_module(name, path) for name, path in [("baseline", args.baseline), ("candidate", args.candidate)]
    }
    if args.uncapped:
        modules["uncapped"] = load_module("uncapped", args.uncapped)
    npu = modules["candidate"]
    results = []
    with ExitStack() as stack:
        codes = npu.HostUvaBuffer((args.table_rows, args.width), torch.float8_e4m3fn, device)
        stack.callback(codes.close)
        scales = npu.HostUvaBuffer((args.table_rows, args.width // npu.SCALE_GROUP), torch.uint8, device)
        stack.callback(scales.close)
        generator = torch.Generator().manual_seed(args.seed)
        codes.tensor.copy_(torch.randn(codes.tensor.shape, generator=generator).to(torch.float8_e4m3fn))
        scales.tensor.copy_(torch.randint(123, 132, scales.tensor.shape, dtype=torch.uint8, generator=generator))
        for tokens in args.tokens:
            ids_batches_cpu = [
                torch.randint(args.table_rows, (tokens, args.heads), generator=generator, dtype=torch.int32)
                for _ in range(args.id_batches)
            ]
            ids_batches = [ids.to(device) for ids in ids_batches_cpu]
            outputs = {
                name: torch.empty((tokens * args.heads, args.width), dtype=torch.bfloat16, device=device)
                for name in modules
            }
            scale = torch.ldexp(torch.ones_like(scales.tensor, dtype=torch.float32), scales.tensor.int() - 127)
            expected_batches = []
            for ids_cpu in ids_batches_cpu:
                expected = codes.tensor.view(torch.uint8)[ids_cpu.long()].view(torch.float8_e4m3fn).float()
                expected = expected.unflatten(-1, (-1, npu.SCALE_GROUP)) * scale[ids_cpu.long()].unsqueeze(-1)
                expected_batches.append(expected.flatten(-2).reshape(-1, args.width).bfloat16())
            graphs = {}
            for name, module in modules.items():
                print(f"compile {name} tokens={tokens}", flush=True)
                functions = []
                for ids, expected in zip(ids_batches, expected_batches):

                    def fn(module=module, output=outputs[name], ids=ids):
                        return module.gather_dequantize_host_uva(
                            codes, scales, ids, local_heads=args.heads, output=output
                        )

                    fn()
                    torch.npu.synchronize()
                    torch.testing.assert_close(outputs[name].cpu(), expected, rtol=0, atol=0)
                    functions.append(fn)
                graphs[name] = capture(functions, args.iterations)
            samples = {name: [] for name in graphs}
            names = list(graphs)
            for repeat in range(args.repeats):
                # Rotate the order to reduce bias from service activity/temperature.
                order = names[repeat % len(names) :] + names[: repeat % len(names)]
                for name in order:
                    samples[name].append(measure(graphs[name], args.iterations))
            record = {
                "tokens": tokens,
                "heads": args.heads,
                "width": args.width,
                "table_rows": args.table_rows,
                "id_batches": args.id_batches,
                "seed": args.seed,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "median_us": {name: statistics.median(values) for name, values in samples.items()},
                "samples_us": samples,
                "correct": True,
            }
            results.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "samples_us"}), flush=True)
            if args.output:
                Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
            if args.profile:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(Path(args.profile) / str(tokens))),
                ) as profiler:
                    for name, graph in graphs.items():
                        with torch.autograd.profiler.record_function(name):
                            graph.replay()
                            torch.npu.synchronize()
                    profiler.step()
            del graphs


if __name__ == "__main__":
    main()
