"""
read_metrics.py — summarize Prometheus text metrics from a file or stdin.

Usage:
  python read_metrics.py metrics.txt
  curl http://localhost:8000/metrics | python read_metrics.py -

Outputs:
  - gRPC latency summary (per method/grpc_type)
  - Per-phase inference summary (per model/phase)
  - Optional process RSS/CPU if present
"""
import sys
import math
import argparse
from collections import defaultdict
from prometheus_client.parser import text_string_to_metric_families

def parse_args():
    p = argparse.ArgumentParser(description="Summarize Prometheus text metrics.")
    p.add_argument("path", help="Path to metrics text file, or '-' for stdin")
    p.add_argument("--quantiles", default="0.5,0.95,0.99",
                   help="Comma-separated quantiles to print (default: 0.5,0.95,0.99)")
    return p.parse_args()

def load_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def labels_key(labels: dict, exclude=()):
    return tuple(sorted((k, v) for k, v in labels.items() if k not in exclude))

def build_histograms(families):
    """
    Return dict: {metric_name: { series_key(tuple): {'buckets':[(le,count),...], 'sum':float, 'count':int, 'labels':dict} } }
    """
    histos = defaultdict(lambda: defaultdict(lambda: {"buckets": [], "sum": 0.0, "count": 0.0, "labels": {}}))
    for fam in families:
        if fam.type != "histogram":
            continue
        for s in fam.samples:
            # s = Sample(name, labels, value, timestamp, exemplar, native_histogram)
            name, labels, value = s.name, dict(s.labels), float(s.value)
            if name.endswith("_bucket"):
                le_raw = labels["le"]
                le = float("inf") if le_raw in ("+Inf", "Inf") else float(le_raw)
                key = labels_key(labels, exclude=("le",))
                d = histos[fam.name][key]
                d["buckets"].append((le, value))
                d["labels"] = {k: v for k, v in labels.items() if k != "le"}
            elif name.endswith("_sum"):
                key = labels_key(labels)
                histos[fam.name][key]["sum"] = value
                histos[fam.name][key]["labels"] = labels
            elif name.endswith("_count"):
                key = labels_key(labels)
                histos[fam.name][key]["count"] = value
                histos[fam.name][key]["labels"] = labels
    # sort buckets by le
    for fam in histos.values():
        for series in fam.values():
            series["buckets"].sort(key=lambda x: x[0])
    return histos

def quantile_from_buckets(q, buckets, total_count):
    """
    Approximate histogram_quantile with linear interpolation within the located bucket.
    buckets: list of (le, cumulative_count) sorted by le.
    """
    if total_count <= 0:
        return float("nan")
    target = q * total_count
    prev_le = 0.0
    prev_cum = 0.0
    for le, cum in buckets:
        if target <= cum:
            inc = cum - prev_cum
            if inc <= 0.0:
                return prev_le
            if math.isinf(le):
                # last bucket is +Inf; we can't bound upper edge — return lower edge
                return prev_le
            # linear interpolate within bucket
            frac = (target - prev_cum) / inc
            return prev_le + frac * (le - prev_le)
        prev_le, prev_cum = le, cum
    return prev_le  # fallback

def fmt_s(val):
    if math.isnan(val):
        return "nan"
    if val >= 1:
        return f"{val:0.3f}s"
    # show ms for sub-second
    return f"{val*1e3:0.1f}ms"

def print_series_table(title, rows, order_cols, header):
    if not rows:
        return
    print(f"\n=== {title} ===")
    # sort rows by provided columns
    rows.sort(key=lambda r: tuple(r.get(c, "") for c in order_cols))
    # compute col widths
    cols = header
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    # header
    print("  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  " + "  ".join("-"*w for w in widths))
    # rows
    for r in rows:
        line = []
        for c, w in zip(cols, widths):
            v = r.get(c, "")
            line.append(str(v).ljust(w))
        print("  " + "  ".join(line))

def main():
    args = parse_args()
    text = load_text(args.path)
    families = list(text_string_to_metric_families(text))
    histos = build_histograms(families)
    qs = [float(x) for x in args.quantiles.split(",")]

    # gRPC latency summary
    grpc_rows = []
    for key, series in histos.get("grpc_server_latency_seconds", {}).items():
        cnt = series["count"]
        sm = series["sum"]
        avg = sm / cnt if cnt else float("nan")
        buckets = series["buckets"]
        row = {
            "method": series["labels"].get("method", ""),
            "grpc_type": series["labels"].get("grpc_type", ""),
            "count": int(cnt),
            "avg": fmt_s(avg),
        }
        for q in qs:
            row[f"p{int(q*100)}"] = fmt_s(quantile_from_buckets(q, buckets, cnt))
        grpc_rows.append(row)

    print_series_table(
        "gRPC latency (server-side)",
        grpc_rows,
        order_cols=["method", "grpc_type"],
        header=["method", "grpc_type", "count", "avg"] + [f"p{int(q*100)}" for q in qs],
    )

    # inference phase summary
    phase_rows = []
    for key, series in histos.get("inference_phase_seconds", {}).items():
        cnt = series["count"]
        sm = series["sum"]
        avg = sm / cnt if cnt else float("nan")
        buckets = series["buckets"]
        row = {
            "model": series["labels"].get("model", ""),
            "phase": series["labels"].get("phase", ""),
            "count": int(cnt),
            "avg": fmt_s(avg),
        }
        for q in qs:
            row[f"p{int(q*100)}"] = fmt_s(quantile_from_buckets(q, buckets, cnt))
        phase_rows.append(row)

    print_series_table(
        "Inference phases",
        phase_rows,
        order_cols=["model", "phase"],
        header=["model", "phase", "count", "avg"] + [f"p{int(q*100)}" for q in qs],
    )

    # optional: process info if present
    rss = next((s.value for f in families if f.name=="process_resident_memory_bytes"
                for s in f.samples if s.name=="process_resident_memory_bytes"), None)
    cpu = next((s.value for f in families if f.name=="process_cpu_seconds_total"
                for s in f.samples if s.name=="process_cpu_seconds_total"), None)
    if rss is not None or cpu is not None:
        print("\n=== Process ===")
        if rss is not None:
            print(f"  RSS: {int(float(rss))/1e9:0.3f} GB")
        if cpu is not None:
            print(f"  CPU time: {float(cpu):0.2f} s")

if __name__ == "__main__":
    main()
