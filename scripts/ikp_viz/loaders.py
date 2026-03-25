import json


def load_region_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    regions = {r["region"]: r for r in data.get("regions", [])}
    return {
        "kernel": data.get("kernel", ""),
        "kernel_id": data.get("kernel_id", 0),
        "kernel_addr": data.get("kernel_addr", 0),
        "regions": regions,
    }


def load_locality_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mem_trace_sample(path, max_records=512):
    sample = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample.append(json.loads(line))
            if len(sample) >= max_records:
                break
    return sample
