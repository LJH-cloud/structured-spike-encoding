"""Check source identities and, when supplied, verify historical aggregate tables.

Uses only Python's standard library. Does not evaluate images or checkpoints.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("psnr", "ssim", "mae", "mse")
SEEDS = {"1314", "350665371", "827664235"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(observed, expected, label):
    require(math.isfinite(observed) and math.isfinite(expected), label + " nonfinite")
    require(math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"{label}: {observed} != {expected}")


def rows(filename):
    with (ROOT / "results" / "initialization" / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row, *columns):
    return tuple(row[column] for column in columns)


def main():
    provenance = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text())
    for path, expected in provenance["published_file_sha256"].items():
        observed = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        require(observed == expected, "file identity changed: " + path)
    if not (ROOT / "results/initialization/seed_metrics.csv").is_file():
        print(json.dumps({"status": "PASS", "source_files_checked": len(provenance["published_file_sha256"]),
                          "result_tables": "not included in code-only release",
                          "numeric_checks": 0}, indent=2))
        return
    split_specs = {
        "train": (1678, "208ed00458a74b751ca9d67a1fe4ea5dcb296b2da4b4f0e61d32ac10934724f5"),
        "dev": (425, "a3198aca49fe14d7de919a26369e295570c74a4606e2ca3d43c0373fccccb85c"),
        "test": (1111, "70a2aa3dedee542be8f2d25580e28717bbc51c02b9c87c454d100dc696e57b30"),
    }
    splits = {}
    for split, (count, digest) in split_specs.items():
        content = (ROOT / "splits" / (split + ".txt")).read_bytes()
        require(hashlib.sha256(content).hexdigest() == digest, split + " hash")
        ids = content.decode().splitlines()
        require(len(ids) == len(set(ids)) == count, split + " count/duplicates")
        splits[split] = set(ids)
    require(not splits["train"] & splits["dev"], "train/dev overlap")
    require(not splits["train"] & splits["test"], "train/test overlap")
    parent = lambda sid: sid.split("__")[1].rsplit("_", 1)[0]
    train_parents = {parent(sid) for sid in splits["train"]}
    dev_parents = {parent(sid) for sid in splits["dev"]}
    test_parents = {parent(sid) for sid in splits["test"]}
    require(not train_parents & dev_parents, "train/dev parent overlap")
    strata = {"seen_parent": train_parents & test_parents,
              "unseen_parent": test_parents - train_parents}
    cap_rows = rows("capture_seed_metrics.csv")
    cap_keys = [key(r, "system", "arm", "seed", "phase", "capture") for r in cap_rows]
    require(len(cap_keys) == len(set(cap_keys)), "duplicate capture rows")
    caps = defaultdict(list)
    for row in cap_rows:
        require(row["seed"] in SEEDS, "unexpected seed")
        caps[key(row, "system", "arm", "seed", "phase")].append(row)
    seeds = rows("seed_metrics.csv")
    dimensions = ("system", "arm", "seed", "phase", "stratum", "aggregation")
    seed_index = {key(row, *dimensions): row for row in seeds}
    require(len(seed_index) == len(seeds), "duplicate seed rows")
    checks = 0
    for row in seeds:
        require(int(row["endpoint_step"]) == 100000, "incomplete endpoint")
        selected = caps[key(row, "system", "arm", "seed", "phase")]
        if row["stratum"] in strata:
            selected = [r for r in selected if r["capture"] in strata[row["stratum"]]]
        else:
            require(row["stratum"] == "all_" + row["phase"], "unknown stratum")
        require(len(selected) == int(row["captures"]), "capture count")
        count = sum(int(r["images"]) for r in selected)
        require(count == int(row["images"]), "image count")
        for metric in METRICS:
            if row["aggregation"] == "capture_macro":
                value = statistics.mean(float(r[metric]) for r in selected)
            else:
                require(row["aggregation"] == "image_micro", "unknown aggregation")
                value = sum(float(r[metric]) * int(r["images"]) for r in selected) / count
            close(float(row[metric]), value, "capture-to-seed " + metric)
            checks += 1
    summary_columns = ("system", "arm", "phase", "stratum", "aggregation")
    grouped = defaultdict(list)
    for row in seeds:
        grouped[key(row, *summary_columns)].append(row)
    for row in rows("arm_summary.csv"):
        selected = grouped[key(row, *summary_columns)]
        require({r["seed"] for r in selected} == SEEDS and len(selected) == 3, "three seeds required")
        for metric in METRICS:
            values = [float(r[metric]) for r in selected]
            close(float(row[metric + "_mean"]), statistics.mean(values), metric + " arm mean")
            close(float(row[metric + "_sample_sd"]), statistics.stdev(values), metric + " arm SD")
            checks += 2
    contrasts = rows("paired_seed_contrasts.csv")
    contrast_dimensions = ("system", "contrast", "seed", "phase", "stratum", "aggregation")
    require(len({key(r, *contrast_dimensions) for r in contrasts}) == len(contrasts), "duplicate contrasts")
    contrast_groups = defaultdict(list)
    for row in contrasts:
        comparator = row["contrast"].removeprefix("full_minus_")
        require(row["contrast"] == "full_minus_" + comparator, "unknown contrast")
        tail = key(row, "seed", "phase", "stratum", "aggregation")
        full = seed_index[(row["system"], "full", *tail)]
        other = seed_index[(row["system"], comparator, *tail)]
        for metric in METRICS:
            close(float(row[metric]), float(full[metric]) - float(other[metric]), metric + " paired difference")
            checks += 1
        contrast_groups[key(row, "system", "contrast", "phase", "stratum", "aggregation")].append(row)
    for row in rows("paired_contrast_summary.csv"):
        selected = contrast_groups[key(row, "system", "contrast", "phase", "stratum", "aggregation")]
        require({r["seed"] for r in selected} == SEEDS and len(selected) == 3, "paired seeds")
        for metric in METRICS:
            values = [float(r[metric]) for r in selected]
            close(float(row[metric + "_mean"]), statistics.mean(values), metric + " difference mean")
            close(float(row[metric + "_sample_sd"]), statistics.stdev(values), metric + " difference SD")
            require(int(row[metric + "_positive_seeds"]) == sum(v > 0 for v in values), metric + " positive seeds")
            require(int(row[metric + "_negative_seeds"]) == sum(v < 0 for v in values), metric + " negative seeds")
            checks += 4
    lineage = rows("lineage.csv")
    require(len(lineage) == len({r["job"] for r in lineage}) == 18, "18 unique endpoints")
    kinds = Counter(r["lineage"] for r in lineage)
    require(kinds["original_fresh"] == 14 and sum(kinds.values()) == 18, "fresh/recovered lineage")
    require(sum("resum" in r["lineage"] for r in lineage) == 4, "four explicit recoveries")
    for row in lineage:
        require(row["excluded"] == "False" and row["endpoint_step"] == "100000", "accepted final endpoint")
        require(row["dev_images"] == "425" and row["test_images"] == "1111", "evaluation coverage")
    print(json.dumps({"status": "PASS", "numeric_checks": checks,
                      "source_files_checked": len(provenance["published_file_sha256"]),
                      "split_counts": {k: len(v) for k, v in splits.items()},
                      "new_endpoint_lineage": dict(kinds)}, indent=2))


if __name__ == "__main__":
    main()
