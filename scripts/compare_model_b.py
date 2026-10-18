"""Read existing results and write a separate A/B/SOTA comparison for Model B."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from en3150_a03.model_b_measure import profile_model
from en3150_a03.models.model_a import ModelA
from en3150_a03.models.model_b import ModelB
from en3150_a03.models.mobilenet_v2 import build_mobilenet_v2
from en3150_a03.models.efficientnet_b0 import build_model as build_efficientnet_b0


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_path(path):
    absolute = path.resolve()
    return str(absolute.relative_to(ROOT)) if absolute.is_relative_to(ROOT) else str(absolute)


def comparison(model_b_run, model_a_run, mobilenet_run, output, efficientnet_run=None):
    metadata = read(ROOT / "data/splits_torch_v1/metadata.json")
    expected_classes = metadata["classes"]
    if output.exists():
        raise FileExistsError(f"Choose a new comparison directory: {output}")
    profiles = {
        "ModelA": profile_model(ModelA()), "ModelB": profile_model(ModelB()),
        "MobileNetV2": profile_model(build_mobilenet_v2(pretrained=False)),
    }
    if efficientnet_run is not None:
        profiles["EfficientNet-B0"] = profile_model(build_efficientnet_b0(pretrained=False))
    rows = []
    for model_name, folder in (("ModelA", model_a_run), ("ModelB", model_b_run), ("MobileNetV2", mobilenet_run)):
        if folder is None:
            continue
        config = read(folder / ("settings.json" if model_name == "MobileNetV2" else "experiment_config.json"))
        if model_name == "MobileNetV2":
            if config["class_to_idx"] != metadata["class_to_idx"]:
                raise ValueError("MobileNet class mapping differs")
            for name, digest in metadata["manifest_sha256"].items():
                if config["manifest_sha256"][name] != digest:
                    raise ValueError(f"MobileNet split provenance differs: {name}")
            metrics = [read(folder / "test_metrics.json")]
        else:
            if config["dataset_sha256"] != metadata["dataset_sha256"]:
                raise ValueError(f"Dataset differs: {folder}")
            if model_name == "ModelB" and config["manifest_sha256"] != metadata["manifest_sha256"]:
                raise ValueError("Model B split differs")
            metrics = read(folder / "comparison.json")
        for metric in metrics:
            if metric["class_names"] != expected_classes:
                raise ValueError(f"Metric class order differs: {folder}")
            profile = profiles[model_name]
            if metric.get("total_parameters", metric.get("trainable_parameters")) != profile["total_parameters"]:
                raise ValueError(f"Architecture count differs from saved run: {folder}")
            rows.append(dict(model=model_name, optimizer=metric.get("optimizer", "adam_two_stage"),
                             accuracy=metric["accuracy"], macro_precision=metric["macro_precision"],
                             macro_recall=metric["macro_recall"], parameters=profile["total_parameters"],
                             fp32_weights_kib=profile["estimated_float32_size_kb"], macs=profile["macs"],
                             mean_epoch_seconds=metric.get("mean_epoch_seconds"), source=source_path(folder)))
    if efficientnet_run is not None:
        metric = read(efficientnet_run / "metrics.json")
        test = metric["test"]
        if metric["smoke_test"] or metric["image_size"] != 64:
            raise ValueError("Expected a full 64x64 EfficientNet-B0 run")
        if test["class_order"] != expected_classes:
            raise ValueError("EfficientNet-B0 class order differs")
        if metric["trainable_parameters"] != profiles["EfficientNet-B0"]["total_parameters"]:
            raise ValueError("EfficientNet-B0 parameter count differs")
        if sum(map(sum, test["confusion_matrix"])) != metadata["counts"]["test"]:
            raise ValueError("EfficientNet-B0 test count differs")
        profile = profiles["EfficientNet-B0"]
        rows.append(dict(model="EfficientNet-B0", optimizer="adam",
                         accuracy=test["accuracy"], macro_precision=test["macro_precision"],
                         macro_recall=test["macro_recall"], parameters=profile["total_parameters"],
                         fp32_weights_kib=profile["estimated_float32_size_kb"], macs=profile["macs"],
                         mean_epoch_seconds=sum(item["seconds"] for item in metric["history"]) / len(metric["history"]),
                         source=source_path(efficientnet_run)))
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Model B integration comparison", "",
             "Metrics are read from saved teammate runs. No model is selected by test performance.", "",
             "| Model | Optimizer | Accuracy | Parameters | FP32 weights (KiB) | Conv/Linear MACs | Mean epoch (s) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        seconds = "unavailable" if row["mean_epoch_seconds"] is None else f"{row['mean_epoch_seconds']:.2f}"
        lines.append(f"| {row['model']} | {row['optimizer']} | {row['accuracy']:.4f} | "
                     f"{row['parameters']:,} | {row['fp32_weights_kib']:.2f} | {row['macs']:,} | {seconds} |")
    lines += ["", profiles["ModelB"]["mac_convention"], "",
              "Epoch times come from different devices and runs: they cannot establish architecture speedups. "
              "Model A/B use ToTensor without augmentation; MobileNetV2 and EfficientNet-B0 use pretrained "
              "weights, flips and ImageNet normalization. Accuracy compares complete training recipes; "
              "A/B provides the controlled convolution comparison. "
              "FP32 weights exclude runtime activations and optimizer memory.", "",
              "Model A's existing results identify the dataset fingerprint and class order but do not save split "
              "manifest hashes. EfficientNet-B0 records the shared split path and class order but not "
              "manifest hashes or the dataset fingerprint; its exact data provenance cannot be independently "
              "proven from that run metadata alone."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-b-run", type=Path, required=True)
    parser.add_argument("--model-a-run", type=Path, default=ROOT / "results/model_a/modelA_20261011")
    parser.add_argument("--mobilenet-run", type=Path, default=ROOT / "results/mobilenet_v2/run_01")
    parser.add_argument("--efficientnet-run", type=Path,
                        default=ROOT / "results/efficientnet_b0/adam/run_20261009_193426")
    parser.add_argument("--output-dir", type=Path, required=True, help="New comparison directory")
    args = parser.parse_args()
    comparison(args.model_b_run, args.model_a_run, args.mobilenet_run, args.output_dir,
               args.efficientnet_run)
    print(f"Saved comparison: {args.output_dir}")

if __name__ == "__main__":
    main()
