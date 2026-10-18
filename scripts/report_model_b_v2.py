"""Generate a Model B v2 report from saved metrics, without modifying any run."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: json.loads((args.run_dir/name).read_text())
    config, summary, profile = read("experiment_config.json"), read("training_summary.json"), read("architecture_profile.json")
    result = read("test_metrics.json") if (args.run_dir/"test_metrics.json").exists() else None
    lines = ["# Model B v2 experiment", "",
        f"Architecture: custom inverted-residual depthwise CNN; channel attention={config['attention']}.",
        f"Parameters: **{profile['total_parameters']:,}**; FP32 weights: **{profile['estimated_float32_size_kb']:.2f} KiB**; "
        f"Conv/Linear MACs: **{profile['macs']:,}** at 64x64.", "",
        f"Optimizer: {config['optimizer']}; peak learning rate: {config['learning_rate']}; "
        f"completed epochs: {summary['epochs_completed']}/{config['epochs']}; seed: {config['seed']}.",
        f"Selected epoch: {summary['best_epoch']}; validation accuracy: **{summary['best_val_accuracy']:.2%}**; "
        f"validation cross-entropy: {summary['best_val_loss']:.5f}.", "",
        "Checkpoint selection uses minimum validation loss. Normalization uses training images only. "
        f"Training augmentation enabled: {config['augmentation']}. When enabled, it uses flips, quarter-turns, reflected crops and mild color jitter. "
        "Label smoothing applies to training; validation/test metrics use ordinary targets.", "",
        "## Saved result comparison", "",
        "| Model/recipe | Parameters | FP32 weights (KiB) | MACs | Test accuracy |",
        "| --- | ---: | ---: | ---: | ---: |"]
    rows = []
    baseline = ROOT/"results/model_b/modelB_gpu_01/comparison.json"
    if baseline.exists():
        for row in json.loads(baseline.read_text()):
            rows.append(dict(model="Model B v1 / "+row["optimizer"], parameters=row["trainable_parameters"],
                weights_kib=row["estimated_float32_size_kb"], macs=row["macs"], accuracy=row["accuracy"]))
    rows.append(dict(model="Model B v2 / "+config["optimizer"], parameters=profile["total_parameters"],
        weights_kib=profile["estimated_float32_size_kb"], macs=profile["macs"], accuracy=result["accuracy"] if result else None))
    for row in rows:
        accuracy = f"{row['accuracy']:.2%}" if row["accuracy"] is not None else "not evaluated"
        lines.append(f"| {row['model']} | {row['parameters']:,} | {row['weights_kib']:.2f} | {row['macs']:,} | {accuracy} |")
    if result:
        lines += ["", f"Macro precision: {result['macro_precision']:.4f}; macro recall: {result['macro_recall']:.4f}."]
    lines += ["", "## Interpretation", "",
        "Version 2 reinvests the parameter budget in deeper/wider features, batch normalization, linear projections "
        "and residual connections. SE gates model channel importance; global average pooling produces the classifier input. "
        "The architecture and training recipe both change, so a gain cannot be attributed to attention alone. "
        "Use the no-attention ablation with the same training recipe to test that component.", "",
        profile["mac_convention"], "",
        "The original v1 is a matched-width Model A comparison; v2 targets accuracy under the 100k limit. "
        "V2 uses more memory/compute than v1 and Model A. FP32 weights exclude activations and optimizer state. "
        "Timing across different computers does not establish speedup. The frozen split is image-level, "
        "not geographically independent. One seed does not establish statistical superiority.", "",
        "## Sources", "",
        "- MobileNets: https://arxiv.org/abs/1704.04861",
        "- MobileNetV2: https://arxiv.org/abs/1801.04381",
        "- Squeeze-and-Excitation: https://arxiv.org/abs/1709.01507",
        "- MobileNetV3 hard gating: https://arxiv.org/abs/1905.02244",
        "- Network In Network/global average pooling: https://arxiv.org/abs/1312.4400",
        "- Batch Normalization: https://arxiv.org/abs/1502.03167",
        "- AdamW: https://arxiv.org/abs/1711.05101"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"experiment_report.md").write_text("\n".join(lines)+"\n")
    (args.output_dir/"comparison.json").write_text(json.dumps(rows, indent=2)+"\n")
    with (args.output_dir/"comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(args.output_dir/"experiment_report.md")

if __name__=="__main__": main()
