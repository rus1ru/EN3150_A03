"""Matched single-seed A/B study; all six recipes freeze before test evaluation."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from en3150_a03.data import EuroSATDataset, verify_splits
from en3150_a03.models.model_a import ModelA
from en3150_a03.models.model_b_v2 import ModelBV2
from en3150_a03.model_b_v2_data import training_statistics
from en3150_a03.model_a_evaluate import evaluate_model
from en3150_a03.model_b_measure import profile_model
from en3150_a03.model_b_train import atomic_json, atomic_checkpoint, save_history, synchronize

OPTIMIZERS = {"adam": .001, "sgd": .01, "sgd_momentum": .01}
MODELS = {"model_a": ModelA, "model_b": ModelBV2}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_optimizer(model, name):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=OPTIMIZERS[name], weight_decay=0.)
    return torch.optim.SGD(model.parameters(), lr=OPTIMIZERS[name],
                           momentum=.9 if name == "sgd_momentum" else 0., weight_decay=0.)


def loss_epoch(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    total, loss_sum, correct = 0, 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
    return loss_sum / total, correct / total


def figures(folder, history, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for prefix, label in [("train", "Training"), ("val", "Validation")]:
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            ax.plot([r["epoch"] for r in history], [r[prefix+"_"+metric] for r in history], label=label)
            ax.set(xlabel="Epoch", ylabel=metric.capitalize())
            ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "training_curves.png", dpi=160)
    plt.close(fig)
    matrix, classes = result["confusion_matrix"], result["class_names"]
    with (folder / "confusion_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *classes])
        writer.writerows([[name, *row] for name, row in zip(classes, matrix)])
    fig, ax = plt.subplots(figsize=(10, 8))
    heat = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(heat, ax=ax)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i][j] > max(map(max, matrix))/2 else "black")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new study output: {output}")
    audit = verify_splits(args.data_root, args.split_dir)
    meta = json.loads((args.split_dir / "metadata.json").read_text())
    stats = training_statistics(args.data_root, args.split_dir)
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])])
    training = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])])
    output.mkdir(parents=True)
    source_paths = [Path(__file__), ROOT/"src/en3150_a03/models/model_a.py",
                    ROOT/"src/en3150_a03/models/model_b_v2.py", ROOT/"src/en3150_a03/data.py",
                    ROOT/"src/en3150_a03/model_b_v2_data.py", ROOT/"src/en3150_a03/model_a_evaluate.py"]
    config = dict(seed=args.seed, epochs=args.epochs, batch_size=64, workers=0,
        optimizers=OPTIMIZERS, momentum=.9, weight_decay=0., scheduler=None,
        loss="plain cross entropy", label_smoothing=0., precision="FP32; AMP and TF32 disabled",
        augmentation="independent horizontal/vertical flips p=0.5; training only", normalization=stats,
        checkpoint_selection="minimum validation cross entropy", device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        torch=torch.__version__, dataset_sha256=audit["dataset_sha256"], manifest_sha256=meta["manifest_sha256"],
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        scope="Matched complete architectures, not an isolated convolution-type ablation; single seed; prior test reuse")
    atomic_json(output/"protocol.json", config)
    # Pre-create all train/validation datasets; no test loader exists during training.
    datasets = {"train":EuroSATDataset(args.data_root,args.split_dir/"train.csv",training),
                "val":EuroSATDataset(args.data_root,args.split_dir/"val.csv",evaluation)}
    for model_name, builder in MODELS.items():
        for opt in OPTIMIZERS:
            folder = output/model_name/opt
            folder.mkdir(parents=True)
            seed_all(args.seed)
            model = builder()
            profile = profile_model(model)
            atomic_json(folder/"architecture_profile.json", profile)
            # Reseed initialization and shuffle independently of model/profiling RNG use.
            seed_all(args.seed)
            model = builder().to(device)
            optimizer = make_optimizer(model,opt)
            loaders = {name:DataLoader(dataset,batch_size=64,shuffle=name=="train",num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed)) for name,dataset in datasets.items()}
            # Shared global augmentation RNG starts at the same point for every recipe.
            seed_all(args.seed)
            history, best = [], float("inf")
            for epoch in range(1,args.epochs+1):
                synchronize(device)
                start=time.perf_counter()
                train_loss, train_accuracy=loss_epoch(model,loaders["train"],device,optimizer)
                val_loss, val_accuracy=loss_epoch(model,loaders["val"],device)
                synchronize(device)
                row=dict(epoch=epoch,train_loss=train_loss,train_accuracy=train_accuracy,
                         val_loss=val_loss,val_accuracy=val_accuracy,seconds=time.perf_counter()-start)
                history.append(row)
                save_history(history,folder)
                if val_loss < best:
                    best=val_loss
                    atomic_checkpoint(folder/"best_model.pt",dict(model_state_dict=model.state_dict(),epoch=epoch,val_loss=val_loss))
                print(f"{model_name}/{opt} {epoch:02}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f} acc={val_accuracy:.2%} {row['seconds']:.1f}s",flush=True)
            atomic_json(folder/"training_summary.json",dict(complete=True,epochs=args.epochs,
                        mean_epoch_seconds=sum(r["seconds"] for r in history)/len(history)))
            del model,optimizer,loaders
            if device.type=="cuda": torch.cuda.empty_cache()
    atomic_json(output/"training_complete.json",dict(complete=True,recipes=6))
    # All six fixed recipes have finished before any test evaluation.
    test = DataLoader(EuroSATDataset(args.data_root,args.split_dir/"test.csv",evaluation),batch_size=64,shuffle=False)
    comparisons=[]
    for model_name,builder in MODELS.items():
        for opt in OPTIMIZERS:
            folder=output/model_name/opt
            model=builder().to(device)
            checkpoint=torch.load(folder/"best_model.pt",map_location=device,weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            result=evaluate_model(model,test,meta["classes"],device)
            result.update(model=model_name,optimizer=opt,best_epoch=checkpoint["epoch"])
            atomic_json(folder/"test_metrics.json",result)
            import csv
            with (folder/"training_history.csv").open() as stream:
                history=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
            figures(folder,history,result)
            comparisons.append(result)
            del model
    atomic_json(output/"comparison.json",comparisons)
    lines=["# Controlled A/B study", "", "Single seed; matched recipes. No learning-rate search. Prior test reuse remains a limitation.", "",
           "| Model | Optimizer | Test accuracy | Macro precision | Macro recall |", "|---|---|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['model']} | {r['optimizer']} | {r['accuracy']:.4%} | {r['macro_precision']:.4%} | {r['macro_recall']:.4%} |")
    (output/"comparison.md").write_text("\n".join(lines)+"\n")
    atomic_json(output/"complete.json",dict(complete=True,recipes=6))
    print("Study complete:",output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=ROOT/"data/raw/EuroSAT_RGB")
    parser.add_argument("--split-dir",type=Path,default=ROOT/"data/splits_torch_v1")
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.epochs<1: parser.error("epochs must be positive")
    run(args)

if __name__=="__main__":
    main()
