\
\
\
\
from __future__ import annotations
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
import yaml
RUN_MODE = "smoke"
DATA_YAML = r""
DEVICE = "0"
MODEL_YAML = "models/yolo11m_opt.yaml"
PRETRAIN_WEIGHTS = "yolo11m.pt"
USE_OFFICIAL_PRETRAINED = False
ACKNOWLEDGE_105_LABELS = False
USE_WIOU_FOR_FULL = True
FINAL_TEST_EVALUATION = False
WORKERS = 0
SEED = 42
PROJECT_NAME = "reconstruction_training"
ROOT = Path(__file__).resolve().parent
PAPER_HYP = dict(
    lr0=0.00944, lrf=0.01082, momentum=0.9008, weight_decay=0.00049,
    warmup_epochs=3.44721, warmup_momentum=0.835,
    box=6.72157, cls=0.46, dfl=1.22685,
    hsv_h=0.0139, hsv_s=0.44243, hsv_v=0.31247,
    bgr=0.0, degrees=0.0, translate=0.10447, scale=0.55597,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.42321,
    mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
)
def sha256_file(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def as_path(name: str | Path) -> Path:
    p = Path(name)
    return p if p.is_absolute() else (ROOT / p).resolve()
def audit_dataset(data_file: Path, model_file: Path, mode: str) -> dict:
    if not data_file.is_file():
        raise FileNotFoundError(f"DATA_YAML not found: {data_file}\nSet DATA_YAML at the top of this script")
    if not model_file.is_file():
        raise FileNotFoundError(f"Model configuration not found: {model_file}")
    data = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    model = yaml.safe_load(model_file.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(model, dict):
        raise ValueError("data.yaml and model.yaml must contain YAML mappings")
    if not data.get("train") or not data.get("val"):
        raise ValueError("data.yaml must contain train and val entries")
    names = data.get("names")
    nc = data.get("nc")
    if names is None:
        raise ValueError("Provide the complete names list or mapping in data.yaml to verify class definitions")
    if isinstance(names, dict):
        indices = sorted(int(i) for i in names)
        if indices != list(range(len(indices))):
            raise ValueError("Class indices in data.yaml names must be consecutive, starting from zero")
        names_seq = [str(names[i] if i in names else names[str(i)]) for i in indices]
    elif isinstance(names, list):
        names_seq = [str(x) for x in names]
    else:
        raise ValueError("names must be a list or a dictionary indexed consecutively from zero")
    if len(set(names_seq)) < len(names_seq):
        raise ValueError("Duplicate class names were found in names; verify class labels")
    nc = int(nc) if nc is not None else len(names_seq)
    if nc != len(names_seq):
        raise ValueError(f"data.yaml nc={nc} but the number of names is {len(names_seq)}")
    model_nc = int(model.get("nc", -1))
    if model_nc != nc:
        raise ValueError(f"Model nc={model_nc}, data nc={nc}. Ultralytics would override the class count; training is blocked to avoid an unintended configuration. Verify labels and update the model YAML.")
    if nc == 105:
        print("[IMPORTANT] The 105 labels span multiple taxonomic ranks. Standard YOLO Detect does not enforce hierarchical multilabel consistency.")
        if mode != "smoke" and not ACKNOWLEDGE_105_LABELS:
            raise RuntimeError("Before full training, verify the 105-label definitions, annotations for each image, and mapping of the 38 terminal biological classes. Set ACKNOWLEDGE_105_LABELS=True only after verification; this flag does not modify the model.")
    print(f"[DATA] nc={nc}; first five class names={names_seq[:5]}")
    if all(s.startswith("taxon_") for s in names_seq[:min(5, len(names_seq))]):
        raise ValueError("Placeholder taxon_*** class names are still in use; replace them with actual dataset labels")
    data_root = Path(data.get("path") or data_file.parent)
    if not data_root.is_absolute():
        data_root = (data_file.parent / data_root).resolve()
    inspected, seen, bad, examples = 0, set(), [], []
    for split in ("train", "val"):
        source = data.get(split)
        if not isinstance(source, str) or source.startswith(("http://", "https://")):
            continue
        image_path = Path(source)
        image_path = image_path if image_path.is_absolute() else data_root / image_path
        if image_path.is_dir():
            parts = list(image_path.parts)
            if "images" in parts:
                ix = len(parts) - 1 - parts[::-1].index("images")
                parts[ix] = "labels"
                label_path = Path(*parts)
                if label_path.is_dir():
                    for txt in label_path.rglob("*.txt"):
                        inspected += 1
                        lines = [line.strip() for line in txt.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                        if len(lines) > 1 and len(examples) < 3:
                            examples.append(f"{split}/{txt.name}: {len(lines)} boxes")
                        for n, line in enumerate(lines, 1):
                            fields = line.split()
                            try:
                                if len(fields) != 5:
                                    raise ValueError("Expected five YOLO detection columns: class cx cy w h")
                                idx = int(fields[0])
                                vals = [float(v) for v in fields[1:]]
                                if not (0 <= idx < nc and all(0 <= v <= 1 for v in vals)
                                        and vals[2] > 0 and vals[3] > 0):
                                    raise ValueError("Invalid class ID or normalized bounding-box coordinates")
                                seen.add(idx)
                            except (ValueError, OverflowError) as e:
                                bad.append(f"{txt}:{n} -> {e}")
                                if len(bad) >= 8:
                                    break
                        if len(bad) >= 8:
                            break
    if bad:
        raise ValueError("Invalid YOLO annotations found:\n\n" + "\n".join(bad))
    print(f"[DATA] Local annotation files checked: {inspected}; unique class IDs observed: {len(seen)}")
    if examples:
        print("[NOTICE] Examples of multiple boxes in one image: ", "; ".join(examples), ". If there is one specimen per image, verify whether multiple boxes represent taxonomic-rank annotations.")
    if nc == 105 and seen and len(seen) < nc:
        print(f"[NOTICE] Only {len(seen)}/{nc} class IDs were observed in the checked annotations; this is diagnostic information, not necessarily an error.")
    return {"nc": nc, "names": names_seq, "sampled_label_files": inspected,
            "sampled_label_class_ids": sorted(seen), "data_file": str(data_file)}
def load_tuned_hyperparams() -> dict:
    p = ROOT / "tuning_results" / "best_params.json"
    if not p.is_file():
        raise FileNotFoundError(f"Tuning result not found: {p}. Run tune_rebuilt.py first or set RUN_MODE='paper'")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {**PAPER_HYP, **obj["best_hyperparameters"]}
def build_train_kwargs(mode: str, hyp: dict, run_name: str) -> dict:
    common = dict(
        data=str(as_path(DATA_YAML)), epochs=1 if mode == "smoke" else 400,
        imgsz=640 if mode == "smoke" else 960, batch=2 if mode == "smoke" else 4,
        nbs=8, optimizer="SGD", patience=50,
        close_mosaic=0 if mode == "smoke" else 50,
        workers=WORKERS, device=DEVICE, seed=SEED, deterministic=False,
        amp=True, cache=False, save=True, val=True, plots=True,
        project=str(ROOT / "runs" / PROJECT_NAME), name=run_name,
        exist_ok=False,
    )
    common.update(hyp)
    if mode == "smoke":
        common.update(warmup_epochs=0.0, mosaic=0.0, mixup=0.0)
    return common
def make_model(model_yaml: Path, pretrain: bool):
    from reconstructed_yolo11m import register_model_modules
    register_model_modules()
    from ultralytics import YOLO
    model = YOLO(str(model_yaml), task="detect")
    strides = [round(float(i), 2) for i in model.model.stride]
    if strides != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"P2/P3/P4/P5strides are incorrect: {strides}")
    if pretrain:
        f = as_path(PRETRAIN_WEIGHTS)
        if not f.is_file():
            raise FileNotFoundError(f"Pretrained weights not found: {f}. Download official yolo11m.pt or disable USE_OFFICIAL_PRETRAINED.")
        print(f"[WEIGHTS] Loading official pretrained weights; only compatible layers can be transferred: {f}")
        model.load(str(f))
    else:
        print("[WEIGHTS] This model is randomly initialized; these are not original manuscript weights")
    return model
def activate_wiou():
    from reconstructed_yolo11m import enable_wiou_v3
    enable_wiou_v3()
    from ultralytics.utils.loss import BboxLoss
    if BboxLoss.__name__ != "WIoUv3BboxLoss":
        raise RuntimeError("WIoU replacement was not activated")
    print("[LOSS] Reconstructed WIoUv3 has been enabled. The exact original implementation and parameters were not specified, so this should not be described as an exact reproduction.")
def run_training():
    mode = RUN_MODE.lower().strip()
    if mode not in ("smoke", "paper", "tuned"):
        raise ValueError("RUN_MODE must be smoke, paper, or tuned")
    print("\n====== YOLO11m-opt Reconstructed Training ======")
    print("Project directory:", ROOT)
    print("Python:", sys.executable)
    print("Mode:", mode, "GPU:", DEVICE)
    model_yaml = as_path(MODEL_YAML)
    data_yaml = as_path(DATA_YAML)
    info = audit_dataset(data_yaml, model_yaml, mode)
    hyp = PAPER_HYP.copy() if mode != "tuned" else load_tuned_hyperparams()
    import torch
    import ultralytics
    print(f"[ENV] Python {platform.python_version()}, torch {torch.__version__}, ultralytics {ultralytics.__version__}")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU training was requested, but torch.cuda.is_available() returned False. Check the PyTorch CUDA installation.")
    use_wiou = mode != "smoke" and USE_WIOU_FOR_FULL
    if use_wiou:
        activate_wiou()
    model = make_model(model_yaml, USE_OFFICIAL_PRETRAINED)
    run_name = {"smoke":"00_smoke", "paper":"01_paper_params", "tuned":"02_tuned_params"}[mode]
    kwargs = build_train_kwargs(mode, hyp, run_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = ROOT / "experiment_records" / f"{run_name}_{stamp}"
    record_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "notice":"RECONSTRUCTED experiment; not the lost original weights/code",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "mode": mode, "dataset": info, "yaml_sha256": sha256_file(model_yaml),
        "data_yaml_sha256": sha256_file(data_yaml),
        "model_file": str(model_yaml), "weights": str(as_path(PRETRAIN_WEIGHTS)) if USE_OFFICIAL_PRETRAINED else None,
        "wiou_v3_reconstruction": use_wiou,
        "hyperparameter_origin": "user notes, not original run logs" if mode == "paper" else "NEW tuning results" if mode == "tuned" else "smoke check",
        "train_arguments": kwargs,
        "environment": {"python": platform.python_version(), "pytorch":torch.__version__, "ultralytics":ultralytics.__version__},
    }
    (record_dir/"experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    shutil.copy2(model_yaml, record_dir/"model_used.yaml")
    shutil.copy2(data_yaml, record_dir/"dataset_used.yaml")
    print(f"[RECORD] {record_dir}")
    print(f"[TRAIN] epochs={kwargs['epochs']} imgsz={kwargs['imgsz']} batch={kwargs['batch']} WIoU={use_wiou}")
    results = model.train(**kwargs)
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        output_dir = Path(trainer.save_dir)
        best = output_dir/"weights"/"best.pt"
        last = output_dir/"weights"/"last.pt"
        print("\nTraining output directory:", output_dir)
        print("best.pt:", best, "exists:", best.is_file())
        print("last.pt:", last, "exists:", last.is_file())
        (record_dir/"output_dir.txt").write_text(str(output_dir), encoding="utf-8")
        if FINAL_TEST_EVALUATION:
            if not info or not yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")).get("test"):
                raise ValueError("No test split is defined; final held-out test evaluation cannot be performed")
            print("[TEST] Evaluate the held-out test set once after finalizing the model")
            from reconstructed_yolo11m import register_model_modules
            register_model_modules()
            from ultralytics import YOLO
            best_model = YOLO(str(best), task="detect")
            metrics = best_model.val(data=str(data_yaml), split="test", imgsz=960, device=DEVICE)
            print("[TEST] mAP@0.5:0.95 =", metrics.box.map)
    print("[DONE] Training finished. These results are from a new reconstructed experiment, not recovered original manuscript best.pt weights")
    return results
if __name__ == "__main__":
    run_training()
