"""Non-destructive, process-local Ultralytics integration.

The parser in Ultralytics resolves custom class names through its own global
namespace, and requires special channel propagation. This adapter patches only
the current Python process, not site-packages. It is version-guarded and fails
rather than silently misinterpreting a YAML. Import/register BEFORE YOLO().
"""
from __future__ import annotations
import inspect
import re
import textwrap
from .modules import SPDConv, DySample, PaperCBAM

_PATCH_FLAG = "_paper_yolo11m_parser_patch_v1"


def register_model_modules() -> None:
    """Register SPDConv / DySample / PaperCBAM with Ultralytics parse_model.

    Supports parse_model variants with 'base_modules = frozenset' and the
    standard BatchNorm2d branch. If upstream changes, stops with instructions.
    """
    from ultralytics.nn import tasks
    if getattr(tasks, _PATCH_FLAG, False):
        return

    old_fn = tasks.parse_model
    source = textwrap.dedent(inspect.getsource(old_fn))
    # Version guard: don't mutate a parser with unknown structure.
    if source.count("base_modules = frozenset(") != 1:
        raise RuntimeError("Unsupported Ultralytics parse_model: missing base_modules. Pin a compatible 8.3.x and inspect manually.")
    if source.count("            Classify,") != 1:
        raise RuntimeError("Unsupported Ultralytics parse_model: cannot inject SPDConv into base_modules.")
    branch = re.search(r"(?m)^        elif m is (?:torch\.)?nn\.BatchNorm2d:", source)
    if not branch:
        raise RuntimeError("Unsupported Ultralytics parse_model: BatchNorm2d branch not found.")
    # Inject definitions into the function's globals before any YAML rows are parsed.
    inject = (
        "    # PAPER_OPT injected definitions (not upstream Ultralytics)\n"
        "    globals().update(SPDConv=SPDConv, DySample=DySample, PaperCBAM=PaperCBAM)\n"
    )
    source = source.replace("    base_modules = frozenset(", inject + "    base_modules = frozenset(", 1)
    source = source.replace("            Classify,", "            SPDConv,\n            Classify,", 1)
    # DySample and CBAM preserve channels. SPDConv belongs to standard base_modules,
    # which automatically applies official YOLO depth/width channel scaling.
    custom_branch = (
        "        elif m in {DySample, PaperCBAM}:\n"
        "            c2 = ch[f]\n"
        "            args = [ch[f], *args]\n"
    )
    source = re.sub(r"(?m)^(        elif m is (?:torch\.)?nn\.BatchNorm2d:)",
                    lambda m: custom_branch + m.group(1), source, count=1)
    for name, cls in {"SPDConv": SPDConv, "DySample": DySample, "PaperCBAM": PaperCBAM}.items():
        tasks.__dict__[name] = cls
    namespace = {}
    try:
        exec(compile(source, inspect.getsourcefile(old_fn) or "parse_model_patch", "exec"), tasks.__dict__, namespace)
    except Exception as exc:
        raise RuntimeError("Ultralytics parser adapter could not be installed. No silent fallback.") from exc
    tasks.parse_model = namespace["parse_model"]
    setattr(tasks, _PATCH_FLAG, True)


def enable_wiou_v3(alpha: float = 1.9, delta: float = 3.0, momentum: float = 0.01) -> None:
    """Switch bounding-box regression from CIoU to reconstructed WIoU-v3.

    Requires a recent Ultralytics BboxLoss.forward using (pred_dist,
    pred_bboxes, anchor_points, target_bboxes, target_scores,
    target_scores_sum, fg_mask, imgsz, stride).
    Call in the TRAINING process before model.train() only.
    """
    from ultralytics.utils import loss as ul_loss
    from .wiou import WIoUv3BboxLoss
    params = tuple(inspect.signature(ul_loss.BboxLoss.forward).parameters)
    expected = ("self", "pred_dist", "pred_bboxes", "anchor_points", "target_bboxes",
                "target_scores", "target_scores_sum", "fg_mask", "imgsz", "stride")
    if params != expected:
        raise RuntimeError(f"BboxLoss signature differs from supported version: {params}. Refusing to substitute WIoU.")
    WIoUv3BboxLoss.alpha = alpha
    WIoUv3BboxLoss.delta = delta
    WIoUv3BboxLoss.momentum = momentum
    ul_loss.BboxLoss = WIoUv3BboxLoss
