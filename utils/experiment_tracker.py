import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_RECORD_ROOT = "/home/yangrui/Project/Base-model/local_results/experiments"
DEFAULT_CONDA_PYTHON = "/home/yangrui/miniconda3/bin/conda run -n base-model --no-capture-output python"


def _safe_name(value, max_len=120):
    text = str(value).strip()
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unnamed")[:max_len]


def _cfg_get(cfg, key, default=None):
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _contrastive_summary(cfg):
    try:
        contrastive = cfg.loss_configs.get("contrastive", None)
    except Exception:
        contrastive = None
    if not contrastive:
        return []

    rows = [
        ("contrastive.enable", contrastive.get("enable", None)),
        ("contrastive.weight", contrastive.get("weight", None)),
        ("contrastive.warmup_epochs", contrastive.get("warmup_epochs", None)),
        ("contrastive.temperature", contrastive.get("temperature", None)),
        ("contrastive.dynamic_temperature.enable", contrastive.get("dynamic_temperature", {}).get("enable", None)),
        ("contrastive.roi_size", contrastive.get("roi_size", None)),
        ("contrastive.patch_size", contrastive.get("patch_size", None)),
        ("contrastive.area_threshold", contrastive.get("area_threshold", None)),
        ("contrastive.pseudo_confidence", contrastive.get("pseudo_confidence", None)),
        ("contrastive.max_branch_diff", contrastive.get("max_branch_diff", None)),
    ]
    return [(k, v) for k, v in rows if v is not None]


def save_experiment_record(cfg, script_file, checkpoint_dir, logger=None):
    """
    Save a dataset-grouped experiment record independent from checkpoint files.

    The record is intentionally lightweight:
      - README.md: human-readable experiment card
      - train_config.yaml: full Hydra config used by this run
      - run.sh: replay entry with the same command-line overrides
      - overrides.txt: raw Hydra overrides
    """
    record_root = Path(str(_cfg_get(cfg, "experiment_record_root", DEFAULT_RECORD_ROOT)))
    dataset = _safe_name(_cfg_get(cfg, "data_name", "UnknownDataset"))
    run_name = str(_cfg_get(cfg, "loss_name", Path(script_file).stem))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_name = f"{timestamp}_{_safe_name(run_name)}"
    record_dir = record_root / dataset / record_name
    record_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(script_file).resolve()
    project_root = script_path.parent.parent
    rel_script = script_path.relative_to(project_root).as_posix()
    overrides = sys.argv[1:]
    override_lines = " \\\n  ".join(shlex.quote(arg) for arg in overrides)
    override_block = f" \\\n  {override_lines}" if override_lines else ""

    conda_python = str(_cfg_get(cfg, "experiment_python_cmd", DEFAULT_CONDA_PYTHON))
    run_sh = f"""#!/usr/bin/env bash
set -e

# Experiment ID: {timestamp}
# Experiment Name: {run_name}
# Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Dataset: {_cfg_get(cfg, "data_name", "UnknownDataset")}
# Train Script: {rel_script}
# Checkpoint Dir: {checkpoint_dir}
# Record Dir: {record_dir}

cd {shlex.quote(project_root.as_posix())}

{conda_python} {shlex.quote(rel_script)}{override_block}
"""

    (record_dir / "run.sh").write_text(run_sh, encoding="utf-8")
    try:
        os.chmod(record_dir / "run.sh", 0o755)
    except OSError:
        pass

    (record_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=False), encoding="utf-8")
    (record_dir / "overrides.txt").write_text("\n".join(overrides) + ("\n" if overrides else ""), encoding="utf-8")

    key_rows = [
        ("data_name", _cfg_get(cfg, "data_name")),
        ("loss_name", _cfg_get(cfg, "loss_name")),
        ("label_suffix", _cfg_get(cfg, "label_suffix")),
        ("pseudo_label_mode", _cfg_get(cfg, "pseudo_label_mode", "hard")),
        ("ramp_epochs", _cfg_get(cfg, "ramp_epochs")),
        ("pseudo_weight", _cfg_get(cfg, "pseudo_weight")),
        ("batch_size", _cfg_get(cfg, "batch_size")),
        ("repeats", _cfg_get(cfg, "repeats")),
    ]
    key_rows.extend(_contrastive_summary(cfg))

    summary_lines = "\n".join(f"- `{k}`: `{v}`" for k, v in key_rows if v is not None)
    readme = f"""# Experiment Record

## Basic Info

- Experiment ID: `{timestamp}`
- Experiment Name: `{run_name}`
- Date: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`
- Dataset: `{_cfg_get(cfg, "data_name", "UnknownDataset")}`
- Train Script: `{rel_script}`
- Checkpoint Dir: `{checkpoint_dir}`

## Key Settings

{summary_lines}

## Files

- `run.sh`: replay command for this experiment
- `train_config.yaml`: full Hydra config snapshot
- `overrides.txt`: command-line overrides used in this run
"""
    (record_dir / "README.md").write_text(readme, encoding="utf-8")

    if logger is not None:
        logger.info(f"Experiment record saved to: {record_dir}")
    return str(record_dir)
