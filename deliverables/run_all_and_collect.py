"""
Run validators for all parts and collect outputs into `deliverables/`.

Usage (from project root):
    python deliverables/run_all_and_collect.py
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEL = ROOT / "deliverables"
FIG = DEL / "figures"
ANL = DEL / "analysis"
DEL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
ANL.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

PART_OUTPUTS = {
    "part1_mlp": ROOT / "outputs" / "part1_mlp",
    "part2_cnn": ROOT / "outputs" / "part2_cnn" / "part2_cnn",
    "part3_rnn": ROOT / "outputs" / "part3_rnn" / "part3_rnn",
}


CANONICAL_FIGURES = {
    "part1_mlp": {
        "confusion_matrix.png": "confusion_matrix.png",
        "metrics_summary.png": "metrics_summary.png",
        "lime_explanation.png": "lime_explanation_part1.png",
        "initialization_comparison.png": "initialization_comparison.png",
    },
    "part2_cnn": {
        "mlp_vs_cnn.png": "mlp_vs_cnn.png",
        "ablation.png": "ablation.png",
        "feature_maps.png": "feature_maps.png",
        "confusion_matrix.png": "confusion_matrix_part2.png",
        "lime_explanation.png": "lime_explanation_part2.png",
    },
    "part3_rnn": {
        "models_comparison.png": "models_comparison.png",
        "models_detailed_comparison.png": "models_detailed_comparison.png",
        "memory_decay.png": "memory_decay.png",
        "grad_clip.png": "grad_clip.png",
    },
}


def _collect_part(part_name: str, result: dict) -> None:
    part_dir = PART_OUTPUTS[part_name]
    part_figs = part_dir / "figures"
    if part_figs.exists():
        for fig_file in part_figs.iterdir():
            if fig_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                shutil.copy(fig_file, FIG / f"{part_name}_{fig_file.name}")
                canonical = CANONICAL_FIGURES.get(part_name, {}).get(fig_file.name)
                if canonical:
                    shutil.copy(fig_file, FIG / canonical)

    part_analysis = part_dir / "analysis"
    if part_analysis.exists():
        for json_file in part_analysis.iterdir():
            if json_file.suffix.lower() == ".json":
                shutil.copy(json_file, ANL / f"{part_name}_{json_file.name}")

    checklist = part_dir / "checklist_summary.json"
    if checklist.exists():
        shutil.copy(checklist, DEL / f"checklist_summary_{part_name}.json")

    for key in ("confusion_png", "metrics_summary_png", "lime_png", "mlp_vs_cnn", "feature_maps",
                "ablation_png", "grad_clip_png", "models_comparison_png"):
        path = result.get(key)
        if path and Path(path).exists():
            shutil.copy(path, FIG / Path(path).name)


def main() -> None:
    collected = {"parts": {}}
    for part_name in PART_OUTPUTS:
        try:
            mod = importlib.import_module(f"{part_name}.validate_checklist")
            importlib.reload(mod)
            print(f"Running validator for {part_name}...")
            result = mod.run()
            collected["parts"][part_name] = result
            _collect_part(part_name, result)
            print(f"  OK: {part_name}")
        except Exception as exc:
            print(f"  ERROR {part_name}: {exc}")
            collected["parts"][part_name] = {"status": "error", "error": str(exc)}

    # Part 2 LIME needs a saved CNN checkpoint
    try:
        from part2_cnn.generate_lime import FIGURES as LIME_FIGURES  # noqa: WPS433
        import part2_cnn.generate_lime as lime_mod  # noqa: WPS433
        importlib.reload(lime_mod)
        if (LIME_FIGURES / "lime_explanation.png").exists():
            shutil.copy(LIME_FIGURES / "lime_explanation.png", FIG / "part2_lime_explanation.png")
    except Exception as exc:
        print(f"  SKIP part2 LIME: {exc}")

    (DEL / "collection_summary.json").write_text(json.dumps(collected, indent=2), encoding="utf-8")
    print("Collected artifacts into deliverables/ (figures, analysis, checklist summaries)")


if __name__ == "__main__":
    main()
