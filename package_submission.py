"""Package the DL project into submission_final_new.zip.

Steps:
- run deliverables/run_all_and_collect.py (unless --skip-run)
- copy figure PNGs into deliverables/figures
- create deliverables/submission_final_new.zip

Run from the project root:
    python package_submission.py
    python package_submission.py --skip-run
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(__file__)
PY = sys.executable
ZIP_NAME = "submission_final_new.zip"


def run_script(script_rel: str) -> int:
    path = os.path.join(ROOT, script_rel)
    if not os.path.exists(path):
        print(f"SKIP (not found): {script_rel}")
        return 0
    print("RUN ->", path)
    proc = subprocess.run([PY, path], cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("ERR:", proc.stderr)
    return proc.returncode


def copy_pngs_to_deliverables() -> int:
    dst = os.path.join(ROOT, "deliverables", "figures")
    os.makedirs(dst, exist_ok=True)
    copied = 0
    for f in glob.glob(os.path.join(ROOT, "**", "*.png"), recursive=True):
        if os.path.abspath(f).startswith(os.path.abspath(dst)):
            continue
        try:
            shutil.copy(f, dst)
            copied += 1
        except OSError as exc:
            print("copy failed", f, exc)
    print("Copied", copied, "PNG files to deliverables/figures")
    return copied


def make_zip() -> str:
    base = os.path.join(ROOT, "deliverables")
    zip_path = os.path.join(base, ZIP_NAME)
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except PermissionError:
            zip_path = os.path.join(base, "submission_final_new_tmp.zip")

    include_names = {
        "report.md",
        "appendix.md",
        "submission_notebook.ipynb",
        "requirements.txt",
        "collection_summary.json",
    }
    include_prefixes = ("checklist_summary_",)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            for name in files:
                if name.endswith(".zip"):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, ROOT)
                arc = rel.replace("\\", "/")
                if name in include_names or name.startswith(include_prefixes):
                    zf.write(full, arc)
                    continue
                if "/figures/" in arc.replace("\\", "/") or "/analysis/" in arc.replace("\\", "/"):
                    zf.write(full, arc)
                    continue
                if name.endswith(".py") and "/deliverables/" in arc.replace("\\", "/"):
                    zf.write(full, arc)

        for part in ("part1_mlp", "part2_cnn", "part3_rnn"):
            part_dir = os.path.join(ROOT, part)
            for root, _, files in os.walk(part_dir):
                for name in files:
                    if not name.endswith((".py", ".ipynb", ".md")):
                        continue
                    full = os.path.join(root, name)
                    arc = os.path.relpath(full, ROOT).replace("\\", "/")
                    zf.write(full, arc)

        for extra in ("README.md", "requirements.txt", "run_project.ps1", "run_all.py"):
            full = os.path.join(ROOT, extra)
            if os.path.exists(full):
                zf.write(full, extra.replace("\\", "/"))

    print("Created", zip_path)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package DL project for submission")
    parser.add_argument("--skip-run", action="store_true", help="Skip re-running validators")
    args = parser.parse_args()

    print("Project root:", ROOT)

    if not args.skip_run:
        if run_script(os.path.join("deliverables", "run_all_and_collect.py")) != 0:
            raise SystemExit("run_all_and_collect.py failed")
    else:
        print("SKIP run_all_and_collect.py (--skip-run)")

    copy_pngs_to_deliverables()
    zip_path = make_zip()

    print("\nDone. Submission archive:")
    print(zip_path)


if __name__ == "__main__":
    main()
