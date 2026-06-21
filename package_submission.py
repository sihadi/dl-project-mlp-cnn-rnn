"""Package the DL project for submission.

Steps performed:
- run deliverables/run_all_and_collect.py if present
- copy all figure PNGs to deliverables/figures
- generate deliverables/report.html (if script exists)
- generate deliverables/report_generated.pdf using deliverables/make_pdf_from_report_and_figures.py
- create deliverables/submission_final.zip

Run this script from the project root with the same Python interpreter used for the project.
"""
import sys
import os
import subprocess
import shutil
import glob


ROOT = os.path.dirname(__file__)
PY = sys.executable


def run_script(script_rel):
    path = os.path.join(ROOT, script_rel)
    if not os.path.exists(path):
        print(f'SKIP (not found): {script_rel}')
        return 0, ''
    print('RUN ->', path)
    proc = subprocess.run([PY, path], cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print('ERR:', proc.stderr)
    return proc.returncode, proc.stdout


def copy_pngs_to_deliverables():
    dst = os.path.join(ROOT, 'deliverables', 'figures')
    os.makedirs(dst, exist_ok=True)
    files = glob.glob(os.path.join(ROOT, '**', '*.png'), recursive=True)
    copied = 0
    for f in files:
        # skip files already in deliverables/figures
        if os.path.abspath(f).startswith(os.path.abspath(dst)):
            continue
        try:
            shutil.copy(f, dst)
            copied += 1
        except Exception as e:
            print('copy failed', f, e)
    print('Copied', copied, 'PNG files to deliverables/figures')
    return copied


import zipfile


def make_zip():
    base = os.path.join(ROOT, 'deliverables')
    zip_path = os.path.join(base, 'submission_final.zip')
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except PermissionError:
            zip_path = os.path.join(base, 'submission_final_new.zip')

    include_names = {
        'report.md', 'report.html', 'report_generated.pdf', 'appendix.md',
        'submission_notebook.ipynb', 'requirements.txt', 'collection_summary.json',
    }
    include_prefixes = ('checklist_summary_',)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            for name in files:
                if name == 'submission_final.zip':
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, ROOT)
                arc = rel.replace('\\', '/')
                if name in include_names or name.startswith(include_prefixes):
                    zf.write(full, arc)
                    continue
                if '/figures/' in arc.replace('\\', '/') or '/analysis/' in arc.replace('\\', '/'):
                    zf.write(full, arc)
                    continue
                if name.endswith('.py') and '/deliverables/' in arc.replace('\\', '/'):
                    zf.write(full, arc)

        for part in ('part1_mlp', 'part2_cnn', 'part3_rnn'):
            part_dir = os.path.join(ROOT, part)
            for root, _, files in os.walk(part_dir):
                for name in files:
                    if not name.endswith(('.py', '.ipynb', '.md')):
                        continue
                    full = os.path.join(root, name)
                    arc = os.path.relpath(full, ROOT).replace('\\', '/')
                    zf.write(full, arc)

    print('Created', zip_path)
    return zip_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Package DL project for submission')
    parser.add_argument('--skip-run', action='store_true', help='Skip re-running validators')
    args = parser.parse_args()

    print('Project root:', ROOT)

    if not args.skip_run:
        run_script(os.path.join('deliverables', 'run_all_and_collect.py'))
    else:
        print('SKIP run_all_and_collect.py (--skip-run)')

    # 2. copy PNGs
    copy_pngs_to_deliverables()

    # 3. generate report.html if script exists
    run_script(os.path.join('deliverables', 'generate_report_html.py'))

    # 4. generate PDF
    run_script(os.path.join('deliverables', 'make_pdf_from_report_and_figures.py'))

    # 5. zip deliverables
    zip_path = make_zip()

    print('\nDone. Submission archive:')
    print(zip_path)


if __name__ == '__main__':
    main()
