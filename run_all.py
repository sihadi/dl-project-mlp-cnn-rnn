from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from importlib import import_module

PARTS = [
    ('part1_mlp', 'part1_mlp.train'),
    ('part2_cnn', 'part2_cnn.train'),
    ('part3_rnn', 'part3_rnn.train'),
]


def run_all(output_root: Path = Path('outputs')):
    results = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        futures = []
        for name, module_path in PARTS:
            mod = import_module(module_path)
            futures.append(ex.submit(mod.run, output_root))
        for f in futures:
            results.append(f.result())
    return results


if __name__ == '__main__':
    print(run_all())
