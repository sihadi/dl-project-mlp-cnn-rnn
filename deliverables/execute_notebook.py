import nbformat
from nbclient import NotebookClient
import os

BASE = os.path.dirname(__file__)
in_nb = os.path.join(BASE, 'submission_notebook.ipynb')
out_nb = os.path.join(BASE, 'submission_notebook_executed.ipynb')

print('Executing', in_nb)
nb = nbformat.read(in_nb, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()
nbformat.write(nb, out_nb)
print('Wrote', out_nb)
