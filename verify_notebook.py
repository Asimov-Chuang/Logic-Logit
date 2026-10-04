"""Execute the tutorial with this Python environment and save a checked copy."""
import json
import os
from pathlib import Path
import sys
import tempfile
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent
    notebook = nbformat.read(root / 'Gentle-Introduction-to-Logic-Logit.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    # Use an ephemeral kernelspec so the check cannot silently select another
    # environment's Python and does not modify the user's Jupyter setup.
    with tempfile.TemporaryDirectory(prefix='logic-logit-kernel-') as temp:
        spec = Path(temp) / 'kernels' / 'logic-logit-check'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Logic-Logit verification', 'language': 'python',
        }))
        previous = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = temp + (os.pathsep + previous if previous else '')
        try:
            NotebookClient(notebook, timeout=600, kernel_name='logic-logit-check',
                           resources={'metadata': {'path': str(root)}}).execute()
        finally:
            if previous is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = previous
    target = root / 'results' / 'executed-tutorial.ipynb'
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, target)
    print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells successfully.')
    print(target)


if __name__ == '__main__':
    main()
