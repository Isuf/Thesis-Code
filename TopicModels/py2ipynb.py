"""Create a notebook containing code from a script.
Run as:  python make_nb.py my_script.py
"""
import sys

import nbformat
from nbformat.v4 import new_notebook, new_code_cell

python_fileName = "D:\Tema NTNU\Code\Thesis-Code\TopicModels\FindTopics.py"
ipython_fileName ="Notebooks/topic_models.ipynb"

nb = new_notebook()
with open(python_fileName) as f:
    code = f.read()

nb.cells.append(new_code_cell(code))
nbformat.write(nb, ipython_fileName)
print("IPython Notebook created Successfully") 