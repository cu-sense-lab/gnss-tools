"""
Author: Brian Breitsch
Date: 2025-01-02
"""

import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt

def mermaid(graph: str):
    '''Display a mermaid graph using the `mermaid.ink` service'''
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))
