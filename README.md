Immune Network Simulator

Author: lopzedu-arch


This code is part of the work presented in the article: Model of the macrophage-lymphocyte bridge
in the immune response. It allows the simulation of the immune network dynamics proposed within the aforementioned article.

The code is structured into three fundamental components:

    Solving of differential equations – Numerical integration of the network dynamics.

    Transformation of boolean expressions – Conversion into their fuzzy-logic probabilistic interpretation.

    Interactive interface – A graphical user interface for running and visualizing simulations.

Compatibility and Execution

There are no restrictions regarding the type of notebook or environment that can be used. However, we recommend the following options:
Option 1: Google Colab (Quick setup, cloud-based)

    Copy the entire code from the file Mixed_Network.py and paste it into a Google Colab cell.

    Run the cell.

    After execution, a link to a localhost extension (created by Gradio) will appear.

    Click on the link to launch the interactive interface.

    Note: This method is convenient for immediate visualization but may be slower (approx. 10 seconds per simulation).

Option 2: Local Environment (Recommended for speed)

Running the code on your local machine significantly improves performance:

    Expected speed: ~2 seconds per simulation (compared to ~10 seconds on Colab).

    Requirements: Python 3.8+ and the dependencies listed in requirements.txt.
