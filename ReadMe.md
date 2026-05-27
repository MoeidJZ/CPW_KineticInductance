# Superconducting CPW Resonator Design & Kinetic Inductance Calculator

This repository contains tools and standard operating procedures (SOPs) for the design, calculation, and HFSS simulation of superconducting Coplanar Waveguide (CPW) resonators, with a specific focus on incorporating thin-film kinetic inductance (e.g., $TaC_xN_{1-x}$).

## Overview
This toolkit bridges the gap between layout design (Qiskit Metal), finite element simulation (Ansys HFSS), and physical property extraction. It includes:
1. **Qiskit Metal Design Script (`Qiskit_design_Q11.1_TaCN7.0.ipynb`):** A Jupyter notebook for generating chip layouts of CPW resonators and pushing the structures to Ansys HFSS for Eigenmode and Driven Modal simulations.
2. **CPW Resonance & Impedance GUI (`Resonance_Frequencies_GUI.py`):** A Python Tkinter interface to calculate resonance frequencies, impedance, and equivalent $L_{eq}/C_{eq}$ circuits by factoring in geometry, measurement temperature, and film-specific kinetic inductance.
3. **HFSS Meshing SOP (`Meshing_CPW_resonators.docx`):** A proven setup guide to ensure eigenvalue convergence in HFSS.

## Video Tutorials
For a full walkthrough on designing these resonators using Qiskit Metal and setting up the HFSS simulations, please refer to NYU Nano lab's video guides:
* [Part 1: Designing CPW Resonators in Qiskit Metal](https://www.youtube.com/watch?v=9INNvUQs3GM&list=PLGlyoYYcG7gD5o5cx2g5c_EkcxhC6WyG_&index=10)
* [Part 2: Simulating CPW Resonators in Ansys HFSS](https://www.youtube.com/watch?v=qX1Hd2kUm18&list=PLGlyoYYcG7gD5o5cx2g5c_EkcxhC6WyG_&index=11)

### Important Design Note (Coupling Length)
> **Note:** If you are following the video tutorials linked above, please be aware of a small clarification: The physical lengths detailed in the calculations and simulations *must* include the total coupling length of the resonator alongside the feedline. This detail was inadvertently omitted in the videos but is critical for accurate frequency targeting.

## Using the Python GUI
The `Resonance_Frequencies_GUI.py` script requires `numpy`, `scipy`, `pandas`, and (optionally) `mpmath` for precise elliptical integral evaluation.

The tool operates in two modes:
* **Mode 1 (Simulation Data Extraction):** Input your drawn physical lengths and your *simulated* resonance frequencies. **Note: These simulated frequencies must be derived directly from HFSS simulating only the geometric properties, and explicitly do not consider kinetic inductance.** The tool calculates the effective dielectric constant ($\epsilon_{eff}$) and maps out the expected downward frequency shift once the thin-film kinetic inductance is applied.
* **Mode 2 (Physical Length Target Generator):** Provide a known $\epsilon_{eff}$ and your **Target Shifted Frequencies** (the frequency you actually want to measure at cryogenic temperatures). The tool will back-calculate the exact physical lengths you need to draw in Qiskit Metal, as well as the simulated target frequencies you should expect to see in your HFSS setup (which inherently lack kinetic inductance).

### Theory & Equivalent Circuits
The GUI automatically provides the equivalent parallel lumped $L_{eq}$ and $C_{eq}$ values for the fundamental mode of your resonators. 
* **$\lambda/2$ Resonators:** $L_{eq} = \frac{2 L_{total}}{\pi^2}$, $C_{eq} = \frac{C_{total}}{2}$
* **$\lambda/4$ Resonators:** $L_{eq} = \frac{8 L_{total}}{\pi^2}$, $C_{eq} = \frac{C_{total}}{2}$

## References
This codebase is heavily rooted in the following literature:
* **CPW Circuit Equivalents:** Göppl, M., et al. "Coplanar waveguide resonators for circuit quantum electrodynamics." *Journal of Applied Physics* 104.11 (2008): 113904. [DOI: 10.1063/1.3010859](https://pubs.aip.org/aip/jap/article-abstract/104/11/113904/145728/Coplanar-waveguide-resonators-for-circuit-quantum?redirectedFrom=fulltext)
* **Kinetic Inductance Modeling:** "Investigation of kinetic inductance and microwave loss in superconducting TaCxN1−x resonators." *Applied Physics Letters* 127.19 (2025): 192603. [DOI: 10.1063/5.0234721](https://pubs.aip.org/aip/apl/article-abstract/127/19/192603/3372272/Investigation-of-kinetic-inductance-and-microwave?redirectedFrom=fulltext)
