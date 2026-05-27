import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from dataclasses import dataclass
from typing import Literal, Union
from scipy.constants import mu_0 as mu0, epsilon_0 as eps0, hbar, k as kB, c
import pandas as pd

try:
    from mpmath import ellipk
    USE_MPMATH = True
except ImportError:
    USE_MPMATH = False

Number = Union[float, int, np.ndarray]

@dataclass
class Film:
    Tc: Number
    rho: Number
    t: Number
    T: Number = 50e-3

@dataclass
class CPW:
    w: Number
    s: Number

@dataclass
class Design:
    mode: Literal["half", "quarter"] = "half"

def elliptic_k_approx(k):
    if k == 0: return np.pi / 2
    if k == 1: return np.inf
    a, b = 1, np.sqrt(1 - k**2)
    for _ in range(10):
        a_new = (a + b) / 2
        b = np.sqrt(a * b)
        a = a_new
        if abs(a - b) < 1e-15: break
    return np.pi / (2 * a)

def _to_array(x, N):
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = np.full(N, float(arr))
    return arr

def _cpw_elliptics(w: np.ndarray, s: np.ndarray):
    k0 = w / (w + 2*s)
    k0p = np.sqrt(1.0 - k0**2)
    if USE_MPMATH:
        K = np.array([float(ellipk(k**2)) for k in k0])
        Kp = np.array([float(ellipk(kp**2)) for kp in k0p])
    else:
        K = np.array([elliptic_k_approx(k) for k in k0])
        Kp = np.array([elliptic_k_approx(kp) for kp in k0p])
    return K, Kp

def _Ll_per_len(w, s):
    K, Kp = _cpw_elliptics(w, s)
    return mu0/4.0 * (Kp / K)

def _Cl_per_len(eps_eff, w, s):
    K, Kp = _cpw_elliptics(w, s)
    return 4.0 * eps0 * eps_eff * (K / Kp)

def _delta_gap(Tc):
    return 1.764 * kB * Tc

def _tanh_factor(T, Tc):
    Δ = _delta_gap(Tc)
    return np.tanh(Δ / (2*kB*T))

def _Lk_per_square(rho, t, T, Tc):
    Δ = _delta_gap(Tc)
    tf = _tanh_factor(T, Tc)
    Rn_square = rho / t 
    return hbar * Rn_square / (np.pi * Δ * tf)

def _factor(mode: str):
    return 2.0 if mode == "half" else 4.0

def design_from_lengths_and_simfr(N, cpw, film, design, lengths_m, fr_sim_Hz):
    l = _to_array(lengths_m, N)
    fr = _to_array(fr_sim_Hz, N)
    w = _to_array(cpw.w, N)
    s = _to_array(cpw.s, N)
    F = _factor(design.mode)
    
    eps_eff = (c / (F * l * fr))**2
    Ll = _Ll_per_len(w, s)
    Cl = _Cl_per_len(eps_eff, w, s)
    
    Lgeo = Ll * l
    C = Cl * l
    Lk_sq = _Lk_per_square(film.rho, film.t, film.T, film.Tc)
    Lk = Lk_sq * (l / w)
    
    fr_shifted = 1.0 / (F * np.sqrt((Lgeo + Lk) * C))
    Zr = np.sqrt((Lk+Lgeo) / C)
    alpha = Lk / (Lgeo+Lk)

    if design.mode == "half":
        Leq = 2.0 * (Lgeo + Lk) / (np.pi**2)
    else:
        Leq = 8.0 * (Lgeo + Lk) / (np.pi**2)
    Ceq = C / 2.0

    return {
        "eps_eff": eps_eff, "Ll_per_len_Hperm": Ll, "Cl_per_len_Fperm": Cl,
        "L_H": Lgeo, "C_F": C, "Lk_H": Lk, "Lk_per_square_H": Lk_sq,
        "fr_sim_Hz": fr, "fr_shifted_Hz": fr_shifted, "Zr_ohm": Zr, "alpha": alpha,
        "Leq_H": Leq, "Ceq_F": Ceq
    }

def design_from_shifted_fr_and_eps(N, cpw, film, design, fr_shifted_target_Hz, eps_eff):
    fr_s = _to_array(fr_shifted_target_Hz, N)
    ee = _to_array(eps_eff, N)
    w = _to_array(cpw.w, N)
    s = _to_array(cpw.s, N)
    F = _factor(design.mode)

    Ll = _Ll_per_len(w, s)
    Cl = _Cl_per_len(ee, w, s)
    Lk_sq = _Lk_per_square(film.rho, film.t, film.T, film.Tc)
    
    L_total_per_len = Ll + (Lk_sq / w)
    l = 1.0 / (F * fr_s * np.sqrt(L_total_per_len * Cl))
    
    Lgeo = Ll * l
    C = Cl * l
    Lk = Lk_sq * (l / w)
    
    fr_sim = 1.0 / (F * np.sqrt(Lgeo * C))
    Zr = np.sqrt((Lgeo+Lk) / C)
    alpha = Lk / (Lk+Lgeo)
    
    if design.mode == "half":
        Leq = 2.0 * (Lgeo + Lk) / (np.pi**2)
    else:
        Leq = 8.0 * (Lgeo + Lk) / (np.pi**2)
    Ceq = C / 2.0

    return {
        "length_m": l, "Ll_per_len_Hperm": Ll, "Cl_per_len_Fperm": Cl,
        "L_H": Lgeo, "C_F": C, "fr_sim_Hz": fr_sim, "Lk_H": Lk,
        "Lk_per_square_H": Lk_sq, "fr_shifted_Hz": fr_s, "Zr_ohm": Zr, "alpha": alpha,
        "Leq_H": Leq, "Ceq_F": Ceq
    }

class CPWResonatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CPW Resonator Analysis Tool")
        self.root.geometry("1400x900")
        
        self.mode = tk.StringVar(value="calculate_eps")
        self.design_mode = tk.StringVar(value="quarter")
        self.num_resonators = tk.IntVar(value=8)
        
        self.Tc = tk.DoubleVar(value=6.0)
        self.rho = tk.DoubleVar(value=215.0)
        self.t = tk.DoubleVar(value=45.0)
        self.T = tk.DoubleVar(value=50e-3)
        
        self.w = tk.DoubleVar(value=10.0)
        self.s = tk.DoubleVar(value=6.3)
        
        self.lengths_str = tk.StringVar(value="4.51, 4.236, 4.0, 3.755, 3.55, 3.35, 3.17, 3.0")
        self.sim_freqs_str = tk.StringVar(value="6.17, 6.52, 6.9, 7.3, 7.7, 8.14, 8.52, 9.0")
        self.target_freqs_str = tk.StringVar(value="6.0, 6.3, 6.6, 6.9, 7.2, 7.5, 7.8, 8.1")
        self.eps_eff = tk.DoubleVar(value=6.5)
        
        self.results = None
        self.create_widgets()
        
    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="Input Parameters")
        
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        theory_frame = ttk.Frame(notebook)
        notebook.add(theory_frame, text="Theory & Equations")
        
        self.create_input_widgets(input_frame)
        self.create_results_widgets(results_frame)
        self.create_theory_widgets(theory_frame)
        
    def create_input_widgets(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        mode_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Calculate ε_eff from Simulation Data", variable=self.mode, value="calculate_eps").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Calculate Physical Length from Target Shifted Frequency (Provided ε_eff)", variable=self.mode, value="use_eps").pack(anchor=tk.W)
        
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Parameters", padding=10)
        basic_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(basic_frame, text="Number of Resonators:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(basic_frame, textvariable=self.num_resonators, width=15).grid(row=0, column=1, padx=5)
        ttk.Label(basic_frame, text="Design Mode:").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Combobox(basic_frame, textvariable=self.design_mode, values=["half", "quarter"], width=12).grid(row=0, column=3, padx=5)
        
        film_frame = ttk.LabelFrame(scrollable_frame, text="Film Parameters", padding=10)
        film_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(film_frame, text="Tc (K):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(film_frame, textvariable=self.Tc, width=15).grid(row=0, column=1, padx=5)
        ttk.Label(film_frame, text="ρ just above Tc (μΩ·cm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(film_frame, textvariable=self.rho, width=15).grid(row=0, column=3, padx=5)
        ttk.Label(film_frame, text="Thickness (nm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(film_frame, textvariable=self.t, width=15).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(film_frame, text="Resonator Operating Temp (K):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(film_frame, textvariable=self.T, width=15).grid(row=1, column=3, padx=5, pady=5)
        
        cpw_frame = ttk.LabelFrame(scrollable_frame, text="CPW Parameters", padding=10)
        cpw_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(cpw_frame, text="Center Width w (μm):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(cpw_frame, textvariable=self.w, width=15).grid(row=0, column=1, padx=5)
        ttk.Label(cpw_frame, text="Gap s (μm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(cpw_frame, textvariable=self.s, width=15).grid(row=0, column=3, padx=5)
        
        arrays_frame = ttk.LabelFrame(scrollable_frame, text="Input Arrays", padding=10)
        arrays_frame.pack(fill=tk.X, padx=5, pady=5)
        
        calc_frame = ttk.LabelFrame(arrays_frame, text="Mode 1: Calculate ε_eff", padding=10)
        calc_frame.pack(fill=tk.X, pady=5)
        ttk.Label(calc_frame, text="Physical Lengths (mm) - comma separated:").pack(anchor=tk.W)
        ttk.Entry(calc_frame, textvariable=self.lengths_str, width=80).pack(fill=tk.X, pady=2)
        ttk.Label(calc_frame, text="HFSS Simulated Frequencies (GHz) - comma separated:").pack(anchor=tk.W, pady=(10,0))
        ttk.Entry(calc_frame, textvariable=self.sim_freqs_str, width=80).pack(fill=tk.X, pady=2)
        
        use_frame = ttk.LabelFrame(arrays_frame, text="Mode 2: Use ε_eff to find lengths", padding=10)
        use_frame.pack(fill=tk.X, pady=5)
        ttk.Label(use_frame, text="ε_eff:").pack(anchor=tk.W)
        ttk.Entry(use_frame, textvariable=self.eps_eff, width=15).pack(anchor=tk.W, pady=2)
        ttk.Label(use_frame, text="Target Shifted Frequencies w/ Kinetic Inductance (GHz) - comma separated:").pack(anchor=tk.W, pady=(10,0))
        ttk.Entry(use_frame, textvariable=self.target_freqs_str, width=80).pack(fill=tk.X, pady=2)
        
        ttk.Button(scrollable_frame, text="Calculate", command=self.calculate).pack(pady=20)
        self.status_label = ttk.Label(scrollable_frame, text="Ready", foreground="green")
        self.status_label.pack(pady=5)
        
    def create_results_widgets(self, parent):
        columns = ('Res', 'Length (mm)', 'ε_eff', 'Simulated fr (GHz)', 'Shifted fr (GHz)', 
                  'Lk (nH)', 'L□ (pH/□)', 'Z (Ω)', 'α', 'Leq (nH)', 'Ceq (pF)')
        self.results_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=90, anchor=tk.CENTER)
            
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set)
        
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        summary_frame = ttk.LabelFrame(parent, text="Summary Statistics", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        self.summary_text = tk.Text(summary_frame, height=5, wrap=tk.WORD)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Button(parent, text="Export to CSV", command=self.export_results).pack(pady=10)

    def create_theory_widgets(self, parent):
        theory_text = tk.Text(parent, wrap=tk.WORD, padx=20, pady=20, font=("Arial", 11))
        theory_text.pack(fill=tk.BOTH, expand=True)
        
        content = """CPW Resonator Theory & Equivalent Circuits

1. Coplanar Waveguide Base Equations
The geometric inductance (Ll) and capacitance (Cl) per unit length are governed by the geometry (w, s) and the effective dielectric constant (ε_eff). 

2. Kinetic Inductance
Total inductance includes the kinetic contribution of the superconducting film:
L_total = L_geo + L_k
L_k = L_k_sq * (l / w)

*Important Note on Variables:* L_k_sq (pH/□) is determined by the film thickness, Tc, the normal-state resistivity (ρ) measured just above the superconducting transition, and the actual operational temperature of the resonator during the microwave measurement.

3. Resonance Shift
The fractional frequency shift due to kinetic inductance is characterized by the kinetic inductance fraction (α).
α = L_k / (L_geo + L_k)
f_shifted = f_simulated * sqrt(1 - α)

4. Lumped Element Equivalent Circuits (First Mode)
Near the fundamental resonance, distributed transmission lines can be approximated by a parallel LC equivalent circuit.

For Half-Wavelength (λ/2) Resonators:
• Leq = 2 * L_total / (π^2)
• Ceq = C_total / 2

For Quarter-Wavelength (λ/4) Resonators:
• Leq = 8 * L_total / (π^2)
• Ceq = C_total / 2

References: 
- Göppl et al., J. Appl. Phys. 104, 113904 (2008)
- APL kinetic inductance papers (e.g., TaCxN1-x modeling).
"""
        theory_text.insert(tk.END, content)
        theory_text.config(state=tk.DISABLED)
        
    def parse_array(self, text):
        return np.array([float(x.strip()) for x in text.split(',')])
    
    def calculate(self):
        try:
            self.status_label.config(text="Calculating...", foreground="orange")
            self.root.update()
            
            N = self.num_resonators.get()
            cpw = CPW(w=self.w.get()*1e-6, s=self.s.get()*1e-6)
            film = Film(Tc=self.Tc.get(), rho=self.rho.get()*1e-8, t=self.t.get()*1e-9, T=self.T.get())
            design = Design(mode=self.design_mode.get())
            
            if self.mode.get() == "calculate_eps":
                lengths = self.parse_array(self.lengths_str.get()) * 1e-3
                sim_freqs = self.parse_array(self.sim_freqs_str.get()) * 1e9
                results = design_from_lengths_and_simfr(N, cpw, film, design, lengths, sim_freqs)
            else:
                target_freqs = self.parse_array(self.target_freqs_str.get()) * 1e9
                eps_eff = self.eps_eff.get()
                results = design_from_shifted_fr_and_eps(N, cpw, film, design, target_freqs, eps_eff)
                
            self.results = []
            Lk_sq_val = results['Lk_per_square_H'] if np.isscalar(results['Lk_per_square_H']) else results['Lk_per_square_H'][0]
            
            for i in range(N):
                l_val = lengths[i] if self.mode.get() == "calculate_eps" else results['length_m'][i]
                eps_val = results['eps_eff'][i] if self.mode.get() == "calculate_eps" else eps_eff
                
                self.results.append({
                    'resonator': i + 1, 'length': l_val, 'eps_eff': eps_val,
                    'fr_sim': results['fr_sim_Hz'][i], 'fr_shifted': results['fr_shifted_Hz'][i],
                    'Lk': results['Lk_H'][i], 'Lk_sq': Lk_sq_val, 'Zr': results['Zr_ohm'][i],
                    'alpha': results['alpha'][i], 'Leq': results['Leq_H'][i], 'Ceq': results['Ceq_F'][i]
                })
            
            self.display_results()
            self.status_label.config(text="Success", foreground="green")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Error", str(e))
    
    def display_results(self):
        for item in self.results_tree.get_children(): self.results_tree.delete(item)
        for r in self.results:
            values = (
                r['resonator'], f"{r['length']*1000:.3f}", f"{r['eps_eff']:.3f}",
                f"{r['fr_sim']/1e9:.3f}", f"{r['fr_shifted']/1e9:.3f}",
                f"{r['Lk']*1e9:.2f}", f"{r['Lk_sq']*1e12:.2f}",
                f"{r['Zr']:.1f}", f"{r['alpha']:.3f}", 
                f"{r['Leq']*1e9:.2f}", f"{r['Ceq']*1e12:.3f}"
            )
            self.results_tree.insert('', tk.END, values=values)
            
        avg_eps_eff = np.mean([r['eps_eff'] for r in self.results])
        avg_alpha = np.mean([r['alpha'] for r in self.results])
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, f"Avg ε_eff: {avg_eps_eff:.2f} | Avg α: {avg_alpha:.3f} | L□: {self.results[0]['Lk_sq']*1e12:.2f} pH/□")
    
    def export_results(self):
        if not self.results: return
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(defaultextension=".csv")
        if filename: pd.DataFrame(self.results).to_csv(filename, index=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = CPWResonatorGUI(root)
    root.mainloop()