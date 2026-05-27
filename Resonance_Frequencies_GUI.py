import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Union
from scipy.constants import mu_0 as mu0, epsilon_0 as eps0, hbar, k as kB, c
import pandas as pd
import mpmath

# Try to import mpmath, fallback to approximation if not available
try:
    from mpmath import ellipk
    USE_MPMATH = True
except ImportError:
    USE_MPMATH = False
    print("Warning: mpmath not found. Using approximation for elliptic integrals.")

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
    """Approximation for complete elliptic integral K(k) when mpmath is not available"""
    if k == 0:
        return np.pi / 2
    if k == 1:
        return np.inf
    
    # Arithmetic-geometric mean approximation
    a, b = 1, np.sqrt(1 - k**2)
    for _ in range(10):  # Usually converges quickly
        a_new = (a + b) / 2
        b = np.sqrt(a * b)
        a = a_new
        if abs(a - b) < 1e-15:
            break
    
    return np.pi / (2 * a)

def _to_array(x, N):
    """Broadcast scalars to length-N arrays."""
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

def _Rn(rho, l, t, w):
    return rho * l / (t * w)

def _Lk_total(rho, l, t, w, T, Tc):
    Δ = _delta_gap(Tc)
    tf = _tanh_factor(T, Tc)
    Rn = _Rn(rho, l, t, w)
    return hbar * Rn / (np.pi * Δ * tf)

def _Lk_per_square(rho, t, T, Tc):
    """Calculate kinetic inductance per square (pH/square)"""
    Δ = _delta_gap(Tc)
    tf = _tanh_factor(T, Tc)
    Rn_square = rho / t  # Sheet resistance
    Lk_square = hbar * Rn_square / (np.pi * Δ * tf)
    return Lk_square

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

    fr_test = 1.0 / (F * np.sqrt(Lgeo * C))
    Lk = _Lk_total(film.rho, l, film.t, w, film.T, film.Tc)
    Lk_sq = _Lk_per_square(film.rho, film.t, film.T, film.Tc)
    fr_shifted = 1.0 / (F * np.sqrt((Lgeo + Lk) * C))
    Zr = np.sqrt((Lk+Lgeo) / C)
    alpha = Lk / (Lgeo+Lk)


    return {
        "eps_eff": eps_eff,
        "Ll_per_len_Hperm": Ll,
        "Cl_per_len_Fperm": Cl,
        "L_H": Lgeo, "C_F": C,
        "fr_test_Hz": fr_test,
        "Lk_H": Lk,
        "Lk_per_square_H": Lk_sq,
        "fr_shifted_Hz": fr_shifted,
        "Zr_ohm": Zr,
        "alpha": alpha
    }

def design_from_target_fr_and_eps(N, cpw, film, design, fr_target_Hz, eps_eff):
    fr = _to_array(fr_target_Hz, N)
    ee = _to_array(eps_eff, N)
    w = _to_array(cpw.w, N)
    s = _to_array(cpw.s, N)

    F = _factor(design.mode)
    l = c / (F * fr * np.sqrt(ee))

    Ll = _Ll_per_len(w, s)
    Cl = _Cl_per_len(ee, w, s)

    Lgeo = Ll * l
    C = Cl * l

    fr_bare = 1.0 / (F * np.sqrt(Lgeo * C))
    Lk = _Lk_total(film.rho, l, film.t, w, film.T, film.Tc)
    Lk_sq = _Lk_per_square(film.rho, film.t, film.T, film.Tc)
    fr_with_kinetic = 1.0 / (F * np.sqrt((Lgeo + Lk) * C))
    Zr = np.sqrt((Lgeo+Lk) / C)
    alpha = Lk / (Lk+Lgeo)

    return {
        "length_m": l,
        "Ll_per_len_Hperm": Ll,
        "Cl_per_len_Fperm": Cl,
        "L_H": Lgeo, "C_F": C,
        "fr_bare_Hz": fr_bare,
        "Lk_H": Lk,
        "Lk_per_square_H": Lk_sq,
        "fr_with_kinetic_Hz": fr_with_kinetic,
        "Zr_ohm": Zr,
        "alpha": alpha
    }

class CPWResonatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CPW Resonator Analysis Tool")
        self.root.geometry("1400x900")
        
        # Variables
        self.mode = tk.StringVar(value="calculate_eps")
        self.design_mode = tk.StringVar(value="half")
        self.num_resonators = tk.IntVar(value=8)
        
        # Film parameters
        self.Tc = tk.DoubleVar(value=6.0)
        self.rho = tk.DoubleVar(value=215.0)  # μΩ.cm
        self.t = tk.DoubleVar(value=15.0)     # nm
        self.T = tk.DoubleVar(value=50e-3)
        
        # CPW parameters
        self.w = tk.DoubleVar(value=10.0)     # μm
        self.s = tk.DoubleVar(value=6.3)      # μm
        
        # Array inputs
        self.lengths_str = tk.StringVar(value="4.51, 4.236, 4.0, 3.755, 3.55, 3.35, 3.17, 3.0")  # mm
        self.sim_freqs_str = tk.StringVar(value="6.17, 6.52, 6.9, 7.3, 7.7, 8.14, 8.52, 9.0")    # GHz
        self.target_freqs_str = tk.StringVar(value="6.17, 6.52, 6.9, 7.3, 7.7, 8.14, 8.52, 9.0")  # GHz
        self.eps_eff = tk.DoubleVar(value=6.5)
        
        self.results = None
        self.create_widgets()
        
    def create_widgets(self):
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input frame
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="Input Parameters")
        
        # Results frame
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        self.create_input_widgets(input_frame)
        self.create_results_widgets(results_frame)
        
    def create_input_widgets(self, parent):
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mode selection
        mode_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Calculate ε_eff from simulation data", 
                       variable=self.mode, value="calculate_eps").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Use provided ε_eff value", 
                       variable=self.mode, value="use_eps").pack(anchor=tk.W)
        
        # Basic parameters
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Parameters", padding=10)
        basic_frame.pack(fill=tk.X, padx=5, pady=5)
        
        basic_grid = ttk.Frame(basic_frame)
        basic_grid.pack(fill=tk.X)
        
        ttk.Label(basic_grid, text="Number of Resonators:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(basic_grid, textvariable=self.num_resonators, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(basic_grid, text="Design Mode:").grid(row=0, column=2, sticky=tk.W, padx=5)
        mode_combo = ttk.Combobox(basic_grid, textvariable=self.design_mode, 
                                 values=["half", "quarter"], width=12)
        mode_combo.grid(row=0, column=3, padx=5)
        
        # Film parameters
        film_frame = ttk.LabelFrame(scrollable_frame, text="Film Parameters", padding=10)
        film_frame.pack(fill=tk.X, padx=5, pady=5)
        
        film_grid = ttk.Frame(film_frame)
        film_grid.pack(fill=tk.X)
        
        ttk.Label(film_grid, text="Tc (K):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(film_grid, textvariable=self.Tc, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(film_grid, text="ρ (μΩ·cm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(film_grid, textvariable=self.rho, width=15).grid(row=0, column=3, padx=5)
        
        ttk.Label(film_grid, text="Thickness (nm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(film_grid, textvariable=self.t, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(film_grid, text="Temperature (K):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(film_grid, textvariable=self.T, width=15).grid(row=1, column=3, padx=5, pady=5)
        
        # CPW parameters
        cpw_frame = ttk.LabelFrame(scrollable_frame, text="CPW Parameters", padding=10)
        cpw_frame.pack(fill=tk.X, padx=5, pady=5)
        
        cpw_grid = ttk.Frame(cpw_frame)
        cpw_grid.pack(fill=tk.X)
        
        ttk.Label(cpw_grid, text="Center Width w (μm):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(cpw_grid, textvariable=self.w, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(cpw_grid, text="Gap s (μm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(cpw_grid, textvariable=self.s, width=15).grid(row=0, column=3, padx=5)
        
        # Input arrays frame
        arrays_frame = ttk.LabelFrame(scrollable_frame, text="Input Arrays", padding=10)
        arrays_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Mode 1: Calculate eps_eff
        calc_frame = ttk.LabelFrame(arrays_frame, text="Calculate ε_eff Mode", padding=10)
        calc_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(calc_frame, text="Lengths (mm) - comma separated:").pack(anchor=tk.W)
        ttk.Entry(calc_frame, textvariable=self.lengths_str, width=80).pack(fill=tk.X, pady=2)
        
        ttk.Label(calc_frame, text="Simulated Frequencies (GHz) - comma separated:").pack(anchor=tk.W, pady=(10,0))
        ttk.Entry(calc_frame, textvariable=self.sim_freqs_str, width=80).pack(fill=tk.X, pady=2)
        
        # Mode 2: Use eps_eff
        use_frame = ttk.LabelFrame(arrays_frame, text="Use ε_eff Mode", padding=10)
        use_frame.pack(fill=tk.X, pady=5)
        
        eps_grid = ttk.Frame(use_frame)
        eps_grid.pack(fill=tk.X)
        
        ttk.Label(eps_grid, text="ε_eff:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(eps_grid, textvariable=self.eps_eff, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(use_frame, text="Target Frequencies (GHz) - comma separated:").pack(anchor=tk.W, pady=(10,0))
        ttk.Entry(use_frame, textvariable=self.target_freqs_str, width=80).pack(fill=tk.X, pady=2)
        
        # Calculate button
        calc_button = ttk.Button(scrollable_frame, text="Calculate", command=self.calculate, 
                               style="Accent.TButton")
        calc_button.pack(pady=20)
        
        # Status label
        self.status_label = ttk.Label(scrollable_frame, text="Ready", foreground="green")
        self.status_label.pack(pady=5)
        
    def create_results_widgets(self, parent):
        # Results display
        results_label = ttk.Label(parent, text="Results Table", font=("Arial", 12, "bold"))
        results_label.pack(pady=10)
        
        # Treeview for results
        columns = ('Resonator', 'Length (mm)', 'ε_eff', 'L (nH)', 'C (pF)', 
                  'Lk (nH)', 'L□ (pH/□)', 'f_shifted (GHz)', 'Z (Ω)', 'α')
        
        self.results_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == 'L□ (pH/□)':
                self.results_tree.column(col, width=100, anchor=tk.CENTER)
            else:
                self.results_tree.column(col, width=110, anchor=tk.CENTER)
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Summary frame
        summary_frame = ttk.LabelFrame(parent, text="Summary Statistics", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=5, wrap=tk.WORD)
        summary_scroll = ttk.Scrollbar(summary_frame, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export button
        export_button = ttk.Button(parent, text="Export to CSV", command=self.export_results)
        export_button.pack(pady=10)
        
    def parse_array(self, text):
        """Parse comma-separated string into numpy array"""
        try:
            values = [float(x.strip()) for x in text.split(',')]
            return np.array(values)
        except:
            raise ValueError("Invalid array format")
    
    def calculate(self):
        try:
            self.status_label.config(text="Calculating...", foreground="orange")
            self.root.update()
            
            # Get parameters (convert units to SI)
            N = self.num_resonators.get()
            
            cpw = CPW(w=self.w.get()*1e-6, s=self.s.get()*1e-6)  # μm to m
            film = Film(Tc=self.Tc.get(), 
                       rho=self.rho.get()*1e-8,  # μΩ·cm to Ω·m
                       t=self.t.get()*1e-9,      # nm to m
                       T=self.T.get())
            design = Design(mode=self.design_mode.get())
            
            if self.mode.get() == "calculate_eps":
                # Mode 1: Calculate eps_eff
                lengths = self.parse_array(self.lengths_str.get()) * 1e-3  # mm to m
                sim_freqs = self.parse_array(self.sim_freqs_str.get()) * 1e9  # GHz to Hz
                
                if len(lengths) != N or len(sim_freqs) != N:
                    raise ValueError(f"Arrays must have {N} elements each")
                
                results = design_from_lengths_and_simfr(N, cpw, film, design, lengths, sim_freqs)
                
                # Prepare results for display
                self.results = []
                # Lk_sq is a scalar (material property), extract it once
                Lk_sq_value = results['Lk_per_square_H'] if np.isscalar(results['Lk_per_square_H']) else results['Lk_per_square_H'][0]
                
                for i in range(N):
                    self.results.append({
                        'resonator': i + 1,
                        'length': lengths[i],
                        'eps_eff': results['eps_eff'][i],
                        'L': results['L_H'][i],
                        'C': results['C_F'][i],
                        'Lk': results['Lk_H'][i],
                        'Lk_sq': Lk_sq_value,
                        'fr_shifted': results['fr_shifted_Hz'][i],
                        'Zr': results['Zr_ohm'][i],
                        'alpha': results['alpha'][i]
                    })
                    
            else:
                # Mode 2: Use eps_eff
                target_freqs = self.parse_array(self.target_freqs_str.get()) * 1e9  # GHz to Hz
                eps_eff = self.eps_eff.get()
                
                if len(target_freqs) != N:
                    raise ValueError(f"Target frequencies array must have {N} elements")
                
                results = design_from_target_fr_and_eps(N, cpw, film, design, target_freqs, eps_eff)
                
                # Prepare results for display
                self.results = []
                # Lk_sq is a scalar (material property), extract it once
                Lk_sq_value = results['Lk_per_square_H'] if np.isscalar(results['Lk_per_square_H']) else results['Lk_per_square_H'][0]
                
                for i in range(N):
                    self.results.append({
                        'resonator': i + 1,
                        'length': results['length_m'][i],
                        'eps_eff': eps_eff,
                        'L': results['L_H'][i],
                        'C': results['C_F'][i],
                        'Lk': results['Lk_H'][i],
                        'Lk_sq': Lk_sq_value,
                        'fr_shifted': results['fr_with_kinetic_Hz'][i],
                        'Zr': results['Zr_ohm'][i],
                        'alpha': results['alpha'][i]
                    })
            
            self.display_results()
            self.status_label.config(text="Calculation completed successfully", foreground="green")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Calculation Error", str(e))
    
    def display_results(self):
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Populate results
        for result in self.results:
            values = (
                result['resonator'],
                f"{result['length']*1000:.3f}",  # mm
                f"{result['eps_eff']:.2f}",
                f"{result['L']*1e9:.2f}",  # nH
                f"{result['C']*1e12:.2f}",  # pF
                f"{result['Lk']*1e9:.2f}",  # nH
                f"{result['Lk_sq']*1e12:.2f}",  # pH/square
                f"{result['fr_shifted']/1e9:.3f}",  # GHz
                f"{result['Zr']:.1f}",
                f"{result['alpha']:.3f}"
            )
            self.results_tree.insert('', tk.END, values=values)
        
        # Update summary
        self.update_summary()
    
    def update_summary(self):
        if not self.results:
            return
        
        avg_eps_eff = np.mean([r['eps_eff'] for r in self.results])
        avg_alpha = np.mean([r['alpha'] for r in self.results])
        Lk_sq = self.results[0]['Lk_sq']  # Same for all resonators (material property)
        freq_range = (min([r['fr_shifted'] for r in self.results])/1e9, 
                     max([r['fr_shifted'] for r in self.results])/1e9)
        z_range = (min([r['Zr'] for r in self.results]), 
                  max([r['Zr'] for r in self.results]))
        
        summary_text = f"""Average ε_eff: {avg_eps_eff:.2f}
Average α (kinetic inductance fraction): {avg_alpha:.3f}
Kinetic inductance per square (L□): {Lk_sq*1e12:.2f} pH/□ (material characteristic)
Frequency range: {freq_range[0]:.3f} - {freq_range[1]:.3f} GHz
Impedance range: {z_range[0]:.1f} - {z_range[1]:.1f} Ω"""
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
    
    def export_results(self):
        if not self.results:
            messagebox.showwarning("No Results", "No results to export. Please run calculation first.")
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                df = pd.DataFrame(self.results)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export Successful", f"Results exported to {filename}")
                
        except ImportError:
            messagebox.showerror("Export Error", "pandas is required for CSV export")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

def main():
    root = tk.Tk()
    app = CPWResonatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()