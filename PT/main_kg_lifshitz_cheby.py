"""
Main — Arquitectura Polinomial para QNMs de Klein-Gordon en Lifshitz 2D
═══════════════════════════════════════════════════════════════════════

Sin soluciones analíticas — criterio de convergencia basado en
estabilidad de ω y magnitud de la loss PDE.
"""

import os
import pickle
from datetime import datetime
import argparse
from matplotlib import rc
import numpy as np
import torch
import matplotlib.pyplot as plt

from obtengo_fqn_kg_lifshitz_cheby import (
    run_Scan_KG_Lifshitz,
    QNMs,
    DEVICE,
    plot_qnm_convergence,
    plot_loss,
    plot_wavefunction,
    generar_diagnostico,
)

N_RUNS = 20


def setup_fonts():
    try:
        rc('font', **{'family': 'serif',
                      'serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']})
        rc('text', usetex=False)
    except Exception as e:
        print(f"⚠ Advertencia fuentes: {e}")
        rc('font', **{'family': 'serif'})

    FONT_SIZE = 12
    plt.rc('font',   size=FONT_SIZE)
    plt.rc('axes',   titlesize=FONT_SIZE)
    plt.rc('axes',   labelsize=FONT_SIZE)
    plt.rc('legend', fontsize=FONT_SIZE - 1)
    plt.rc('xtick',  labelsize=FONT_SIZE - 1)
    plt.rc('ytick',  labelsize=FONT_SIZE - 1)


def guardar_resumen(all_results, run_dir, N_values, hyperparams_base, loss_threshold=1e-4):
    """Resumen con filtrado por calidad de loss."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  RESUMEN KG-LIFSHITZ POLINOMIAL — {N_RUNS} corridas")
    lines.append(f"  l={hyperparams_base['l']}, m={hyperparams_base['m']}, "
                 f"j={hyperparams_base['j']}")
    lines.append(f"  force_re_zero={hyperparams_base['force_re_zero']}")
    lines.append(f"  Umbral convergencia: loss_PDE < {loss_threshold:.0e}")
    lines.append("=" * 70)

    for N in N_values:
        datos = [r for r in all_results if r['N'] == N]
        if not datos:
            continue
        conv   = [r for r in datos if float(r.get('loss_pde_final', 1.0)) < loss_threshold]
        re_all = np.array([r['omega1'] for r in datos])
        im_all = np.array([r['omega2'] for r in datos])
        lines.append(f"\n-- Modo N={N} -------------------------------------------------")
        lines.append(f"  Epochs usados : {datos[0].get('epochs_used', '?')}")
        lines.append(f"  Convergidas   : {len(conv)}/{len(datos)}")
        lines.append(f"  Media(todas)  : {re_all.mean():.6f}+{im_all.mean():.6f}i  "
                     f"s=({re_all.std():.4f},{im_all.std():.4f})")
        if conv:
            rc_ = np.array([r['omega1'] for r in conv])
            ic_ = np.array([r['omega2'] for r in conv])
            lines.append(f"  Media(conv)   : {rc_.mean():.6f}+{ic_.mean():.6f}i  "
                         f"s=({rc_.std():.4f},{ic_.std():.4f})  <- USAR")
            if ic_.std() < 0.05:
                lines.append("  Consistencia  : BUENA")
            elif ic_.std() < 0.15:
                lines.append("  Consistencia  : ACEPTABLE")
            else:
                lines.append("  Consistencia  : REVISAR")
        datos_sorted = sorted(datos, key=lambda r: float(r.get('loss_pde_final', 1.0)), reverse=True)
        for r in datos_sorted:
            lv  = float(r.get('loss_pde_final', 1.0))
            tag = "v" if lv < loss_threshold else "x"
            lines.append(f"    run={r['run']:>2} {tag} -> {r['omega1']:.6f}+{r['omega2']:.6f}i "
                         f"| loss={lv:.2e} | {r['run_time']:.1f}s")

    lines.append("\n" + "=" * 70)
    txt = "\n".join(lines)

    print("\n" + txt)
    ruta = os.path.join(run_dir, "resumen.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(txt)


def get_epochs_for_mode(N, base_epochs):
    """
    Escala los epochs según el modo.
    Modos superiores tienen polinomios de mayor grado y paisajes más complejos.

    N=0 → 1.0×   N=1 → 1.5×   N=2 → 2.5×   N≥3 → 4.0×
    """
    scale = {0: 1.0, 1: 1.5, 2: 2.5, 3: 4.0}
    return int(base_epochs * scale.get(N, 4.0))


def main(use_amp=False):
    print("\n" + "=" * 70)
    print(" " * 10 + "BÚSQUEDA DE QNMs — ARQUITECTURA POLINOMIAL")
    print(" " * 10 + "KLEIN-GORDON EN AGUJERO LIFSHITZ 2D")
    print("=" * 70 + "\n")

    # ── Hiperparámetros ────────────────────────────────────────
    hyperparams_base = {
        'potential'     : 'KG-Lifshitz-2D',
        'architecture'  : 'Polynomial',
        'l'             : 1.0,       # radio AdS
        'm'             : 1.0,       # masa del campo
        'j'             : 1.0,       # exponente dinámico de Lifshitz
        'z0'            : 0.0,
        'zf'            : 1.0,
        'n_train'       : 200,
        'neurons'       : 0,
        'epochs'        : int(5e2),  # base — se escala por modo
        'learning_rate' : 5e-3,
        'minibatch'     : 1,
        'optimizer'     : 'Adam+LBFGS',
        'activation'    : 'N/A (polynomial)',
        'device'        : str(DEVICE),
        'amp'           : False,
        'losses'        : ['PDE', 'Anticollapse', 'Barriers',
                           'Boundary(U(1)=(-1)^N)'],
        'N_values'      : [5],
        'base_degree'   : 3,
        'force_re_zero' : True,   # Re(ω)=0 exacto — confirmado por N=0 y N=1
    }

    import math
    gamma_val = 1.5 * math.sqrt(1.0 + 4.0 / 9.0 * hyperparams_base['m']**2
                                 * hyperparams_base['l']**2)
    print("Configuración:")
    for k, v in hyperparams_base.items():
        print(f"  {k:<18}: {v}")
    print(f"  {'gamma':<18}: {gamma_val:.6f}")

    # Previsualizar epochs por modo
    print("\n  Epochs por modo (escalado):")
    for N in hyperparams_base['N_values']:
        ep_N = get_epochs_for_mode(N, hyperparams_base['epochs'])
        print(f"    N={N} → {ep_N} epochs  ({ep_N / hyperparams_base['epochs']:.1f}×)")

    # ── Directorios ───────────────────────────────────────────
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    historial_dir = os.path.join(base_dir, 'resultados_kg_lifshitz')
    os.makedirs(historial_dir, exist_ok=True)

    timestamp  = datetime.now().strftime("%d%m%y%H%M")
    N_values   = hyperparams_base['N_values']
    all_results = []
    total_time  = 0.0

    z0           = hyperparams_base['z0']
    zf           = hyperparams_base['zf']
    neurons      = hyperparams_base['neurons']
    base_epochs  = hyperparams_base['epochs']
    n_train      = hyperparams_base['n_train']
    lr           = hyperparams_base['learning_rate']
    minibatch    = hyperparams_base['minibatch']
    l_param      = hyperparams_base['l']
    m_param      = hyperparams_base['m']
    j_param      = hyperparams_base['j']
    base_degree  = hyperparams_base['base_degree']
    force_re_zero = hyperparams_base['force_re_zero']

    run_dir = os.path.join(historial_dir, f"kg_lif_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Bucle ─────────────────────────────────────────────────
    for run_i in range(N_RUNS):
        known_im_qnms = [-1.222,-2.1164,-3.1131,-4.1113,-5.1106]
        print(f"\n📋 Barreras iniciales: {known_im_qnms}")

        for N in N_values:
            #epochs_N = get_epochs_for_mode(N, base_epochs)
            epochs_N = base_epochs
            poly_deg = max(1, N + base_degree)
            re_tag   = '[Re=0]' if force_re_zero else '[Re libre]'

            print(f"\n{'#'*60}")
            print(f"  Corrida {run_i+1}/{N_RUNS}  |  Modo N={N}  {re_tag}  "
                  f"grado={poly_deg}  epochs={epochs_N}")
            print(f"  l={l_param}, m={m_param}, j={j_param}")
            print(f"{'#'*60}")

            path = os.path.join(run_dir, f"run_{run_i+1:02d}_N={N}")
            os.makedirs(path, exist_ok=True)

            hyperparams_run = hyperparams_base.copy()
            hyperparams_run.update({
                'N'          : N,
                'run'        : run_i + 1,
                'degree'     : N,
                'poly_deg'   : poly_deg,
                'epochs_used': epochs_N,
            })
            hyperparams_run.pop('N_values', None)

            model, loss_hists, run_time = run_Scan_KG_Lifshitz(
                z0, zf, neurons, epochs_N, n_train, lr, minibatch,
                use_amp=False, compile_model=False, log_interval=None,
                target_mode=N, known_im_qnms=known_im_qnms.copy(),
                l_param=l_param, m_param=m_param, j_param=j_param,
                base_degree=base_degree,
                force_re_zero=force_re_zero,
            )

            omega1, omega2 = QNMs(loss_hists, path, run_time, hyperparams_run)

            generar_diagnostico(
                loss_hists=loss_hists,
                path=path,
                runTime=run_time,
                hyperparams=hyperparams_run,
                target_mode=N,
                known_im_qnms=known_im_qnms.copy(),
            )

            print("\n📊 Generando gráficas...")
            plot_qnm_convergence(loss_hists, path, k_avg=500)
            plot_loss(loss_hists, path, logscale=True)
            plot_wavefunction(model, loss_hists, path, n_points=500, target_mode=N)
            print("✓ Gráficas generadas")

            loss_pde_final = float(np.array(loss_hists[1])[-1])

            pickle_path = os.path.join(path, "run_data.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'model'      : model,
                    'loss_hists' : loss_hists,
                    'run_time'   : run_time,
                    'omega1'     : omega1,
                    'omega2'     : omega2,
                    'hyperparams': hyperparams_run,
                }, f)

            all_results.append({
                'N'             : N,
                'run'           : run_i + 1,
                'omega1'        : omega1,
                'omega2'        : omega2,
                'run_time'      : run_time,
                'loss_pde_final': f"{loss_pde_final:.4e}",
                'epochs_used'   : epochs_N,
            })
            total_time += run_time

            # ── Actualizar barreras ──────────────────────────
            if loss_pde_final < 1e-4:
                known_im_qnms.append(omega2)
                print(f"\n  ✅ Modo N={N} aceptado (PDE={loss_pde_final:.2e}) "
                      f"→ Barreras: {known_im_qnms}")
            else:
                known_im_qnms.append(omega2)
                print(f"\n  ⚠️  Modo N={N} con PDE alta ({loss_pde_final:.2e}), "
                      f"añadido con cautela: {known_im_qnms}")

    results_path = os.path.join(run_dir, 'all_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    guardar_resumen(all_results, run_dir, N_values, hyperparams_base)

    print(f"\n⏱️  Tiempo total: {total_time/60:.2f} min")
    print(f"📁 Resultados en: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QNM Polinomial — KG Lifshitz 2D')
    parser.add_argument('--runs',    type=int,   default=None)
    parser.add_argument('--epochs',  type=int,   default=None,
                        help='Epochs BASE (se escala por modo)')
    parser.add_argument('--modes',   nargs='+',  type=int, default=None,
                        help='Modos a buscar, e.g. --modes 0 1 2')
    parser.add_argument('--l',       type=float, default=None, help='Radio AdS')
    parser.add_argument('--m',       type=float, default=None, help='Masa del campo')
    parser.add_argument('--j',       type=float, default=None, help='Exponente Lifshitz')
    parser.add_argument('--re-free', action='store_true',
                        help='Liberar Re(ω) (por defecto force_re_zero=True)')
    args = parser.parse_args()

    if args.runs is not None:
        N_RUNS = args.runs

    setup_fonts()
    main()