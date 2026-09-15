"""
Main — Arquitectura Polinomial para QNMs de Schwarzschild
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

from obtengo_fqn_schwarzschild_cheb import (
    run_Scan_Schwarzschild,
    QNMs,
    DEVICE,
    plot_qnm_convergence,
    plot_loss,
    plot_wavefunction,
    generar_diagnostico,
)

N_RUNS = 10


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
    lines.append(f"  RESUMEN Schwarzschild POLINOMIAL — {N_RUNS} corridas")
    lines.append(f"  L={hyperparams_base['L']}, M={hyperparams_base['M']}, "
                 f"S={hyperparams_base['S']}")
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


def main(use_amp=False, cli_overrides=None):
    print("\n" + "=" * 70)
    print(" " * 10 + "BÚSQUEDA DE QNMs — ARQUITECTURA POLINOMIAL")
    print(" " * 10 + "Schwarzschild")
    print("=" * 70 + "\n")

    # ── Hiperparámetros ────────────────────────────────────────
    hyperparams_base = {
        'potential'     : 'Schwarzschild',
        'architecture'  : 'Polynomial',
        # L : r_
        'L'             : 8,
        # M : r+
        'M'             : 10,
        # S : m
        'S'             : 3/4,       # gravitacional — cambia a 0 o 1 si corresponde
        # ── Dominio ────────────────────────────────────────────────────────
        'z0'            : 0.0,
        'zf'            : 1.0,
        # ── Entrenamiento ──────────────────────────────────────────────────
        'n_train'       : 200,
        'neurons'       : 0,
        'epochs'        : int(10e2),  # base — se escala por modo con get_epochs_for_mode
        'learning_rate' : 5e-3,
        'minibatch'     : 1,
        'optimizer'     : 'Adam+LBFGS',
        'activation'    : 'N/A (polynomial)',
        'device'        : str(DEVICE),
        'amp'           : False,
        'losses'        : ['PDE', 'Anticollapse', 'Barriers'],
        'N_values'      : [1],
        # base_degree=4 da n = max(1, N+4) coeficientes Chebyshev por modo
        # (con 0 sólo hay 2 coeficientes libres para N=0 — muy poco)
        'base_degree'   : 5,
        # force_re_zero=False: Re(ω) es libre
        # Para QNMs de Schwarzschild estándar Re(ω) ≠ 0 (excepto modos algebraicamente especiales)
        # Pon True sólo si tienes evidencia de que Re(ω)=0 para tu modo específico
        'force_re_zero' : True,
    }

    # ── Aplicar overrides de CLI ───────────────────────────────
    if cli_overrides:
        hyperparams_base.update(cli_overrides)
        print("  Overrides CLI aplicados:")
        for k, v in cli_overrides.items():
            print(f"    {k} = {v}")



    # Previsualizar epochs por modo
    print("\n  Epochs por modo (escalado):")
    for N in hyperparams_base['N_values']:
        ep_N = get_epochs_for_mode(N, hyperparams_base['epochs'])
        print(f"    N={N} → {ep_N} epochs  ({ep_N / hyperparams_base['epochs']:.1f}×)")

    # ── Directorios ───────────────────────────────────────────
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    historial_dir = os.path.join(base_dir, 'resultados_schwarzschild')
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
    L_param      = hyperparams_base['L']
    M_param      = hyperparams_base['M']
    S_param      = hyperparams_base['S']
    base_degree  = hyperparams_base['base_degree']
    force_re_zero = hyperparams_base['force_re_zero']

    run_dir = os.path.join(historial_dir, f"schw_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Bucle ─────────────────────────────────────────────────
    for run_i in range(N_RUNS):
        known_im_qnms = [-4.2980]
        print(f"\n📋 Barreras iniciales: {known_im_qnms}")

        for N in N_values:
            epochs_N = base_epochs
            poly_deg = max(1, N + base_degree)
            re_tag   = '[Re=0]' if force_re_zero else '[Re libre]'

            print(f"\n{'#'*60}")
            print(f"  Corrida {run_i+1}/{N_RUNS}  |  Modo N={N}  {re_tag}  "
                  f"grado={poly_deg}  epochs={epochs_N}")
            print(f"  L={L_param}, M={M_param}, S={S_param}")
            print(f"{'#'*60}")

            path = os.path.join(run_dir, f"run_{run_i+1:02d}_N={N}")
            os.makedirs(path, exist_ok=True)

            hyperparams_run = hyperparams_base.copy()
            hyperparams_run.update({
                'N'          : N,
                'run'        : run_i + 1,
                'degree'     : max(1, N + base_degree),  # grado real del polinomio
                'poly_deg'   : poly_deg,
                'epochs_used': epochs_N,
            })
            hyperparams_run.pop('N_values', None)

            model, loss_hists, run_time = run_Scan_Schwarzschild(
                z0, zf, neurons, epochs_N, n_train, lr, minibatch,
                use_amp=False, compile_model=False, log_interval=None,
                target_mode=N, known_im_qnms=known_im_qnms.copy(),
                L_param=L_param, M_param=M_param, S_param=S_param,
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

    print(f"\n⏱️  Tiempo total: {total_time:.2f} s")
    print(f"📁 Resultados en: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QNM Polinomial — KG Lifshitz 2D')
    parser.add_argument('--runs',    type=int,   default=None)
    parser.add_argument('--epochs',  type=int,   default=None,
                        help='Epochs BASE (se escala por modo)')
    parser.add_argument('--modes',   nargs='+',  type=int, default=None,
                        help='Modos a buscar, e.g. --modes 0 1 2')
    parser.add_argument('--l',       type=float, default=None, help='Número cuántico angular multipolar (L)')
    parser.add_argument('--m',       type=float, default=None, help='Masa del agujero negro (M)')
    parser.add_argument('--j',       type=float, default=None, help='Spin del campo: 0=escalar, 1=EM, 2=grav')
    parser.add_argument('--re-free', action='store_true',
                        help='Fuerza Re(ω) libre (por defecto ya es False; flag reservado para compatibilidad)')
    args = parser.parse_args()

    if args.runs is not None:
        N_RUNS = args.runs

    # ── Aplicar argumentos CLI a hyperparams_base ─────────────────────────
    # (deben modificarse ANTES de llamar a main(), que lee hyperparams_base)
    # Se importan aquí con un pequeño truco: main() los leerá desde el módulo.
    # En su lugar, los pasamos explícitamente como overrides dentro de main().
    _cli_overrides = {}
    if args.epochs is not None: _cli_overrides['epochs']        = args.epochs
    if args.modes  is not None: _cli_overrides['N_values']      = args.modes
    if args.l      is not None: _cli_overrides['L']             = args.l
    if args.m      is not None: _cli_overrides['M']             = args.m
    if args.j      is not None: _cli_overrides['S']             = args.j
    if args.re_free:             _cli_overrides['force_re_zero'] = False

    setup_fonts()
    main(cli_overrides=_cli_overrides)