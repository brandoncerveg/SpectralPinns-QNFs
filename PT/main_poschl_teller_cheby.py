"""
Main — Arquitectura Chebyshev+HardMask para QNMs de Pöschl-Teller
═══════════════════════════════════════════════════════════════════

Modos analíticos (V₀=1):  ω_N = 0.5 − (N + 0.5)i
"""

import os
import pickle
from datetime import datetime
import argparse
from matplotlib import rc
import numpy as np
import torch
import matplotlib.pyplot as plt

from obtengo_fqn_poschl_teller_cheby import (
    run_Scan_Poschl_Teller,
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


def guardar_resumen(all_results, run_dir, N_values):
    analitico = {N: (0.5, -(N + 0.5)) for N in N_values}

    lines = []
    lines.append("=" * 65)
    lines.append(f"  RESUMEN CHEBYSHEV+HARDMASK — Pöschl-Teller — {N_RUNS} corridas")
    lines.append(f"  (ordenado de PEOR a MEJOR pérdida PDE)")
    lines.append("=" * 65)

    for N in N_values:
        datos = [r for r in all_results if r['N'] == N]
        if not datos:
            continue

        # Ordenar de mayor PDE loss (peor) a menor (mejor)
        datos_sorted = sorted(datos, key=lambda r: float(r.get('loss_pde_final', 0)),
                              reverse=True)

        re_arr = np.array([r['omega1'] for r in datos])
        im_arr = np.array([r['omega2'] for r in datos])
        re_an, im_an = analitico[N]

        lines.append(f"\n── Modo N={N} ─────────────────────────────────")
        lines.append(f"  Analítico     :  {re_an:.6f} + {im_an:.6f}i")
        lines.append(f"  Media PINN    :  {re_arr.mean():.6f} + {im_arr.mean():.6f}i")
        lines.append(f"  Std           :  {re_arr.std():.6f}   {im_arr.std():.6f}")
        lines.append(f"  Error medio Re:  {abs(re_arr.mean() - re_an):.6f}")
        lines.append(f"  Error medio Im:  {abs(im_arr.mean() - im_an):.6f}")

        # Mejor y peor
        best = min(datos, key=lambda r: float(r.get('loss_pde_final', float('inf'))))
        worst = max(datos, key=lambda r: float(r.get('loss_pde_final', 0)))
        lines.append(f"  Mejor PDE     :  {float(best['loss_pde_final']):.4e}  "
                     f"(run={best['run']}: ω={best['omega1']:.6f}{best['omega2']:+.6f}i)")
        lines.append(f"  Peor  PDE     :  {float(worst['loss_pde_final']):.4e}  "
                     f"(run={worst['run']}: ω={worst['omega1']:.6f}{worst['omega2']:+.6f}i)")
        lines.append(f"\n  {'run':>4}  {'Re(ω)':>10}  {'Im(ω)':>10}  "
                     f"{'err_Re':>9}  {'err_Im':>9}  {'PDE':>10}  {'min':>5}")

        for i, r in enumerate(datos_sorted):
            err_re = abs(r['omega1'] - re_an)
            err_im = abs(r['omega2'] - im_an)
            pde    = float(r.get('loss_pde_final', 0.0))
            t_min  = r['run_time'] / 60
            marker = " ← MEJOR" if r['run'] == best['run'] else ""
            lines.append(f"  {r['run']:>4}  {r['omega1']:>+10.6f}  {r['omega2']:>+10.6f}  "
                         f"{err_re:>9.6f}  {err_im:>9.6f}  {pde:>10.4e}  "
                         f"{t_min:>5.1f}{marker}")

    lines.append("\n" + "=" * 65)
    txt = "\n".join(lines)

    print("\n" + txt)
    ruta = os.path.join(run_dir, "resumen.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"📁 Resumen guardado: {ruta}")


def get_epochs_for_mode(N, base_epochs):
    scale = {0: 1.0, 1: 1.5, 2: 2.5, 3: 4.0}
    return int(base_epochs * scale.get(N, 4.0))

def construir_mensaje_whatsapp(all_results):
    # ordenar de MAYOR pérdida a MENOR
    results_sorted = sorted(
        all_results,
        key=lambda x: x['loss_pde_final'],
        reverse=True
    )

    msg = "📡 QNM Report (ordenado por PDE loss)\n\n"

    for r in results_sorted[:10]:  # top 10 (para no saturar)
        msg += (
            f"N={r['N']} | run={r['run']}\n"
            f"ω = {r['omega1']:.5f} + {r['omega2']:.5f}i\n"
            f"PDE = {r['loss_pde_final']:.2e}\n\n"
        )

    # mejor resultado
    best = min(all_results, key=lambda x: x['loss_pde_final'])
    msg += "🏆 BEST:\n"
    msg += (
        f"N={best['N']} | run={best['run']}\n"
        f"ω = {best['omega1']:.5f} + {best['omega2']:.5f}i\n"
        f"PDE = {best['loss_pde_final']:.2e}"
    )

    return msg

def main(use_amp=False):
    print("\n" + "="*70)
    print(" "*10 + "BÚSQUEDA DE QNMs — CHEBYSHEV + HARD MASK + HAdamD/L-BFGS")
    print(" "*15 + "POTENCIAL DE PÖSCHL-TELLER")
    print("="*70 + "\n")

    # ── Hiperparámetros ────────────────────────────────────────
    hyperparams_base = {
        'potential'     : 'Pöschl-Teller',
        'architecture'  : 'Chebyshev+HardMask',
        'V0'            : 1.0,
        'a'             : 1.0,
        'y0'            : 0.0,
        'yf'            : 1.0,
        'n_train'       : 50000,
        'neurons'       : 0,
        'epochs'        : int(5e2),
        'learning_rate' : 5e-3,
        'minibatch'     : 1,
        'optimizer'     : 'HAdamD + L-BFGS',
        'activation'    : 'Chebyshev T_k',
        'device'        : str(DEVICE),
        'amp'           : False,
        'losses'        : ['PDE(MAE)', 'Anticollapse', 'Barriers',
                           'HardMask(phi(0)=1)', 'Boundary(phi(1)=(-1)^N)'],
        'grid'          : 'Chebyshev-Gauss-Lobatto',
        'N_values'      : [2],
        'base_degree'   : 0
    }

    print("Configuración:")
    for k, v in hyperparams_base.items():
        print(f"  {k:<18}: {v}")

    print("\n  Epochs por modo (escalado):")
    for N in hyperparams_base['N_values']:
        ep_N = get_epochs_for_mode(N, hyperparams_base['epochs'])
        print(f"    N={N} → {ep_N} epochs  ({ep_N / hyperparams_base['epochs']:.1f}×)")

    # ── Directorios ───────────────────────────────────────────
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    historial_dir = os.path.join(base_dir, 'resultados_pt_cheby')
    os.makedirs(historial_dir, exist_ok=True)

    timestamp   = datetime.now().strftime("%d%m%y%H%M")
    N_values    = hyperparams_base['N_values']
    all_results = []
    total_time  = 0.0

    y0          = hyperparams_base['y0']
    yf          = hyperparams_base['yf']
    neurons     = hyperparams_base['neurons']
    base_epochs = hyperparams_base['epochs']
    n_train     = hyperparams_base['n_train']
    lr          = hyperparams_base['learning_rate']
    minibatch   = hyperparams_base['minibatch']
    base_degree = hyperparams_base['base_degree']

    run_dir = os.path.join(historial_dir, f"pt_cheby_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Bucle ─────────────────────────────────────────────────
    for run_i in range(N_RUNS):
        known_im_qnms = [-0.5,-1.5]
        print(f"\n📋 Barreras iniciales: {known_im_qnms}")

        for N in N_values:
            epochs_N = base_epochs 
            poly_deg = max(1, N + base_degree)

            print(f"\n{'#'*60}")
            print(f"  Corrida {run_i+1}/{N_RUNS}  |  Modo N={N}  |  grado={poly_deg}  "
                  f"epochs={epochs_N}")
            print(f"  Analítico: ω = 0.5 − {N + 0.5:.1f}i")
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

            model, loss_hists, run_time = run_Scan_Poschl_Teller(
                y0, yf, neurons, epochs_N, n_train, lr, minibatch,
                use_amp=False, compile_model=False, log_interval=None,
                target_mode=N,
                known_im_qnms=known_im_qnms.copy(),
                base_degree=base_degree,
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
                'loss_pde_final': loss_pde_final,
            })
            total_time += run_time

            # Actualizar barreras
            im_analitico = -(N + 0.5)
            if abs(omega2 - im_analitico) < 0.2:
                known_im_qnms.append(omega2)
                print(f"\n  ✅ Modo N={N} encontrado → Barreras: {known_im_qnms}")
            else:
                known_im_qnms.append(im_analitico)
                print(f"\n  ⚠️  Usando barrera analítica para N={N}: {im_analitico}")

    results_path = os.path.join(run_dir, 'all_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    guardar_resumen(all_results, run_dir, N_values)

    print(f"\n⏱️  Tiempo total: {total_time/60:.2f} min")
    print(f"📁 Resultados en: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='QNM Chebyshev+HardMask — Pöschl-Teller')
    parser.add_argument('--runs',   type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--modes',  nargs='+', type=int, default=None,
                        help='Modos a buscar, e.g. --modes 0 1 2')
    args = parser.parse_args()

    if args.runs is not None:
        N_RUNS = args.runs

    setup_fonts()
    main()
