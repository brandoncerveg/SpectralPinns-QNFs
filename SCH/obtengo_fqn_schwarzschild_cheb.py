import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import os

def get_device():
    """Forzado a CPU según configuración del proyecto."""
    device = torch.device("cpu")
    print("✓ Dispositivo: CPU")
    return device

DEVICE = get_device()
dtype = torch.float64
torch.set_default_dtype(dtype)


class HAdamD(optim.Optimizer):
    """
    HybridAdamDecoupled — Adam para parámetros complejos.

    Para parámetros complejos mantiene momentos desacoplados
    por eje real e imaginario:
        Δu_R = Re(m̂_t) / √(v̂_t^R + ε)
        Δu_I = Im(m̂_t) / √(v̂_t^I + ε)

    Para parámetros reales se reduce a Adam estándar.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']; beta1, beta2 = group['betas']; eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad  = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    if p.is_complex():
                        state['m']    = torch.zeros_like(p.data)
                        state['v_re'] = torch.zeros(p.shape, dtype=p.real.dtype, device=p.device)
                        state['v_im'] = torch.zeros(p.shape, dtype=p.real.dtype, device=p.device)
                    else:
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']

                if p.is_complex():
                    m, v_re, v_im = state['m'], state['v_re'], state['v_im']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v_re.mul_(beta2).add_(grad.real * grad.real, alpha=1 - beta2)
                    v_im.mul_(beta2).add_(grad.imag * grad.imag, alpha=1 - beta2)
                    bc1 = 1.0 - beta1 ** t; bc2 = 1.0 - beta2 ** t
                    m_hat = m / bc1
                    delta_re = m_hat.real / (v_re.div(bc2).sqrt_() + eps)
                    delta_im = m_hat.imag / (v_im.div(bc2).sqrt_() + eps)
                    p.data.add_(torch.complex(delta_re, delta_im), alpha=-lr)
                else:
                    m, v = state['m'], state['v']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bc1 = 1.0 - beta1 ** t; bc2 = 1.0 - beta2 ** t
                    p.data.addcdiv_(m / bc1, (v / bc2).sqrt_() + eps, value=-lr)

        return loss
    

class PolynomialNet(nn.Module):
    """
    Parametrización Chebyshev con condición fuerte U(0)=1.

        U(z) = 1 + M(z) · N(z)

    donde:
        M(z) = eᶻ − 1                     (máscara: M(0)=0 → U(0)=1 exacto; U(1) libre)
        N(z) = Σₖ₌₀ⁿ Aₖ Tₖ(2z−1)          (base Chebyshev, x = 2z−1 ∈ [−1,1])

    Derivadas vía regla del producto:
        M' = M'' = eᶻ
        U'  = eᶻ·N  + (eᶻ−1)·N'
        U'' = eᶻ·N  + 2eᶻ·N' + (eᶻ−1)·N''

    Aₖ complejos entrenables (complex128).
    Grado total: n = max(1, N + base_degree).
    Parámetros: (n+1) complejos + ω.
    """

    def __init__(self, degree, omega_init=(0.3, -0.5),
                 base_degree=4, force_re_zero=False):
        super().__init__()
        self.degree        = degree
        self.force_re_zero = force_re_zero
        self.n             = max(1, degree + base_degree)

        # A₀, A₁, ..., Aₙ  — todos entrenables (n+1 coeficientes)
        init_re = torch.randn(self.n + 1, dtype=dtype) * 0.05
        init_im = torch.randn(self.n + 1, dtype=dtype) * 0.05
        self.coeffs = nn.Parameter(torch.complex(init_re, init_im))

        if force_re_zero:
            self.register_buffer('omega_re', torch.tensor(0.0, dtype=dtype))
        else:
            self.omega_re = nn.Parameter(torch.tensor(omega_init[0], dtype=dtype))

        self.omega_im = nn.Parameter(torch.tensor(omega_init[1], dtype=dtype))

    def forward(self, z):
        """
        Evalua U(z), U'(z), U''(z) con base Chebyshev + mascara exponencial.

        Args:
            z: (batch, 1) tensor real, z ∈ [0, 1]

        Returns:
            U    : (batch,) complejo
            U_d  : (batch,) complejo — dU/dz
            U_dd : (batch,) complejo — d²U/dz²
            omega: escalar complejo
        """
        z_flat = z.squeeze(-1).to(torch.complex128)
        B, dev, n = z_flat.shape[0], z.device, self.n

        # ── Mapear z ∈ [0,1] → x ∈ [−1,1] ─────────────────
        x = 2.0 * z_flat - 1.0

        # ── Chebyshev T_k(x), T'_k(x), T''_k(x) por recurrencia ──
        T   = torch.zeros(B, n + 1, dtype=torch.complex128, device=dev)
        Td  = torch.zeros(B, n + 1, dtype=torch.complex128, device=dev)
        Tdd = torch.zeros(B, n + 1, dtype=torch.complex128, device=dev)

        T[:, 0] = 1.0
        if n >= 1:
            T[:, 1]  = x
            Td[:, 1] = 1.0
        for k in range(2, n + 1):
            T[:, k]   = 2.0 * x * T[:, k-1]  - T[:, k-2]
            Td[:, k]  = 2.0 * T[:, k-1] + 2.0 * x * Td[:, k-1]  - Td[:, k-2]
            Tdd[:, k] = 4.0 * Td[:, k-1] + 2.0 * x * Tdd[:, k-1] - Tdd[:, k-2]

        # N(z)   = Σ Aₖ Tₖ(x)
        # N'(z)  = (dx/dz)·Σ Aₖ T'ₖ(x) = 2·Σ Aₖ T'ₖ(x)
        # N''(z) = (dx/dz)²·Σ Aₖ T''ₖ(x) = 4·Σ Aₖ T''ₖ(x)
        N_val = T   @ self.coeffs
        N_d   = 2.0 * (Td  @ self.coeffs)
        N_dd  = 4.0 * (Tdd @ self.coeffs)

        # ── Máscara M(z) = eᶻ − 1 ───────────────────────────
        # M(0) = 0  →  U(0) = 1  ✓      U(1) libre
        ez   = torch.exp(z_flat)
        M    = ez - 1.0
        M_d  = ez
        M_dd = ez

        # ── U(z) = 1 + M·N ──────────────────────────────────
        U    = 1.0 + M    * N_val
        U_d  =       M_d  * N_val + M    * N_d
        U_dd =       M_dd * N_val + 2.0  * M_d * N_d + M * N_dd

        omega = torch.complex(self.omega_re, self.omega_im)
        return U, U_d, U_dd, omega

    def get_coeffs_complex(self):
        """Devuelve [A₀, A₁, ..., Aₙ] como array complejo numpy."""
        with torch.no_grad():
            return self.coeffs.detach().cpu().numpy().copy()
        

def PDE_residual_Schwarzschild(z, U, U_d, U_dd, omega, L=1.0, M=1.0, S=1.0):
    """
    Residuo de la EDP de perturbación lineal en Schwarzschild
    (forma de Leaver compactificada, ξ ∈ (0,1)):

        ξ(1-ξ)² U'' - Λ₀ U' - S₀ U = 0

    con:
        Λ₀ = 4Miω(2ξ² - 4ξ + 1) - (1 - 3ξ)(1 - ξ)
        S₀ = 16M²ω²(ξ - 2) - 8Miω(1 - ξ) + L(L+1) + (1 - S²)(1 - ξ)

    Parámetros:
        S : spin del campo  (0 = escalar, 1 = EM, 2 = gravitacional)
              ⚠ NO es el exponente de Lifshitz — ese parámetro fue renombrado aquí
        M : masa del agujero negro
        L : número cuántico angular multipolar (L=2 cuadrupolar, etc.)
              ⚠ NO es el radio AdS — ese nombre es un remanente del código Lifshitz

    Devuelve |residuo|² promediado.
    """
    z_c = z.squeeze(-1).to(torch.complex128)

    # Coeficientes de la EDP — todos complejos por contener ω
    A = z_c * (1.0 - z_c) *(1.0 - z_c)

    Lambda_0 = (4.0 * M * 1j * omega * (2.0 * z_c ** 2 - 4.0 * z_c + 1.0)
                - (1.0 - 3.0 * z_c) * (1.0 - z_c))

    S_0 = (16.0 * (M ** 2) * (omega ** 2) * (z_c - 2.0)
           - 8.0 * M * 1j * omega * (1.0 - z_c)
           + L * (L + 1.0)
           + (1.0 - S ** 2) * (1.0 - z_c))

    residual = A * U_dd - Lambda_0 * U_d - S_0 * U
    return (residual.real**2 + residual.imag**2).mean()


# ═══════════════════════════════════════════════════════════════
# PÉRDIDAS AUXILIARES
# ═══════════════════════════════════════════════════════════════

def L_anticollapse(model, sigma=0.5):
    """Penaliza |Aₙ| → 0 para evitar colapso a modo inferior."""
    cn     = model.coeffs[-1]                  # len(coeffs) = n+1 con Chebyshev
    cn_mag = cn.real**2 + cn.imag**2
    return torch.exp(-cn_mag / sigma**2)


def L_c(omega2, c2, a=20.0):
    """Barrera sigmoide superior: penaliza si omega2 > c2."""
    return -torch.sigmoid(a * (c2 - omega2)) + 1


def L_c_lower(omega2, lower_bound, a=20.0):
    """Barrera sigmoide inferior: penaliza si omega2 < lower_bound."""
    return torch.sigmoid(a * (lower_bound - omega2))


def L_re_floor(omega1, floor_val=0.05, a=30.0):
    """Protección: penaliza Re(ω) < floor_val. Solo usar si force_re_zero=False."""
    return torch.sigmoid(a * (floor_val - omega1))


def estimate_target_im(known_im_qnms, safety_factor=1.5,
                       min_window=0.5, require_n_modes=1):
    """
    Estima Im(ω) del próximo modo y devuelve barrera inferior.

    Parámetros
    ----------
    known_im_qnms   : lista de Im(ω) conocidos (ordenados o no)
    safety_factor   : amplía la ventana inferior por este factor (>1 = más holgura)
    min_window      : ventana mínima absoluta bajo el target estimado
    require_n_modes : mínimo de modos para activar barrera inferior;
                      con menos datos el espaciado es pura suposición

    Devuelve
    --------
    estimated_target : float o None
    lower_bound      : float o None  (None = no poner barrera inferior)
    spacing_used     : float o None  (para diagnóstico)
    """
    if not known_im_qnms:
        return None, None, None

    sorted_modes = sorted(known_im_qnms, reverse=True)  # más cercano a 0 primero

    if len(sorted_modes) >= 2:
        spacings = [sorted_modes[i] - sorted_modes[i + 1]
                    for i in range(len(sorted_modes) - 1)]
        # Promedio ponderado: más peso a los espaciados más recientes
        weights = np.arange(1, len(spacings) + 1, dtype=float)
        avg_spacing = float(np.average(spacings, weights=weights))
    else:
        avg_spacing = 0.2  # fallback conservador con un solo modo conocido

    estimated_target = sorted_modes[-1] - avg_spacing

    # Barrera inferior: solo si tenemos suficientes modos conocidos
    if len(sorted_modes) < require_n_modes:
        lower_bound = None   # sin barrera — demasiado poco historial
    else:
        window = max(min_window, abs(avg_spacing) * safety_factor)
        lower_bound = estimated_target - window

    return estimated_target, lower_bound, avg_spacing


def chebyshev_points(n_points, z0=1e-4, zf=1.0-1e-4, device=None):
    """Nodos de Chebyshev en (z0, zf)."""
    k = torch.arange(n_points, dtype=dtype, device=device)
    nodes = torch.cos((2*k + 1) / (2*n_points) * math.pi)
    z = z0 + (zf - z0) * (nodes + 1.0) / 2.0
    return z.reshape(-1, 1)


def run_Scan_Schwarzschild(z0, zf, neurons, epochs, n_train, lr, minibatch_number,
                          use_amp=False, compile_model=False, log_interval=None,
                          target_mode=0, known_im_qnms=None,
                          L_param=1.0, M_param=1.0, S_param=1.0,
                          base_degree=4, force_re_zero=False):
    """
    Entrenamiento con arquitectura polinomial para QNMs de Schwarzschild.

    Adam (30%) + L-BFGS (70%). Scheduling continuo por fracción de época.

    Parámetros físicos:
        L_param       : número cuántico angular multipolar
        M_param       : masa del agujero negro
        S_param       : spin del campo (0=escalar, 1=EM, 2=gravitacional)
        force_re_zero : si True, Re(ω)=0 exacto. Para Schwarzschild usar False.
    """
    _n = max(1, target_mode + base_degree)
    re_tag = '[Re=0 forzado]' if force_re_zero else '[Re libre]'
    print(f"\n{'='*60}")
    print(f"  Modo N={target_mode}  Chebyshev+HardMask  grado={_n}  (N={target_mode}+base={base_degree})  {re_tag}")
    print(f"  params={_n+1} complejos + ω  "
          f"L={L_param} M={M_param} S={S_param}")
    print(f"{'='*60}")

    # ── Inicialización de ω ─────────────────────────────────────
    if known_im_qnms and len(known_im_qnms) > 0:
        estimated_target, _, _ = estimate_target_im(known_im_qnms)
        omega_init = (0.0 if force_re_zero else 0.5,
                      estimated_target if estimated_target is not None else -0.5)
    else:
        omega_init = (0.0 if force_re_zero else 0.5, -0.5)

    # Perturbación: más estrecha cuando tenemos buenos modos conocidos
    if known_im_qnms and len(known_im_qnms) >= 2:
        noise_im = np.random.uniform(-0.15, 0.15)
    else:
        noise_im = np.random.uniform(-0.3, 0.3)

    noise_re = 0.0 if force_re_zero else np.random.uniform(-0.2, 0.2)
    omega_init = (omega_init[0] + noise_re, omega_init[1] + noise_im)

    print(f"  ω init: {omega_init[0]:.4f} + {omega_init[1]:.4f}i")

    model = PolynomialNet(
        degree=target_mode,
        omega_init=omega_init,
        base_degree=base_degree,
        force_re_zero=force_re_zero,
    ).to(DEVICE)
    model.train()  # activar modo entrenamiento explícitamente

    for name, param in model.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

    # ── Puntos de colocación ────────────────────────────────────
    z_train = chebyshev_points(n_train, z0=1e-4, zf=1.0-1e-4, device=DEVICE)
    print(f"  Puntos de colocación: {n_train} (Chebyshev)")

    # ── Optimizador ─────────────────────────────────────────────
    phase1_frac   = 0.20
    phase1_epochs = int(epochs * phase1_frac)

    optimizer_adam = HAdamD(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=phase1_epochs)

    # ── Pesos de pérdidas ───────────────────────────────────────
    w_pde          = 10.0
    w_anticollapse = 3.0  if target_mode >= 1 else 0.0
    w_Lc           = 10.0  if target_mode >= 1 else 0.0
    w_Lc_low       = 0.0
    # re_floor solo activo si Re es libre y no estamos forzando a cero
    w_re_floor     = (0.3 if target_mode >= 1 else 0.0) if not force_re_zero else 0.0

    # ── Barreras ────────────────────────────────────────────────
    barriers_upper    = []
    lower_bound_tensor = None

    if known_im_qnms and target_mode >= 1:
        for im_known in known_im_qnms:
            barriers_upper.append(
                torch.tensor(im_known, dtype=dtype, device=DEVICE)
            )
        # tensor apilado para vectorizar la suma de barreras
        barriers_upper_tensor = torch.stack(barriers_upper) if barriers_upper else None
        estimated_target_val, lower_bound_val, spacing = estimate_target_im(
            known_im_qnms,
            safety_factor=1.5,
            min_window=0.5,
            require_n_modes=1,
        )
        if lower_bound_val is not None:
            lower_bound_tensor = torch.tensor(lower_bound_val, dtype=dtype, device=DEVICE)
            w_Lc_low = 1.0
            print(f"  Target Im(ω) : {estimated_target_val:.4f}")
            print(f"  Espaciado    : {spacing:.4f}")
            print(f"  Barrera inf  : {lower_bound_val:.4f}  "
                  f"(window={abs(estimated_target_val - lower_bound_val):.4f})")
        else:
            print(f"  Barrera inf  : DESACTIVADA (modos conocidos insuficientes para "
                  f"require_n_modes=2, tenemos {len(known_im_qnms)})")
    else:
        barriers_upper_tensor = None

    print(f"  Barreras superiores: {[b.item() for b in barriers_upper]}")
    print(f"  Pesos — PDE:{w_pde} anticollapse:{w_anticollapse} "
          f"Lc:{w_Lc} Lc_low:{w_Lc_low} re_floor:{w_re_floor}")

    # ── Historiales ─────────────────────────────────────────────
    Loss_history         = []
    Loss_PDE_history     = []
    Loss_nontriv_history = []
    Loss_excite_history  = []
    Loss_Lc_history      = []
    omega1_history       = []
    omega2_history       = []

    if log_interval is None:
        log_interval = max(epochs // 10, 1)

    t0 = time.time()

    # ── Pesos iniciales para scheduling ─────────────────────────
    w_Lc_0           = w_Lc
    w_Lc_low_0        = w_Lc_low
    w_re_floor_0      = w_re_floor
    w_anticollapse_0  = w_anticollapse

    # ── Scheduling continuo por fracción de época ────────────────
    decay_start    = 0.40
    decay_rate     = 6.0
    Lc_floor       = 0.03
    pde_only_start = 0.70

    if target_mode >= 1:
        print(f"\n  📐 Scheduling continuo por fracción de época:")
        print(f"     Fase guía       : 0% → {decay_start*100:.0f}%  (pesos máximos)")
        print(f"     Fase transición : {decay_start*100:.0f}% → {pde_only_start*100:.0f}%  "
              f"(decay exponencial, Lc_floor={Lc_floor})")
        print(f"     Fase PDE pura   : {pde_only_start*100:.0f}% → 100%  (solo PDE)")
    else:
        print(f"\n  📐 Modo N=0: sin pérdidas auxiliares, scheduling no aplica")

    # ════════════════════════════════════════════════════════════
    #  BUCLE DE ENTRENAMIENTO
    # ════════════════════════════════════════════════════════════

    last_loss_pde = None
    optimizer_lbfgs = None

    for tt in tqdm(range(epochs), desc=f"Schwarzschild N={target_mode}", mininterval=0.5):

        frac      = tt / epochs
        use_lbfgs = (tt >= phase1_epochs)

        # ── Scheduling continuo ─────────────────────────────────
        if target_mode >= 1:
            if frac >= pde_only_start:
                w_anticollapse = 0.0
                w_Lc           = 0.0
                w_Lc_low       = 0.0
                w_re_floor     = 0.0
            elif frac > decay_start:
                progress = (frac - decay_start) / (pde_only_start - decay_start)
                w_anticollapse = w_anticollapse_0 * math.exp(-decay_rate * progress)
                decay_factor   = math.exp(-decay_rate * 0.5 * progress)
                w_Lc           = max(Lc_floor * w_Lc_0,     w_Lc_0     * decay_factor)
                w_Lc_low       = max(Lc_floor * w_Lc_low_0, w_Lc_low_0 * decay_factor)
                if frac > 0.60:
                    re_progress = (frac - 0.60) / (pde_only_start - 0.60)
                    w_re_floor  = w_re_floor_0 * max(0.1, 1.0 - 0.9 * re_progress)
            # else: fase guía — pesos al máximo

        # ── Forward + Loss ──────────────────────────────────────
        def closure():
            if use_lbfgs:
                optimizer_lbfgs.zero_grad()
            else:
                optimizer_adam.zero_grad(set_to_none=True)

            U, U_d, U_dd, omega = model(z_train)
            omega1 = model.omega_re
            omega2 = model.omega_im

            loss_pde = PDE_residual_Schwarzschild(
                z_train, U, U_d, U_dd, omega,
                L=L_param, M=M_param, S=S_param
            )

            loss_ac = L_anticollapse(model) if target_mode >= 1 else torch.tensor(0.0)

            # Barreras superiores — vectorizadas
            if barriers_upper_tensor is not None:
                loss_Lc_upper = (-torch.sigmoid(20.0 * (barriers_upper_tensor - omega2)) + 1.0).sum()
            else:
                loss_Lc_upper = torch.tensor(0.0, dtype=dtype, device=DEVICE)

            loss_Lc_low = (
                L_c_lower(omega2, lower_bound_tensor, 20.0)
                if lower_bound_tensor is not None
                else torch.tensor(0.0, dtype=dtype, device=DEVICE)
            )

            # re_floor: solo relevante si Re(ω) es parámetro libre
            if not force_re_zero and target_mode >= 1:
                loss_re = L_re_floor(omega1, floor_val=0.05, a=30.0)
            else:
                loss_re = torch.tensor(0.0, dtype=dtype, device=DEVICE)

            total = (w_pde          * loss_pde
                     + w_anticollapse * loss_ac
                     + w_Lc          * loss_Lc_upper
                     + w_Lc_low      * loss_Lc_low
                     + w_re_floor    * loss_re)

            total.backward()

            closure.loss_pde = loss_pde.item()
            closure.loss_ac  = loss_ac.item()
            closure.loss_Lc  = (loss_Lc_upper + loss_Lc_low).item()
            closure.total    = total.item()
            closure.omega1   = omega1.item()
            closure.omega2   = omega2.item()

            return total

        # ── Paso del optimizador ────────────────────────────────
        if use_lbfgs:
            if tt == phase1_epochs:
                optimizer_lbfgs = optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=20,
                    history_size=50,
                    line_search_fn='strong_wolfe'
                )
                print(f"\n  ⚡ Cambiando a L-BFGS en epoch {tt}")

            if optimizer_lbfgs is None:
                raise RuntimeError(
                    f"optimizer_lbfgs no inicializado en epoch {tt}. "
                    f"phase1_epochs={phase1_epochs}, use_lbfgs={use_lbfgs}."
                )
            optimizer_lbfgs.step(closure)
        else:
            closure()
            optimizer_adam.step()
            scheduler.step()

        last_loss_pde = closure.loss_pde

        # ── Registrar cada 10 epochs ────────────────────────────
        if tt % 10 == 0 or tt == epochs - 1:
            Loss_history.append(closure.total)
            Loss_PDE_history.append(closure.loss_pde)
            Loss_nontriv_history.append(0.0)  # vestigial — reservado para uso futuro
            Loss_excite_history.append(closure.loss_ac)
            Loss_Lc_history.append(closure.loss_Lc)
            omega1_history.append(closure.omega1)
            omega2_history.append(closure.omega2)

        # ── Log ─────────────────────────────────────────────────
        if (tt + 1) % log_interval == 0:
            elapsed    = time.time() - t0
            pct        = (tt + 1) / epochs * 100
            coeffs     = model.get_coeffs_complex()
            coeffs_str = "  ".join([f"c{k}={c:.4f}" for k, c in enumerate(coeffs)])

            print(f"\n{'─'*60}")
            print(f"  Epoch {tt+1:>6}/{epochs}  ({pct:.0f}%)  {elapsed:.1f} s")
            print(f"{'─'*60}")
            print(f"  Loss PDE     : {closure.loss_pde:.4e}")
            print(f"  Loss total   : {closure.total:.4e}")
            print(f"  ω            : {closure.omega1:.6f} + {closure.omega2:.6f}i  {re_tag}")
            print(f"  Coeficientes : {coeffs_str}")
            if target_mode >= 1:
                cn_mag = abs(coeffs[-1])
                print(f"  |cₙ|         : {cn_mag:.6f}  (anti-colapso: {closure.loss_ac:.4e})")
            print(f"  Optimizador  : {'L-BFGS' if use_lbfgs else 'Adam'}")
            if target_mode >= 1:
                if frac >= pde_only_start:
                    sched_str = "PDE pura"
                elif frac > decay_start:
                    sched_str = "transición"
                else:
                    sched_str = "guía"
                print(f"  Scheduling   : {sched_str}  "
                      f"ac={w_anticollapse:.3f}  Lc={w_Lc:.3f}  re={w_re_floor:.3f}")
            print(f"{'─'*60}")

    # ── Fin ─────────────────────────────────────────────────────
    t1      = time.time()
    runTime = t1 - t0

    model_cpu = model.cpu()

    # El historial se registra cada 10 epochs — se mantiene sparse (más limpio)
    model_array = [[model_cpu]]

    loss_hists = [Loss_history, Loss_PDE_history, Loss_nontriv_history,
                  omega1_history, omega2_history, model_array,
                  Loss_excite_history, Loss_Lc_history]

    print(f"\n{'='*60}")
    print(f"  RESULTADO — Schwarzschild, Modo N={target_mode}  {re_tag}")
    print(f"{'='*60}")
    print(f"  ω = {omega1_history[-1]:.6f} + {omega2_history[-1]:.6f}i")

    coeffs_final = model_cpu.get_coeffs_complex()
    print(f"  Coeficientes finales:")
    for k, c in enumerate(coeffs_final):
        print(f"    c{k} = {c.real:.6f} + {c.imag:.6f}i")
    print(f"  Tiempo: {runTime:.2f} s")
    print(f"{'='*60}\n")

    return model_cpu, loss_hists, runTime


def QNMs(loss_hists, path, runTime, hyperparams=None):
    """Argmin loss (primario) + promedio ultimo 10% (estabilidad)."""
    o1h = loss_hists[3]; o2h = loss_hists[4]
    n   = len(o1h); k10 = max(1, n // 10)
    tl  = [float(x) for x in loss_hists[0]]
    ep  = int(np.argmin(tl)); ml = tl[ep]
    omega1 = float(o1h[ep]); omega2 = float(o2h[ep])
    o1a = float(np.mean(o1h[-k10:])); o2a = float(np.mean(o2h[-k10:]))
    rs  = float(np.std(o1h[-k10:])); ims = float(np.std(o2h[-k10:]))
    ok  = abs(omega1 - o1a) < 0.05 and abs(omega2 - o2a) < 0.05
    print(f"\n{'='*55}")
    print(f"[Argmin — VALOR PRIMARIO]  epoca {ep}")
    print(f"  w = {omega1:.6f} + {omega2:.6f}i   loss={ml:.4e}")
    print(f"[Promedio ultimo 10% ({k10} pts)]  w_avg={o1a:.6f}+{o2a:.6f}i")
    print(f"  desv(Re)={rs:.4f}  desv(Im)={ims:.4f}  {'✓' if ok else '⚠ usar argmin'}")
    print(f"Tiempo: {runTime:.2f} s\n{'='*55}\n")
    with open(os.path.join(path, 'QNM_Values.txt'), 'w') as f:
        f.write(f"Argmin\nRe:{omega1}\nIm:{omega2}\nloss:{ml:.4e}\n")
        f.write(f"\nAvg10%\nRe:{o1a}\nIm:{o2a}\nsigma:{rs:.6f},{ims:.6f}\n")
        if hyperparams:
            f.write("\nHiperparametros\n")
            for k, v in hyperparams.items(): f.write(f"{k}:{v}\n")
        model = loss_hists[5][0][0]
        if hasattr(model, 'get_coeffs_complex'):
            f.write("\nCoeficientes\n")
            for k, c in enumerate(model.get_coeffs_complex()):
                f.write(f"c{k}={c.real:.8f}+{c.imag:.8f}i\n")
    return omega1, omega2


# ═══════════════════════════════════════════════════════════════
# GRÁFICAS
# ═══════════════════════════════════════════════════════════════

def plot_qnm_convergence(loss_hists, path, k_avg=500):
    """Convergencia de Re(ω) e Im(ω)."""
    omega1_history = np.array(loss_hists[3])
    omega2_history = np.array(loss_hists[4])
    epocas         = np.arange(len(omega1_history)) / 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(epocas, omega1_history, color='#2196F3', lw=1.5, label='Re(ω)')
    ax1.set_ylabel('Re(ω)'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(epocas, omega2_history, color='#9C27B0', lw=1.5, label='Im(ω)')
    ax2.set_ylabel('Im(ω)'); ax2.set_xlabel('Épocas (×1000)')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle('Convergencia ω — Schwarzschild (Chebyshev+HardMask)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(path, "conv_omega_juntas.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_loss(loss_hists, path, logscale=True):
    """Gráficas de todas las pérdidas."""
    epocas_arr = np.arange(len(loss_hists[0])) / 1000

    all_losses = [
        ('Total',        0, '#212121', '-',  2.0),
        ('PDE',          1, '#2196F3', '-',  1.5),
        ('Anticolapso',  6, '#E91E63', '--', 1.3),
        ('Barreras',     7, '#FF9800', '--', 1.3),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, idx, color, ls, lw in all_losses:
        if idx >= len(loss_hists):
            continue
        vals = np.array(loss_hists[idx], dtype=float)
        if np.max(np.abs(vals)) < 1e-15:
            continue
        ax.plot(epocas_arr, vals, color=color, lw=lw, ls=ls, label=name)

    if logscale:
        ax.set_yscale('log')
    ax.set_title('Pérdidas — Schwarzschild (Chebyshev+HardMask)', fontsize=13)
    ax.set_xlabel('Épocas (×1000)')
    ax.set_ylabel('Pérdida')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(path, "loss_todas.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_wavefunction(model, loss_hists, path, n_points=500, target_mode=None):
    """Grafica U(z) evaluando el polinomio."""
    model.eval()
    z_np = np.linspace(1e-4, 1.0 - 1e-4, n_points)
    z_t  = torch.tensor(z_np, dtype=dtype).reshape(-1, 1)

    with torch.no_grad():
        U, _, _, omega = model(z_t)
        U_np = U.cpu().numpy()

    U_re  = U_np.real
    U_im  = U_np.imag
    U_abs = np.abs(U_np)

    omega_re  = model.omega_re.item()
    omega_im  = model.omega_im.item()
    mode_str  = f"N={target_mode}" if target_mode is not None else ""
    omega_str = f"ω = {omega_re:.5f} + {omega_im:.5f}i"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z_np, U_re,  color='#2196F3', lw=1.8, label=r'Re($U$)')
    ax.plot(z_np, U_im,  color='#E91E63', lw=1.8, label=r'Im($U$)')
    ax.plot(z_np, U_abs, color='#4CAF50', lw=1.8, ls='--', label=r'$|U|$')
    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$U(z)$')
    ax.set_title(
        rf'Función de onda — Schwarzschild  {mode_str}'
        + '\n' + rf'${omega_str}$',
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    coeffs      = model.get_coeffs_complex()
    coeffs_text = "\n".join([f"c{k}={c:.3f}" for k, c in enumerate(coeffs)])
    ax.text(0.98, 0.02, coeffs_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontfamily='monospace')

    fig.tight_layout()
    fig.savefig(os.path.join(path, "wavefunction.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    U_abs2 = U_re**2 + U_im**2
    ax.plot(z_np, U_abs2, color='#FF9800', lw=1.8)
    ax.fill_between(z_np, U_abs2, alpha=0.25, color='#FF9800')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$|U|^2$')
    ax.set_title(f'Densidad — Schwarzschild  {mode_str}', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(path, "wavefunction_density.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)

    return z_np, U_re, U_im


def generar_diagnostico(loss_hists, path, runTime, hyperparams, target_mode, known_im_qnms):
    """Diagnóstico completo por corrida."""
    o1 = np.array(loss_hists[3]); o2 = np.array(loss_hists[4])
    lt = np.array(loss_hists[0]); lp = np.array(loss_hists[1])
    n  = len(o1); k10 = max(1, n // 10); ep = int(np.argmin(lt))
    re_best   = float(o1[ep]); im_best = float(o2[ep])
    re_avg    = float(np.mean(o1[-k10:])); im_avg = float(np.mean(o2[-k10:]))
    re_std    = float(np.std(o1[-k10:]));  im_std = float(np.std(o2[-k10:]))
    pde_final = float(lp[-1])
    consistent = abs(re_best - re_avg) < 0.05 and abs(im_best - im_avg) < 0.05

    if pde_final < 1e-6 and re_std < 0.01 and im_std < 0.01:
        veredicto = "CONVERGIO (PDE<1e-6 w estable)"
    elif pde_final < 1e-6:
        veredicto = "CONVERGIO PDE (usar argmin)"
    elif pde_final < 1e-4:
        veredicto = "PROBABLE (PDE<1e-4)"
    else:
        veredicto = "NO convergio (PDE alta)"

    model  = loss_hists[5][0][0]
    coeffs = model.get_coeffs_complex() if hasattr(model, 'get_coeffs_complex') else []
    force_re_zero = getattr(model, 'force_re_zero', False)

    lines = []
    lines.append("=" * 60)
    lines.append(f"  DIAGNÓSTICO POLINOMIAL — Schwarzschild, Modo N={target_mode}")
    lines.append(f"  Re(ω): {'FORZADO A 0' if force_re_zero else 'libre'}")
    lines.append("=" * 60)
    lines.append(f"\n  w argmin : {re_best:.6f}+{im_best:.6f}i [ep {ep}]")
    lines.append(f"  w avg10% : {re_avg:.6f}+{im_avg:.6f}i  s=({re_std:.4f},{im_std:.4f})")
    lines.append(f"  Consist  : {chr(10003) if consistent else chr(9888)}")
    lines.append(f"  Loss PDE : {pde_final:.4e}")
    lines.append(f"  Veredicto: {veredicto}")
    lines.append(f"\n  Coeficientes:")
    for k, c in enumerate(coeffs):
        lines.append(f"    c{k} = {c.real:.8f} + {c.imag:.8f}i")
    lines.append(f"\n  Barreras Im conocidas: {known_im_qnms}")
    lines.append(f"  Tiempo: {runTime:.2f} s")
    lines.append(f"\n  Hiperparámetros:")
    for k, v in hyperparams.items():
        lines.append(f"    {k}: {v}")
    lines.append("=" * 60)

    txt = "\n".join(lines)
    print("\n" + txt)

    with open(os.path.join(path, "diagnostico.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

    return txt