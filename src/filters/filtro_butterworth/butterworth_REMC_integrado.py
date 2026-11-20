# -*- coding: utf-8 -*-
"""
BUTTERWORTH: FUNCIONES BASICAS (REMC), RECONSTRUCCION, ERROR Y GRAFICAS

Integra en una sola estructura:
  1. Calculo de la covarianza normalizada y del espectro Butterworth.
  2. Diagrama de los 2n polos de H(s)H(-s).
  3. Funciones basicas B_j(t) y B_j(w).
  4. Funcion de reconstruccion y varianza/error de reconstruccion.
  5. Comparativas por N, por grupos de orden y por orden fijo.
  6. Validacion numerica del tiempo de covarianza y reporte CSV.

La normalizacion usada es S0=2 y wc=n*sin(pi/(2n)); por tanto K(0)=1
 y el tiempo de covarianza teorico Tc=int_0^inf rho(tau)dtau vale 1 s.

Estructura de salida:
  Resultados_Butterworth/
    orden_P1/
      graficas_filtro/
        espectro_covarianza.png
        polos_Hs_Hmenos_s.png
      N_4/ ... cinco graficas REMC
      N_9/ ...
      comparativas_...png
    orden_P3/ ...
    ...
    graficas_por_orden/
    validacion_Tc.csv

Requisitos: pip install numpy matplotlib
"""

from pathlib import Path
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt

trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# 1. PARAMETROS GENERALES
# ============================================================
ordenes = [1, 3, 5, 7, 9, 11]
valores_N = [2,4,9,19]
delta_T = 0.5                         # separacion uniforme entre muestras [s]

S0 = 2.0
wc_tabla = {n: n * np.sin(np.pi / (2 * n)) for n in ordenes}

# Malla temporal para las funciones basicas
DT = 0.01
t_margen = 5.0

# Malla de frecuencia (solo para graficas de frecuencia)
# Cada orden usa su propia malla, desde -3*wc hasta +3*wc.
FACTOR_FRECUENCIA = 3.0               # mostrar al menos tres veces wc a cada lado
dw = 0.005                            # separacion aproximada [rad/s]

# Ajuste de la escala vertical del retardo de grupo
GD_MODO = "banda_central"             # banda_central | percentil_global | completo
GD_PERCENTILES = (1.0, 99.0)
GD_TOL = 1e-30

# Integracion numerica para validar Tc
TC_TAU_MAX = 9.0
TC_DT = 0.01

# Patron de amplitudes de las muestras
PATRON = [1, 1.4, 0.6, -0.2, -0.5, -1.4, 1, 1.2, 0.5]


# ============================================================
# 2. FUNCIONES AUXILIARES Y MODELO BUTTERWORTH
# ============================================================
def construir_muestras(N):
    """Genera N amplitudes repitiendo el patron original."""
    secuencia = []
    while len(secuencia) < N:
        secuencia += PATRON + [1.2]
    return np.asarray(secuencia[:N], dtype=float)


def crear_butterworth(orden):
    """Devuelve wc, K0, R(tau) y S(w) para el orden indicado.

    Los polos usados en la suma de residuos son los n polos de frecuencia
    del semiplano superior: q_k=wc*exp(j*pi*(2k+1)/(2n)). Como s_k=j*q_k,
    esta expresion es equivalente a la suma sobre los polos S_k del
    semiplano izquierdo del plano s:
        rho(tau)=-(1/n)*sum_k S_k*exp(S_k*abs(tau)).
    """
    if orden < 1:
        raise ValueError("El orden del filtro debe ser un entero >= 1.")

    wc = float(wc_tabla[orden])
    k = np.arange(orden)
    q = wc * np.exp(1j * np.pi * (2 * k + 1) / (2 * orden))
    polos_s = 1j * q

    # K(0) = (1/(2*pi))* integral_{-inf}^{inf} S(w) dw
    K0 = S0 * wc / (2.0 * orden * np.sin(np.pi / (2.0 * orden)))

    def R(tau):
        tau_abs = np.abs(np.asarray(tau, dtype=float))
        suma = np.sum(
            polos_s * np.exp(np.multiply.outer(tau_abs, polos_s)), axis=-1
        )
        cov = -(1.0 / orden) * suma
        # La covarianza es real; se normaliza para obtener rho(0)=1.
        return np.asarray((S0 / 2.0) * cov.real / K0, dtype=float)

    def S(w):
        w_arr = np.asarray(w, dtype=float)
        return S0 / (1.0 + (np.abs(w_arr) / wc) ** (2 * orden))

    return wc, K0, R, S, polos_s


def guardar_figura(fig, carpeta, nombre, dpi=140):
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ruta


def limites_retardo(valores, delta, w=None, wc=None):
    """Calcula limites verticales utiles para el retardo de grupo."""
    v = np.asarray(valores, dtype=float)
    if GD_MODO == "completo":
        p_lo, p_hi = 0.0, 100.0
        seleccion = v
    elif GD_MODO == "banda_central" and w is not None and wc is not None:
        p_lo, p_hi = GD_PERCENTILES
        seleccion = v[:, np.abs(w) <= wc]
    else:
        p_lo, p_hi = 2.0, 98.0
        seleccion = v

    seleccion = seleccion[np.isfinite(seleccion)]
    if not seleccion.size:
        return None

    q_lo, q_hi = np.percentile(seleccion, [p_lo, p_hi])
    rango = q_hi - q_lo
    if rango <= 0:
        return None

    lo = q_lo - 0.10 * rango
    hi = q_hi + 0.10 * rango
    if hi - lo < 2.0 * delta:
        centro = 0.5 * (hi + lo)
        lo, hi = centro - delta, centro + delta
    return (lo, hi) if hi > lo else None


def polos_completos(orden, wc):
    """Calcula los 2n polos de H(s)H(-s) en el plano s."""
    m = np.arange(1, 2 * orden + 1)
    angulos = (2 * m - 1) * np.pi / (2 * orden)
    return 1j * wc * np.exp(1j * angulos)


def graficar_filtro_y_polos(orden, wc, R_fun, S_fun, carpeta_orden):
    """Guarda las dos graficas del programa Butterworth previo.

    Se guardan una sola vez por orden porque no dependen del numero N
    ni de las amplitudes de las muestras.
    """
    carpeta = carpeta_orden / "graficas_filtro"
    carpeta.mkdir(parents=True, exist_ok=True)

    # Figura A: espectro y covarianza, en paneles horizontales.
    w_f = np.linspace(-3.0 * wc, 3.0 * wc, 3000)
    tau_max = 6.0 / (wc * np.sin(np.pi / (2.0 * orden)))
    tau_f = np.linspace(-tau_max, tau_max, 4001)
    S_f = S_fun(w_f)
    rho_f = R_fun(tau_f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(w_f, S_f, color="tab:blue", lw=2)
    ax1.axvline(wc, color="gray", ls="--", lw=1)
    ax1.axvline(-wc, color="gray", ls="--", lw=1)
    ax1.plot(
        [wc, -wc], [S0 / 2.0, S0 / 2.0], "o", color="tab:red", ms=5,
        label=r"$\omega=\pm\omega_c$ (mitad de potencia)"
    )
    ax1.set_xlabel(r"$\omega$ [rad/s]")
    ax1.set_ylabel(r"$S(\omega)$")
    ax1.set_title(rf"Espectro de potencia, $n={orden}$, $\omega_c={wc:.4f}$")
    ax1.set_xlim(-FACTOR_FRECUENCIA * wc, FACTOR_FRECUENCIA * wc)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower center", fontsize=8)

    ax2.plot(tau_f, rho_f, color="tab:green", lw=2, label=r"$\rho_x(\tau)$ (residuos)")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel(r"$\tau$ [s]")
    ax2.set_ylabel(r"$\rho_x(\tau)=K(\tau)/K(0)$")
    ax2.set_title(r"Covarianza normalizada, $T_c=1$ s")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    guardar_figura(fig, carpeta, "espectro_covarianza.png")

    # Figura B: diagrama de polos de H(s)H(-s).
    s_all = polos_completos(orden, wc)
    izq = s_all[s_all.real < 0]
    der = s_all[s_all.real > 0]
    lim = 1.55 * wc

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    flecha = dict(arrowstyle="-|>", color="k", lw=1.2)
    ax.annotate("", xy=(lim, 0), xytext=(-lim, 0), arrowprops=flecha)
    ax.annotate("", xy=(0, lim), xytext=(0, -lim), arrowprops=flecha)
    ax.text(lim, -0.07 * lim, r"$\sigma$", ha="right", va="top", fontsize=14)
    ax.text(0.04 * lim, lim, r"$j\omega$", ha="left", va="top", fontsize=14)
    ax.text(0.97 * lim, 0.93 * lim, r"plano $s$", ha="right", va="top",
            style="italic", fontsize=12)

    ax.add_patch(plt.Circle((0, 0), wc, fill=False, ls="--", color="gray", lw=1.2))
    for polo in s_all:
        ax.plot([0, polo.real], [0, polo.imag], ls=":", color="gray", lw=1)

    ax.plot(izq.real, izq.imag, "x", color="tab:blue", ms=12, mew=2.5,
            label=r"Polos de $H(s)$ (semiplano izquierdo, $S_k$)")
    ax.plot(der.real, der.imag, "x", color="tab:red", ms=12, mew=2.5,
            label=r"Polos de $H(-s)$ (semiplano derecho)")

    # Polos S_k del semiplano izquierdo, en el mismo orden que la suma de residuos.
    k = np.arange(1, orden + 1)
    theta = (2 * k - 1) * np.pi / (2 * orden)
    s_left = wc * (-np.sin(theta) + 1j * np.cos(theta))
    for indice, polo in enumerate(s_left, start=1):
        radio = abs(polo)
        ux, uy = polo.real / radio, polo.imag / radio
        ha = "right" if ux < -0.15 else ("left" if ux > 0.15 else "center")
        va = "bottom" if uy >= -0.15 else "top"
        if abs(polo.imag) < 1e-9:
            ha, va, uy = "center", "bottom", 0.45
        ax.text(polo.real + 0.10 * wc * ux, polo.imag + 0.10 * wc * uy,
                rf"$S_{indice}$", color="tab:blue", fontsize=12, ha=ha, va=va)

    candidatos = der[der.imag >= -1e-9]
    if candidatos.size == 0:
        candidatos = der
    polo_ref = candidatos[np.argmax(candidatos.imag)]
    ax.plot([0, polo_ref.real], [0, polo_ref.imag], ls="--", color="k", lw=1.2)
    angulo = np.angle(polo_ref)
    ax.text(
        0.5 * polo_ref.real + 0.05 * wc * np.sin(angulo),
        0.5 * polo_ref.imag - 0.05 * wc * np.cos(angulo),
        rf"$\omega_c={wc:.4g}$", fontsize=11,
        rotation=np.degrees(angulo), rotation_mode="anchor", ha="center", va="top"
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for x0, etiqueta in ((-wc, r"$-\omega_c$"), (wc, r"$\omega_c$")):
        ax.plot([x0, x0], [-0.025 * lim, 0.025 * lim], color="k", lw=1.2)
        ax.text(x0, -0.05 * lim, etiqueta, ha="center", va="top", fontsize=11)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    ax.set_title(rf"Polos de $H(s)H(-s)$ para $n={orden}$ y $\omega_c={wc:.4g}$",
                 fontsize=12, pad=12)
    fig.tight_layout()
    guardar_figura(fig, carpeta, "polos_Hs_Hmenos_s.png")


# ============================================================
# 3. CARPETA RAIZ Y MALLA DE FRECUENCIA
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

carpeta_raiz = BASE_DIR / "Resultados_Butterworth_Residuos"
carpeta_raiz.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. BUCLE PRINCIPAL: N x ORDEN
# ============================================================
registros_tc = []
resultados_graficas = {}

for N in valores_N:
    T = delta_T * np.arange(N, dtype=float)
    x_muestras = construir_muestras(N)

    nt = int(round((T[-1] - T[0] + 2.0 * t_margen) / DT)) + 1
    t = (T[0] - t_margen) + np.arange(nt) * DT
    idx_T = np.rint((T - t[0]) / DT).astype(int)

    print("\n" + "#" * 72)
    print(f"PROCESANDO N={N} muestras (delta_T={delta_T:g} s, "
          f"T de {T[0]:g} a {T[-1]:g} s)")
    print("#" * 72)

    for orden in ordenes:
        wc, K0, R_fun, S_fun, polos_s = crear_butterworth(orden)
        carpeta_orden = carpeta_raiz / f"orden_P{orden}"
        carpeta_N = carpeta_orden / f"N_{N}"
        carpeta_N.mkdir(parents=True, exist_ok=True)
        prefijo = f"Butterworth_P{orden}_N{N}"

        # Las graficas del filtro y de sus polos no dependen de N: se crean
        # solo en la primera pasada de cada orden.
        carpeta_graficas_filtro = carpeta_orden / "graficas_filtro"
        if not (carpeta_graficas_filtro / "espectro_covarianza.png").exists() or \
           not (carpeta_graficas_filtro / "polos_Hs_Hmenos_s.png").exists():
            graficar_filtro_y_polos(orden, wc, R_fun, S_fun, carpeta_orden)

        print("\n" + "=" * 66)
        print(f"Butterworth: orden n={orden}, N={N}, wc={wc:.6g} rad/s")
        print("=" * 66)

        # Malla de frecuencia especifica para este orden: [-3 wc, +3 wc].
        # Se fija el numero de puntos para conservar una resolucion cercana a dw
        # y asegurar que ambos extremos coincidan exactamente con +/-3 wc.
        w_lim = FACTOR_FRECUENCIA * wc
        n_w = int(np.ceil((2.0 * w_lim) / dw)) + 1
        w = np.linspace(-w_lim, w_lim, n_w)
        Sw = S_fun(w)
        I0 = 2.0 * np.pi * K0

        # --------------------------------------------------------
        # Validacion del tiempo de covarianza
        # --------------------------------------------------------
        tau_tc = np.arange(0.0, TC_TAU_MAX + 0.5 * TC_DT, TC_DT)
        rho_tc = R_fun(tau_tc)
        Tc_int = float(trapz(rho_tc, x=tau_tc))
        Tc_abs = float(trapz(np.abs(rho_tc), x=tau_tc))
        Tc_teorico = float(orden * np.sin(np.pi / (2.0 * orden)) / wc)
        print(f"int rho d tau (0..{TC_TAU_MAX:g}) = {Tc_int:.6f} s "
              f"(teorico: {Tc_teorico:.6f} s)")
        print(f"int |rho| d tau (0..{TC_TAU_MAX:g}) = {Tc_abs:.6f} s")

        # --------------------------------------------------------
        # Matriz de covarianza entre muestras
        # --------------------------------------------------------
        Rmat = R_fun(T[:, None] - T[None, :])
        Rmat = 0.5 * (Rmat + Rmat.T)
        cond_R = float(np.linalg.cond(Rmat))
        print(f"cond(R) = {cond_R:.6e}")
        if cond_R > 1e12:
            print("ADVERTENCIA: R esta mal condicionada; los resultados pueden "
                  "ser sensibles al redondeo numerico.")

        # --------------------------------------------------------
        # Funciones basicas en el tiempo
        # B_j(t) = sum_i rho(t-T_i) a_ij, A=Rmat^{-1}
        # --------------------------------------------------------
        Kmat = R_fun(t[None, :] - T[:, None])
        AK = np.linalg.solve(Rmat, Kmat)
        B_t = AK
        err_interp = float(np.max(np.abs(B_t[:, idx_T] - np.eye(N))))
        print(f"max|B_j(T_k)-delta_jk| = {err_interp:.3e}")

        # Funcion de reconstruccion para las amplitudes dadas
        FR = x_muestras @ B_t

        # --------------------------------------------------------
        # Error/varianza de reconstruccion y puntos medios
        # --------------------------------------------------------
        Var = 1.0 - np.sum(Kmat * AK, axis=0)
        t_medios = 0.5 * (T[:-1] + T[1:])
        Kmat_medios = R_fun(t_medios[None, :] - T[:, None])
        AK_medios = np.linalg.solve(Rmat, Kmat_medios)
        Var_medios = 1.0 - np.sum(Kmat_medios * AK_medios, axis=0)

        mascara = (t >= T[0]) & (t <= T[-1])
        Var_ventana = Var[mascara]
        t_ventana = t[mascara]
        i_max_real = int(np.argmax(Var_ventana))
        i_max_medio = int(np.argmax(Var_medios))
        i_min_medio = int(np.argmin(Var_medios))
        max_error_real = float(Var_ventana[i_max_real])
        max_error_medios = float(Var_medios[i_max_medio])
        min_error_medios = float(Var_medios[i_min_medio])
        print(f"Error maximo en [T0,TN] = {max_error_real:.6e} "
              f"en t={t_ventana[i_max_real]:.6g} s")
        print(f"Error maximo en puntos medios = {max_error_medios:.6e} "
              f"en t={t_medios[i_max_medio]:.6g} s")
        print(f"Error minimo en puntos medios = {min_error_medios:.6e} "
              f"en t={t_medios[i_min_medio]:.6g} s")
        if np.min(Var) < -1e-8:
            print("ADVERTENCIA: Var(t) contiene valores negativos apreciables.")

        # --------------------------------------------------------
        # Funciones basicas en frecuencia: magnitud, fase y retardo
        # --------------------------------------------------------
        E = np.exp(-1j * T[:, None] * w[None, :])
        H = np.linalg.solve(Rmat, E)
        B_w = (2.0 * np.pi / I0) * H * Sw[None, :]
        modulo_B = np.abs(B_w)
        fase_B = np.unwrap(np.angle(B_w), axis=1)

        Hd = np.linalg.solve(Rmat, T[:, None] * E)
        den_gd = np.abs(H) ** 2
        num_gd = np.real(Hd * np.conj(H))
        retardo_grupo = np.full(den_gd.shape, np.nan, dtype=float)
        for j in range(N):
            max_den = np.max(den_gd[j])
            valido = np.isfinite(den_gd[j]) & (den_gd[j] > GD_TOL * max_den)
            retardo_grupo[j, valido] = num_gd[j, valido] / den_gd[j, valido]

        # --------------------------------------------------------
        # Covarianza para graficas y reporte
        # --------------------------------------------------------
        tau_cov = np.linspace(-5.0, 5.0, 2001)
        K_cov = R_fun(tau_cov)

        resultados_graficas[(N, orden)] = {
            "N": N,
            "orden": orden,
            "wc": wc,
            "T": T.copy(),
            "t": t.copy(),
            "B_t": B_t.copy(),
            "FR": FR.copy(),
            "Var": Var.copy(),
            "t_medios": t_medios.copy(),
            "Var_medios": Var_medios.copy(),
            "t_ventana": t_ventana.copy(),
            "max_error_real": max_error_real,
            "w": w.copy(),
            "modulo_B": modulo_B.copy(),
            "fase_B": fase_B.copy(),
            "retardo_grupo": retardo_grupo.copy(),
            "tau_cov": tau_cov,
            "K_cov": K_cov.copy(),
            "Sw": Sw.copy(),
            "Tc_teorico": Tc_teorico,
            "Tc_int": Tc_int,
            "Tc_abs": Tc_abs,
        }

        registros_tc.append({
            "N": N,
            "orden": orden,
            "wc_rad_s": wc,
            "K0": K0,
            "Tc_teorico_s": Tc_teorico,
            f"int_rho_0_{TC_TAU_MAX:g}_s": Tc_int,
            f"int_abs_rho_0_{TC_TAU_MAX:g}_s": Tc_abs,
            "cond_R": cond_R,
            "err_interpolacion": err_interp,
            "error_max_real": max_error_real,
            "error_max_puntos_medios": max_error_medios,
            "error_min_puntos_medios": min_error_medios,
        })

        # ========================================================
        # FIGURA 1: FUNCIONES BASICAS EN EL TIEMPO
        # ========================================================
        fig, ax = plt.subplots(figsize=(11, 6))
        for j in range(N):
            ax.plot(t, B_t[j], linewidth=1.05, label=rf"$B_{{{j+1}}}(t)$")
        ax.grid(True)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$B_j(t)$")
        ax.set_title(f"Funciones basicas en tiempo - Butterworth, n={orden}, N={N}")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
        guardar_figura(fig, carpeta_N, f"{prefijo}_funciones_basicas_tiempo.png")

        # ========================================================
        # FIGURA 2: FUNCION DE RECONSTRUCCION
        # ========================================================
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(t, FR, "b-", linewidth=1.4, label=r"$FR(t)$")
        ax.plot(T, x_muestras, "ro", markerfacecolor="r", label="Muestras")
        ax.grid(True)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Amplitud")
        ax.set_title(f"Funcion de reconstruccion - Butterworth, n={orden}, N={N}")
        ax.legend(loc="best")
        guardar_figura(fig, carpeta_N, f"{prefijo}_reconstruccion.png")

        # ========================================================
        # FIGURA 3: ERROR DE RECONSTRUCCION Y PUNTOS MEDIOS
        # ========================================================
        fig, ax = plt.subplots(figsize=(10, 5.5))
        piso = 1e-12
        ax.semilogy(t, np.maximum(Var, piso), "k-", linewidth=1.25, label=r"$Var(t)$")
        for ti in T:
            ax.axvline(ti, linestyle=":", color="0.65", linewidth=0.8)
        ax.semilogy(t_medios, np.maximum(Var_medios, piso), "ro",
                    markerfacecolor="r", label="Puntos medios")
        ax.plot(t_ventana[i_max_real], max(max_error_real, piso), "mp",
                markerfacecolor="m", markersize=10,
                label=f"Maximo real = {max_error_real:.3e}")
        ax.plot(t_medios[i_min_medio], max(min_error_medios, piso), "bv",
                markerfacecolor="b", markersize=8,
                label=f"Minimo puntos medios = {min_error_medios:.3e}")
        ax.grid(True, which="both")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$Var(t)$")
        ax.set_title(f"Error de reconstruccion - Butterworth, n={orden}, N={N}")
        ax.legend(loc="best")
        guardar_figura(fig, carpeta_N, f"{prefijo}_error_reconstruccion.png")

        # ========================================================
        # FIGURA 4: MAGNITUD, FASE Y RETARDO DE GRUPO
        # ========================================================
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for j in range(N):
            axes[0].plot(w, modulo_B[j], linewidth=1.0, label=rf"$B_{{{j+1}}}$")
            axes[1].plot(w, fase_B[j], linewidth=1.0)
            axes[2].plot(w, retardo_grupo[j], linewidth=1.0)
        axes[0].grid(True)
        axes[0].set_ylabel(r"$|B_j(\omega)|$")
        axes[0].set_title(f"Magnitud de las funciones basicas - n={orden}, N={N}")
        axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
        axes[1].grid(True)
        axes[1].set_ylabel("Fase [rad]")
        axes[1].set_title("Fase de las funciones basicas")
        axes[2].grid(True)
        axes[2].set_xlabel(r"$\omega$ [rad/s]")
        axes[2].set_ylabel(r"$\tau_g$ [s]")
        axes[2].set_title("Retardo de grupo")
        limites = limites_retardo(retardo_grupo, delta_T, w, wc)
        if limites:
            axes[2].set_ylim(*limites)
        for ax in axes:
            ax.set_xlim(-FACTOR_FRECUENCIA * wc, FACTOR_FRECUENCIA * wc)
        fig.tight_layout()
        guardar_figura(fig, carpeta_N, f"{prefijo}_funciones_basicas_frecuencia.png")

        # ========================================================
        # FIGURA 5: COVARIANZA Y DENSIDAD ESPECTRAL
        # ========================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(tau_cov, K_cov, "k-", linewidth=1.25)
        axes[0].grid(True)
        axes[0].set_xlabel(r"$\tau$ [s]")
        axes[0].set_ylabel(r"$K(\tau)/K(0)$")
        axes[0].set_title(rf"Covarianza normalizada; $T_c$ teorico={Tc_teorico:.4f} s")
        axes[1].plot(w, Sw, "b-", linewidth=1.15)
        axes[1].axvline(wc, color="0.4", ls="--", label=rf"$\omega_c={wc:.3f}$")
        axes[1].axvline(-wc, color="0.4", ls="--")
        axes[1].set_xlim(-FACTOR_FRECUENCIA * wc, FACTOR_FRECUENCIA * wc)
        axes[1].grid(True)
        axes[1].set_xlabel(r"$\omega$ [rad/s]")
        axes[1].set_ylabel(r"$S(\omega)$")
        axes[1].set_title("Densidad espectral")
        axes[1].legend()
        fig.tight_layout()
        guardar_figura(fig, carpeta_N, f"{prefijo}_covarianza_espectro.png")

        print(f"Resultados n={orden}, N={N} guardados en: {carpeta_N}")


# ============================================================
# 5. COMPARATIVAS POR N: ORDENES IMPARES Y PARES
# ============================================================
def guardar_comparativa_en_cada_orden(fig, nombre_archivo):
    """Guarda la comparativa en el primer orden y la replica en los demas."""
    primero = carpeta_raiz / f"orden_P{ordenes[0]}" / nombre_archivo
    primero.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(primero, dpi=140, bbox_inches="tight")
    plt.close(fig)
    for orden_destino in ordenes[1:]:
        destino = carpeta_raiz / f"orden_P{orden_destino}" / nombre_archivo
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primero, destino)


def estilo_eje(ax):
    ax.grid(True, alpha=0.28)
    ax.tick_params(axis="both", labelsize=8)


def generar_comparativa_tiempo_error(N, grupo, etiqueta):
    fig, axes = plt.subplots(len(grupo), 2, figsize=(15, 4.4 * len(grupo)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for fila, orden in enumerate(grupo):
        r = resultados_graficas[(N, orden)]
        ax_b, ax_e = axes[fila]
        for j in range(r["B_t"].shape[0]):
            ax_b.plot(r["t"], r["B_t"][j], lw=0.9,
                      color=colores[j % len(colores)], label=rf"$B_{{{j+1}}}(t)$")
        ax_b.set_title(f"Funciones basicas - n={orden}, wc={r['wc']:.5g}")
        ax_b.set_xlabel("t [s]")
        ax_b.set_ylabel(r"$B_j(t)$")
        estilo_eje(ax_b)

        piso = 1e-12
        ax_e.semilogy(r["t"], np.maximum(r["Var"], piso), color="black", lw=1.15,
                      label=r"$Var(t)$")
        ax_e.semilogy(r["t_medios"], np.maximum(r["Var_medios"], piso), "o",
                      color="tab:red", ms=4, label="Puntos medios")
        for ti in r["T"]:
            ax_e.axvline(ti, linestyle=":", color="0.65", linewidth=0.6)
        mask = (r["t"] >= r["T"][0]) & (r["t"] <= r["T"][-1])
        if np.any(mask):
            idx_max = int(np.argmax(r["Var"][mask]))
            ax_e.plot(r["t"][mask][idx_max], max(float(r["Var"][mask][idx_max]), piso),
                      "mp", ms=8, label=f"Max. real={r['max_error_real']:.2e}")
        idx_min = int(np.argmin(r["Var_medios"]))
        ax_e.plot(r["t_medios"][idx_min], max(float(r["Var_medios"][idx_min]), piso),
                  "bv", ms=7, label=f"Min. medios={r['Var_medios'][idx_min]:.2e}")
        ax_e.set_title(f"Error de reconstruccion - n={orden}")
        ax_e.set_xlabel("t [s]")
        ax_e.set_ylabel(r"$Var(t)$")
        estilo_eje(ax_e)
        ax_e.legend(loc="best", fontsize=7)

    fig.suptitle(f"Tiempo y error de reconstruccion - ordenes {etiqueta}; N={N}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    guardar_comparativa_en_cada_orden(
        fig, f"comparativas_{etiqueta}_tiempo_error_N{N}.png"
    )


def generar_comparativa_funciones_frecuencia(N, grupo, etiqueta):
    fig, axes = plt.subplots(len(grupo), 3, figsize=(20, 4.3 * len(grupo)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for fila, orden in enumerate(grupo):
        r = resultados_graficas[(N, orden)]
        for j in range(r["modulo_B"].shape[0]):
            color = colores[j % len(colores)]
            axes[fila, 0].plot(r["w"], r["modulo_B"][j], lw=0.85, color=color,
                               label=rf"$B_{{{j+1}}}$")
            axes[fila, 1].plot(r["w"], r["fase_B"][j], lw=0.85, color=color)
            axes[fila, 2].plot(r["w"], r["retardo_grupo"][j], lw=0.85, color=color)
        axes[fila, 0].set_title(f"Magnitud - n={orden}, wc={r['wc']:.5g}")
        axes[fila, 1].set_title(f"Fase - n={orden}")
        axes[fila, 2].set_title(f"Retardo de grupo - n={orden}")
        axes[fila, 0].set_ylabel(r"$|B_j(\omega)|$")
        axes[fila, 1].set_ylabel("Fase [rad]")
        axes[fila, 2].set_ylabel(r"$\tau_g$ [s]")
        for ax in axes[fila]:
            ax.set_xlabel(r"$\omega$ [rad/s]")
            estilo_eje(ax)
        limites = limites_retardo(r["retardo_grupo"], delta_T, r["w"], r["wc"])
        if limites:
            axes[fila, 2].set_ylim(*limites)
        for ax in axes[fila]:
            ax.set_xlim(-FACTOR_FRECUENCIA * r["wc"], FACTOR_FRECUENCIA * r["wc"])

    fig.suptitle(f"Funciones basicas en frecuencia - ordenes {etiqueta}; N={N}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    guardar_comparativa_en_cada_orden(
        fig, f"comparativas_{etiqueta}_funciones_basicas_frecuencia_N{N}.png"
    )


def generar_comparativa_covarianza_espectro(N, grupo, etiqueta):
    fig, axes = plt.subplots(len(grupo), 2, figsize=(15, 4.1 * len(grupo)), squeeze=False)
    for fila, orden in enumerate(grupo):
        r = resultados_graficas[(N, orden)]
        axes[fila, 0].plot(r["tau_cov"], r["K_cov"], color="black", lw=1.15)
        axes[fila, 0].axvline(0, color="0.65", linestyle=":", lw=0.7)
        axes[fila, 0].set_title(
            f"Covarianza - n={orden}; Tc teorico={r['Tc_teorico']:.4f} s; "
            f"int rho(0..{TC_TAU_MAX:g})={r['Tc_int']:.4f} s"
        )
        axes[fila, 0].set_xlabel(r"$\tau$ [s]")
        axes[fila, 0].set_ylabel(r"$K(\tau)/K(0)$")
        axes[fila, 1].plot(r["w"], r["Sw"], color="tab:blue", lw=1.1)
        axes[fila, 1].axvline(r["wc"], color="0.4", ls="--", lw=0.8)
        axes[fila, 1].axvline(-r["wc"], color="0.4", ls="--", lw=0.8)
        axes[fila, 1].set_xlim(-FACTOR_FRECUENCIA * r["wc"], FACTOR_FRECUENCIA * r["wc"])
        axes[fila, 1].set_title(f"Espectro - n={orden}")
        axes[fila, 1].set_xlabel(r"$\omega$ [rad/s]")
        axes[fila, 1].set_ylabel(r"$S(\omega)$")
        for ax in axes[fila]:
            estilo_eje(ax)

    fig.suptitle(f"Funcion de covarianza y espectro - ordenes {etiqueta}; N={N}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    guardar_comparativa_en_cada_orden(
        fig, f"comparativas_{etiqueta}_covarianza_espectro_N{N}.png"
    )


grupos_orden = {
    "impares": [P for P in ordenes if P % 2 == 1],
    "pares": [P for P in ordenes if P % 2 == 0],
}

for N in valores_N:
    for etiqueta, grupo in grupos_orden.items():
        if not grupo:
            continue
        generar_comparativa_tiempo_error(N, grupo, etiqueta)
        generar_comparativa_funciones_frecuencia(N, grupo, etiqueta)
        generar_comparativa_covarianza_espectro(N, grupo, etiqueta)
    print(f"Comparativas por grupo de orden creadas para N={N}.")


# ============================================================
# 6. COMPARATIVAS POR ORDEN FIJO Y TODOS LOS N
# ============================================================
carpeta_graficas_por_orden = carpeta_raiz / "graficas_por_orden"
carpeta_graficas_por_orden.mkdir(parents=True, exist_ok=True)


def limites_error_comunes(registros):
    positivos = []
    maximo = 0.0
    for r in registros:
        for arr in (np.asarray(r["Var"]), np.asarray(r["Var_medios"])):
            ok = np.isfinite(arr)
            positivos.extend(arr[ok & (arr > 1e-12)].tolist())
            if ok.any():
                maximo = max(maximo, float(np.nanmax(arr[ok])))
    minimo = min(positivos) if positivos else 1e-12
    y_min = max(1e-12, 10.0 ** (np.floor(np.log10(minimo)) - 1.0))
    y_max = max(3.0 * maximo, 10.0 * y_min)
    return y_min, y_max


def graficar_por_orden_tiempo_error(orden):
    registros = [resultados_graficas[(N, orden)] for N in valores_N]
    fig, axes = plt.subplots(len(valores_N), 2,
                             figsize=(15, 4.0 * len(valores_N)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    yerr = limites_error_comunes(registros)

    for fila, r in enumerate(registros):
        N = r["N"]
        ax_b, ax_e = axes[fila]
        for j in range(r["B_t"].shape[0]):
            ax_b.plot(r["t"], r["B_t"][j], lw=0.9,
                      color=colores[j % len(colores)],
                      label=rf"$B_{{{j+1}}}(t)$" if fila == 0 else None)
        ax_b.set_title(rf"Funciones basicas - $n={orden}$, $N={N}$, "
                       rf"$\omega_c={r['wc']:.5g}$")
        ax_b.set_xlabel("Tiempo t [s]")
        ax_b.set_ylabel(r"$B_j(t)$")
        ax_b.grid(True, alpha=0.28)

        piso = 1e-12
        ax_e.semilogy(r["t"], np.maximum(r["Var"], piso), color="black", lw=1.1,
                      label=r"$Var(t)$")
        ax_e.semilogy(r["t_medios"], np.maximum(r["Var_medios"], piso), "o",
                      color="tab:red", ms=3.5, label="Puntos medios")
        for ti in r["T"]:
            ax_e.axvline(ti, color="0.7", linestyle=":", linewidth=0.55)
        mask = (r["t"] >= r["T"][0]) & (r["t"] <= r["T"][-1])
        if np.any(mask):
            vw, tw = r["Var"][mask], r["t"][mask]
            im = int(np.argmax(vw))
            ax_e.plot(tw[im], max(float(vw[im]), piso), "mp", ms=7,
                      label=f"Error maximo={r['max_error_real']:.2e}")
        imin = int(np.argmin(r["Var_medios"]))
        ax_e.plot(r["t_medios"][imin], max(float(r["Var_medios"][imin]), piso),
                  "bv", ms=6, label=f"Error minimo={float(r['Var_medios'][imin]):.2e}")
        ax_e.set_ylim(*yerr)
        ax_e.set_title(rf"Error de reconstruccion - $N={N}$, "
                       rf"$T_c$ teorico={r['Tc_teorico']:.4f} s")
        ax_e.set_xlabel("Tiempo t [s]")
        ax_e.set_ylabel(r"$Var(t)$")
        ax_e.grid(True, which="both", alpha=0.28)
        ax_e.legend(loc="best", fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Funciones basicas", loc="center right",
                   bbox_to_anchor=(1.0, 0.5), fontsize=7,
                   ncol=2 if len(handles) > 12 else 1)
    fig.suptitle(rf"Butterworth - orden fijo $n={orden}$: funciones basicas y error "
                 "para todos los valores de N", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    ruta = carpeta_graficas_por_orden / f"P{orden}_tiempo_funciones_basicas_error_todos_N.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparativa por orden guardada: {ruta}")


def graficar_por_orden_frecuencia(orden):
    registros = [resultados_graficas[(N, orden)] for N in valores_N]
    fig, axes = plt.subplots(len(valores_N), 3,
                             figsize=(20, 4.0 * len(valores_N)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for fila, r in enumerate(registros):
        N = r["N"]
        ax_mag, ax_fase, ax_gd = axes[fila]
        for j in range(r["modulo_B"].shape[0]):
            color = colores[j % len(colores)]
            ax_mag.plot(r["w"], r["modulo_B"][j], color=color, lw=0.85,
                        label=rf"$B_{{{j+1}}}$" if fila == 0 else None)
            ax_fase.plot(r["w"], r["fase_B"][j], color=color, lw=0.85)
            ax_gd.plot(r["w"], r["retardo_grupo"][j], color=color, lw=0.85)
        ax_mag.set_title(rf"Magnitud - $N={N}$, $\omega_c={r['wc']:.5g}$")
        ax_fase.set_title(rf"Fase - $N={N}$")
        ax_gd.set_title(rf"Retardo de grupo - $N={N}$")
        ax_mag.set_ylabel(r"$|B_j(\omega)|$")
        ax_fase.set_ylabel("Fase [rad]")
        ax_gd.set_ylabel(r"$\tau_g$ [s]")
        for ax in (ax_mag, ax_fase, ax_gd):
            ax.set_xlabel(r"Frecuencia $\omega$ [rad/s]")
            ax.grid(True, alpha=0.28)
        limites = limites_retardo(r["retardo_grupo"], delta_T, r["w"], r["wc"])
        if limites:
            ax_gd.set_ylim(*limites)
        for ax in (ax_mag, ax_fase, ax_gd):
            ax.set_xlim(-FACTOR_FRECUENCIA * r["wc"], FACTOR_FRECUENCIA * r["wc"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Indice j", loc="center right",
                   bbox_to_anchor=(1.0, 0.5), fontsize=7,
                   ncol=2 if len(handles) > 12 else 1)
    fig.suptitle(rf"Butterworth - orden fijo $n={orden}$: magnitud, fase y retardo de grupo "
                 "para todos los valores de N", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    ruta = carpeta_graficas_por_orden / f"P{orden}_frecuencia_magnitud_fase_retardo_todos_N.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparativa por orden guardada: {ruta}")


for orden in ordenes:
    graficar_por_orden_tiempo_error(orden)
    graficar_por_orden_frecuencia(orden)


# ============================================================
# 7. CSV DE VALIDACION
# ============================================================
csv_path = carpeta_raiz / "validacion_Tc.csv"
if registros_tc:
    with csv_path.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(registros_tc[0].keys()))
        escritor.writeheader()
        escritor.writerows(registros_tc)


# ============================================================
# 8. RESUMEN FINAL
# ============================================================
numero_casos = len(valores_N) * len(ordenes)
numero_figuras_individuales = numero_casos * 5
numero_figuras_filtro = len(ordenes) * 2
numero_comparativas_distintas = sum(1 for _ in valores_N for _, g in grupos_orden.items() if g) * 3
numero_comparativas_orden = 2 * len(ordenes)

print("\nProceso terminado.")
print(f"Ordenes procesados: {ordenes}")
print(f"Valores de N procesados: {valores_N}")
print(f"Carpeta raiz: {carpeta_raiz}")
print(f"CSV de validacion: {csv_path}")
print(f"Graficas REMC individuales: {numero_figuras_individuales} PNG")
print(f"Graficas del filtro (espectro/covarianza y polos): {numero_figuras_filtro} PNG")
print(f"Comparativas distintas por N y grupo: {numero_comparativas_distintas} PNG")
print(f"Comparativas por orden fijo: {numero_comparativas_orden} PNG")

if ordenes and all(P % 2 == 1 for P in ordenes):
    print("NOTA: todos los ordenes configurados son impares; por eso no se "
          "genera una comparativa de ordenes pares. Agregue ordenes pares a "
          "la lista 'ordenes' para habilitarla.")
