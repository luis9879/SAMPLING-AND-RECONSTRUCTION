#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funcion de covarianza del filtro Butterworth de orden n por el teorema de
residuos de Cauchy (metodologia de los polos S_k del plano s).

Dado n (y opcionalmente w_c) el programa:
  1. calcula los polos S_k del semiplano izquierdo (ecuacion 3.21),
  2. calcula la funcion de covarianza normalizada como suma de residuos:
         R(t) = -(sin(pi/2n)/w_c) * sum_k S_k * exp(S_k*|t|)
     (con w_c = n*sin(pi/2n) el prefactor vale 1/n y tau_c = 1),
  3. grafica en una figura: espectro S_n(w) (izquierda) y R(t) (derecha),
  4. grafica en otra figura los polos de H(s)H(-s) como en la figura 3.4.

Normalizacion: varianza sigma^2 = 1, entonces K(t) = R(t) y
    S_n(w) = 2*tau_c / (1 + (w/w_c)^(2n)),   tau_c = n*sin(pi/2n)/w_c.

Uso:
    python butterworth_covarianza.py 3
    python butterworth_covarianza.py 5 --wc 2.0
    python butterworth_covarianza.py 3 --save --check
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Calculo
# ----------------------------------------------------------------------
def wc_para_tauc_unitario(n):
    """Ecuacion (4): w_c = n sin(pi/2n) hace que tau_c = 1."""
    return n * np.sin(np.pi / (2 * n))


def tiempo_covarianza(n, wc):
    """tau_c = n sin(pi/2n) / w_c."""
    return n * np.sin(np.pi / (2 * n)) / wc


def polos_izquierdos(n, wc):
    """
    Polos S_k del semiplano izquierdo (ecuacion 3.21):
        S_k = w_c [ -sin(theta_k) + j cos(theta_k) ],
        theta_k = (2k-1) pi / (2n),  k = 1..n
    """
    k = np.arange(1, n + 1)
    theta = (2 * k - 1) * np.pi / (2 * n)
    return wc * (-np.sin(theta) + 1j * np.cos(theta)), theta


def polos_todos(n, wc):
    """Los 2n polos de H(s)H(-s): s = j z, z = w_c exp(j(2m-1)pi/2n), m=1..2n."""
    m = np.arange(1, 2 * n + 1)
    theta = (2 * m - 1) * np.pi / (2 * n)
    return 1j * wc * np.exp(1j * theta)


def espectro(w, n, wc, sigma2=1.0):
    """S_n(w) = 2 sigma^2 tau_c / (1 + (w/w_c)^(2n))."""
    w = np.asarray(w, dtype=float)
    with np.errstate(over="ignore"):
        return 2.0 * sigma2 * tiempo_covarianza(n, wc) / (1.0 + (w / wc) ** (2 * n))


def covarianza_residuos(tau, n, wc):
    """
    R(t) = -(sin(pi/2n)/w_c) * sum_k S_k exp(S_k |t|)

    Devuelve (R real, max |parte imaginaria| descartada).
    """
    tau = np.abs(np.atleast_1d(np.asarray(tau, dtype=float)))
    s, _ = polos_izquierdos(n, wc)
    coef = -np.sin(np.pi / (2 * n)) / wc
    suma = (coef * s[None, :] * np.exp(np.outer(tau, s))).sum(axis=1)
    return suma.real, np.max(np.abs(suma.imag))


def covarianza_numerica(tau, n, wc):
    """Verificacion: K(t) = (1/pi) int_0^inf S(w) cos(w t) dw (requiere scipy)."""
    from scipy.integrate import quad

    out = []
    for t in np.atleast_1d(tau):
        if t == 0:
            val, _ = quad(lambda w: espectro(w, n, wc), 0, np.inf, limit=500)
        else:
            val, _ = quad(lambda w: espectro(w, n, wc), 0, np.inf,
                          weight="cos", wvar=float(t))
        out.append(val / np.pi)
    return np.array(out)


# ----------------------------------------------------------------------
# Graficas
# ----------------------------------------------------------------------
def graficar_espectro_y_covarianza(n, wc, tmax=None, check=False):
    tc = tiempo_covarianza(n, wc)
    if tmax is None:
        # la envolvente mas lenta decae como exp(-wc sin(pi/2n) t)
        tmax = 6.0 / (wc * np.sin(np.pi / (2 * n)))

    w = np.linspace(-3 * wc, 3 * wc, 3000)
    tau = np.linspace(-tmax, tmax, 4001)
    R, _ = covarianza_residuos(tau, n, wc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Espectro (izquierda)
    ax1.plot(w, espectro(w, n, wc), color="tab:blue", lw=2)
    ax1.axvline(wc, color="gray", ls="--", lw=1)
    ax1.axvline(-wc, color="gray", ls="--", lw=1)
    ax1.plot([wc, -wc], [espectro(wc, n, wc)] * 2, "o", color="tab:red", ms=5,
             label=r"$\omega=\pm\omega_c$ (mitad de potencia)")
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$S_n(\omega)$")
    ax1.set_title(rf"Espectro de potencia, $n={n}$, $\omega_c={wc:.4f}$")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower center", fontsize=9)

    # Covarianza (derecha)
    ax2.plot(tau, R, color="tab:green", lw=2, label=r"$R_x(\tau)$ (residuos)")
    if check:
        try:
            tp = np.linspace(-tmax, tmax, 41)
            ax2.plot(tp, covarianza_numerica(tp, n, wc), "o", mfc="none",
                     color="k", ms=4, label="Integral numérica (verificación)")
        except ImportError:
            print("scipy no disponible: se omite la verificación en la gráfica.")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel(r"$R_x(\tau)$")
    ax2.set_title(rf"Función de covarianza, $\tau_c={tc:.4f}$")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def graficar_polos(n, wc):
    """Polos de H(s)H(-s) en el plano s, al estilo de la figura 3.4."""
    s_all = polos_todos(n, wc)
    izq = s_all[s_all.real < 0]
    der = s_all[s_all.real > 0]
    lim = 1.55 * wc

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ejes con flecha
    flecha = dict(arrowstyle="-|>", color="k", lw=1.2)
    ax.annotate("", xy=(lim, 0), xytext=(-lim, 0), arrowprops=flecha)
    ax.annotate("", xy=(0, lim), xytext=(0, -lim), arrowprops=flecha)
    ax.text(lim, -0.07 * lim, r"$\sigma$", ha="right", va="top", fontsize=14)
    ax.text(0.04 * lim, lim, r"$j\omega$", ha="left", va="top", fontsize=14)
    ax.text(0.97 * lim, 0.93 * lim, r"plano $s$", ha="right", va="top",
            style="italic", fontsize=12)

    # circunferencia de radio w_c
    ax.add_patch(plt.Circle((0, 0), wc, fill=False, ls="--", color="gray", lw=1.2))

    # lineas radiales hacia cada polo
    for p in s_all:
        ax.plot([0, p.real], [0, p.imag], ls=":", color="gray", lw=1)

    # polos
    ax.plot(izq.real, izq.imag, "x", color="tab:blue", ms=12, mew=2.5,
            label=r"Polos de $H(s)$ (semiplano izquierdo, $S_k$)")
    ax.plot(der.real, der.imag, "x", color="tab:red", ms=12, mew=2.5,
            label=r"Polos de $H(-s)$ (semiplano derecho)")

    # etiquetas S_k de los polos del semiplano izquierdo
    S_k, _ = polos_izquierdos(n, wc)
    for k, p in enumerate(S_k, start=1):
        r = abs(p)
        ux, uy = p.real / r, p.imag / r
        ha = "right" if ux < -0.15 else ("left" if ux > 0.15 else "center")
        va = "bottom" if uy >= -0.15 else "top"
        # el polo real (n impar) se etiqueta encima del eje
        if abs(p.imag) < 1e-9:
            ha, va, uy = "center", "bottom", 0.45
        ax.text(p.real + 0.10 * wc * ux, p.imag + 0.10 * wc * uy,
                rf"$S_{k}$", color="tab:blue", fontsize=12, ha=ha, va=va)

    # radio etiquetado w_c hacia un polo del semiplano derecho (como en la figura)
    cand = der[der.imag >= -1e-9]
    if cand.size == 0:
        cand = der
    p_ref = cand[np.argmax(cand.imag)]
    ax.plot([0, p_ref.real], [0, p_ref.imag], ls="--", color="k", lw=1.2)
    ang = np.angle(p_ref)
    ax.text(0.5 * p_ref.real + 0.05 * wc * np.sin(ang),
            0.5 * p_ref.imag - 0.05 * wc * np.cos(ang),
            rf"$\omega_c={wc:.4g}$", fontsize=11, rotation=np.degrees(ang),
            rotation_mode="anchor", ha="center", va="top")

    # marcas en +-w_c sobre el eje real (en lugar de los ticks del marco)
    ax.set_xticks([])
    ax.set_yticks([])
    for x0, txt in ((-wc, r"$-\omega_c$"), (wc, r"$\omega_c$")):
        ax.plot([x0, x0], [-0.025 * lim, 0.025 * lim], color="k", lw=1.2)
        ax.text(x0, -0.05 * lim, txt, ha="center", va="top", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=9,
              frameon=False)
    ax.set_title(rf"Polos de $H(s)H(-s)$ para $N={n}$ y $\omega_c={wc:.4g}$",
                 fontsize=12, pad=12)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Reporte por consola
# ----------------------------------------------------------------------
def reporte(n, wc, check=False):
    tc = tiempo_covarianza(n, wc)
    S_k, theta = polos_izquierdos(n, wc)
    coef = -np.sin(np.pi / (2 * n)) / wc * S_k  # coeficiente de exp(S_k t)

    print(f"\nFiltro Butterworth de orden n = {n}")
    print(f"  w_c   = {wc:.6f}   (n sin(pi/2n) = {wc_para_tauc_unitario(n):.6f})")
    print(f"  tau_c = n sin(pi/2n)/w_c = {tc:.6f}")
    print("\n  k   theta_k (grados)        S_k                    coef. de exp(S_k|t|)")
    for k in range(n):
        print(f"  {k+1:<3d} {np.degrees(theta[k]):>10.3f}   "
              f"{S_k[k].real:>9.5f}{S_k[k].imag:+.5f}j   "
              f"{coef[k].real:>9.5f}{coef[k].imag:+.5f}j")

    R0, imag_res = covarianza_residuos(0.0, n, wc)
    tc_pol = float(np.real(np.sum(-coef / S_k)))  # int_0^inf e^{S_k t} dt = -1/S_k
    print(f"\n  R(0)                         = {R0[0]:.10f}   (debe ser 1)")
    print(f"  tau_c = int_0^inf R dt        = {tc_pol:.10f}   (suma de -c_k/S_k)")
    print(f"  max |Im| descartada en la suma = {imag_res:.2e}   (debe ser ~0)")

    if check:
        try:
            tp = np.linspace(0, 6.0 / (wc * np.sin(np.pi / (2 * n))), 25)
            err = np.max(np.abs(covarianza_residuos(tp, n, wc)[0]
                                - covarianza_numerica(tp, n, wc)))
            print(f"  max |R_residuos - R_numerica|  = {err:.2e}")
        except ImportError:
            print("  (scipy no disponible: verificación numérica omitida)")


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Covarianza del filtro Butterworth por residuos (polos S_k).")
    ap.add_argument("n", type=int, nargs="?", default=10,
                    help="orden del filtro (entero >= 1), por defecto 3")
    ap.add_argument("--wc", type=float, default=None,
                    help="frecuencia de corte w_c. Por defecto n sin(pi/2n) (tau_c = 1)")
    ap.add_argument("--tmax", type=float, default=None,
                    help="intervalo |tau| <= tmax de la gráfica de covarianza")
    ap.add_argument("--check", action="store_true",
                    help="verifica contra la integral numérica de Fourier (requiere scipy)")
    ap.add_argument("--save", action="store_true",
                    help="guarda las figuras como PNG")
    ap.add_argument("--no-show", action="store_true",
                    help="no abre ventanas de gráficas")
    args = ap.parse_args()

    if args.n < 1:
        ap.error("n debe ser un entero >= 1")
    n = args.n
    wc = args.wc if args.wc is not None else wc_para_tauc_unitario(n)
    if wc <= 0:
        ap.error("w_c debe ser positiva")

    reporte(n, wc, check=args.check)

    fig1 = graficar_espectro_y_covarianza(n, wc, tmax=args.tmax, check=args.check)
    fig2 = graficar_polos(n, wc)

    if args.save:
        f1 = f"butterworth_n{n}_espectro_covarianza.png"
        f2 = f"butterworth_n{n}_polos.png"
        fig1.savefig(f1, dpi=150)
        fig2.savefig(f2, dpi=150, bbox_inches="tight")
        print(f"\n  Figuras guardadas: {f1}, {f2}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
