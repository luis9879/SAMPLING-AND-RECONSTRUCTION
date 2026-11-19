#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funcion de covarianza del filtro Butterworth de orden n
mediante el teorema de residuos de Cauchy.
Se considera Tc = 1, utilizando las formulas correspondientes
al filtro Butterworth del maestro Lozada.
Graficas:
1. Densidad espectral de potencia y funcion de covarianza.
2. Polos de H(s)H(-s) en el plano complejo s.
"""
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------
# 1. Definiciones y calculos
# ----------------------------------------------------------------------
n = 10  # Orden del filtro Butterworth
# Frecuencia de corte para Tc = 1
w_c = n * np.sin(np.pi / (2 * n))
# Angulos de los polos
theta = np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n)
# Polos del semiplano izquierdo
s_k = w_c * (-np.sin(theta) + 1j * np.cos(theta))
# Tiempo de covarianza
T_c = n * np.sin(np.pi / (2 * n)) / w_c
# ----------------------------------------------------------------------
# 2. Calculo de la funcion de covarianza
# ----------------------------------------------------------------------
# Vector de tiempos
tmax = 6.0 / (w_c * np.sin(np.pi / (2 * n)))
tao = np.linspace(-tmax, tmax, 4001)
# Funcion de covarianza mediante residuos
R_x = (-1 / n) * np.sum(
    s_k[:, np.newaxis] *
    np.exp(s_k[:, np.newaxis] * np.abs(tao)),
    axis=0
)
# Se conserva la parte real de la suma
R_x = R_x.real
# ----------------------------------------------------------------------
# 3. Calculo de la densidad espectral de potencia
# ----------------------------------------------------------------------
# Vector de frecuencias
w = np.linspace(-3 * w_c, 3 * w_c, 3000)
# Densidad espectral bilateral para Tc = 1
S_x = 2.0 / (1.0 + (w / w_c) ** (2 * n))
# ----------------------------------------------------------------------
# 4. Primera figura: espectro y funcion de covarianza
# ----------------------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
# ---- Grafica de densidad espectral de potencia ----
ax1.plot(w, S_x, color="tab:blue", lw=2)
# Frecuencias de corte
ax1.axvline(w_c, color="gray", ls="--", lw=1)
ax1.axvline(-w_c, color="gray", ls="--", lw=1)
# Puntos correspondientes a la mitad de potencia
ax1.plot(
    [w_c, -w_c],
    [1, 1],
    "o",
    color="tab:red",
    ms=5,
    label=r"$\omega=\pm\omega_c$ (mitad de potencia)"
)
ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel(r"$S_x(\omega)$")
ax1.set_title(
    rf"Espectro de potencia, $n={n}$, $\omega_c={w_c:.4f}$"
)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower center", fontsize=9)
# ---- Grafica de la funcion de covarianza ----
ax2.plot(
    tao,
    R_x,
    color="tab:green",
    lw=2,
    label=r"$R_x(\tau)$ (residuos)"
)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xlabel(r"$\tau$")
ax2.set_ylabel(r"$R_x(\tau)$")
ax2.set_title(rf"Funcion de covarianza, $T_c={T_c:.4f}$")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper right", fontsize=9)
fig1.tight_layout()
# ----------------------------------------------------------------------
# 5. Calculo de los 2n polos de H(s)H(-s)
# ----------------------------------------------------------------------
m = np.arange(1, 2 * n + 1)
theta_all = (2 * m - 1) * np.pi / (2 * n)
# Polos en el plano complejo s
s_all = 1j * w_c * np.exp(1j * theta_all)
# Separacion de polos por semiplano
izq = s_all[s_all.real < 0]
der = s_all[s_all.real > 0]
# ----------------------------------------------------------------------
# 6. Segunda figura: diagrama de polos
# ----------------------------------------------------------------------
lim = 1.55 * w_c
fig2, ax = plt.subplots(figsize=(6.6, 6.6))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
# Ocultar los bordes de la grafica
for sp in ax.spines.values():
    sp.set_visible(False)
# Ejes con flechas
flecha = dict(arrowstyle="-|>", color="k", lw=1.2)
ax.annotate(
    "",
    xy=(lim, 0),
    xytext=(-lim, 0),
    arrowprops=flecha
)
ax.annotate(
    "",
    xy=(0, lim),
    xytext=(0, -lim),
    arrowprops=flecha
)
# Etiquetas de los ejes
ax.text(
    lim, -0.07 * lim, r"$\sigma$",
    ha="right", va="top", fontsize=14
)
ax.text(
    0.04 * lim, lim, r"$j\omega$",
    ha="left", va="top", fontsize=14
)
ax.text(
    0.97 * lim, 0.93 * lim, r"plano $s$",
    ha="right", va="top", style="italic", fontsize=12
)
# Circunferencia de radio wc
ax.add_patch(
    plt.Circle(
        (0, 0),
        w_c,
        fill=False,
        ls="--",
        color="gray",
        lw=1.2
    )
)
# Lineas radiales hacia cada polo
for p in s_all:
    ax.plot(
        [0, p.real],
        [0, p.imag],
        ls=":",
        color="gray",
        lw=1
    )
# Polos del semiplano izquierdo: H(s)
ax.plot(
    izq.real,
    izq.imag,
    "x",
    color="tab:blue",
    ms=12,
    mew=2.5,
    label=r"Polos de $H(s)$ (semiplano izquierdo, $S_k$)"
)
# Polos del semiplano derecho: H(-s)
ax.plot(
    der.real,
    der.imag,
    "x",
    color="tab:red",
    ms=12,
    mew=2.5,
    label=r"Polos de $H(-s)$ (semiplano derecho)"
)
# Etiquetas de los polos S_k
for k, p in enumerate(s_k, start=1):
    r = abs(p)
    ux = p.real / r
    uy = p.imag / r
    ha = "right" if ux < -0.15 else (
        "left" if ux > 0.15 else "center"
    )
    va = "bottom" if uy >= -0.15 else "top"
    # Tratamiento del polo real
    if abs(p.imag) < 1e-9:
        ha, va, uy = "center", "bottom", 0.45
    ax.text(
        p.real + 0.10 * w_c * ux,
        p.imag + 0.10 * w_c * uy,
        rf"$S_{k}$",
        color="tab:blue",
        fontsize=12,
        ha=ha,
        va=va
    )
# Radio de referencia wc
cand = der[der.imag >= -1e-9]
if cand.size == 0:
    cand = der
p_ref = cand[np.argmax(cand.imag)]
ax.plot(
    [0, p_ref.real],
    [0, p_ref.imag],
    ls="--",
    color="k",
    lw=1.2
)
ang = np.angle(p_ref)
ax.text(
    0.5 * p_ref.real + 0.05 * w_c * np.sin(ang),
    0.5 * p_ref.imag - 0.05 * w_c * np.cos(ang),
    rf"$\omega_c={w_c:.4g}$",
    fontsize=11,
    rotation=np.degrees(ang),
    rotation_mode="anchor",
    ha="center",
    va="top"
)
# Marcas de frecuencia de corte sobre el eje real
ax.set_xticks([])
ax.set_yticks([])
for x0, txt in (
    (-w_c, r"$-\omega_c$"),
    (w_c, r"$\omega_c$")
):
    ax.plot(
        [x0, x0],
        [-0.025 * lim, 0.025 * lim],
        color="k",
        lw=1.2
    )
    ax.text(
        x0,
        -0.05 * lim,
        txt,
        ha="center",
        va="top",
        fontsize=11
    )
# Leyenda y titulo
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.12),
    fontsize=9,
    frameon=False
)
ax.set_title(
    rf"Polos de $H(s)H(-s)$ para $N={n}$ "
    rf"y $\omega_c={w_c:.4g}$",
    fontsize=12,
    pad=12
)
fig2.tight_layout()
# ----------------------------------------------------------------------
# 7. Mostrar resultados numericos
# ----------------------------------------------------------------------
print("\nFiltro Butterworth")
print("---------------------------")
print(f"Orden del filtro: n = {n}")
print(f"Frecuencia de corte: wc = {w_c:.6f}")
print(f"Tiempo de covarianza: Tc = {T_c:.6f}")
print(f"Valor de R(0): R(0) = {R_x[len(tao)//2]:.10f}")
print("\nPolos del semiplano izquierdo:")
for k, p in enumerate(s_k, start=1):
    print(f"S_{k} = {p.real:.5f} {p.imag:+.5f}j")
print("\nNumero total de polos:", len(s_all))
# ----------------------------------------------------------------------
# 8. Mostrar las figuras
# ----------------------------------------------------------------------
plt.show()