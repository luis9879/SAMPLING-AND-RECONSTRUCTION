# Estudio dual en el tiempo y en la frecuencia para los 3 citcuitos RC. Las gráficas se guardan para cada filtro y el dominio correspondiente.
import os
import numpy as np
from numpy.linalg import inv
from fractions import Fraction
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# PARAMETROS GENERALES
tc_values = [1,4/5, 3/5, 2/5]          # tiempos de covarianza a comparar
N = 11                                # numero de muestras
deltaT = 0.3                          # separacion uniforme entre muestras
TI = np.array([deltaT * i for i in range(1, N + 1)])   # iniciando en 0.3

colors = ['black', 'orange', 'red']
line_styles = ['-', '--', '-.','---']

DPI = 400                           
FIGSIZE_T = (10, 6)
FIGSIZE_F = (13, 6)

# Eje de tiempo para graficar B_k(t)
t0 = np.linspace(0, TI[-1] + deltaT, 1500)

# Eje de frecuencia para graficar B_k(jw)
f0 = np.linspace(-10, 10, 4000)
w0 = 2 * np.pi * f0

# Carpeta raiz de resultados
BASE_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "Resultados_RC_Retardo_grupo")

# ----------------------------------------------------------------------------------
# DEFINICION DE LAS 3 CONFIGURACIONES: covarianza, espectro y nombres de carpeta
# ----------------------------------------------------------------------------------
def R_sal(tau, alpha, etapas):
    """Funcion de covarianza normalizada segun el numero de etapas."""
    a_tau = np.abs(tau)
    if etapas == 1:
        # (4.15)
        return np.exp(-alpha * a_tau)
    elif etapas == 2:
        # (4.19)
        return (1 + alpha * a_tau) * np.exp(-alpha * a_tau)
    elif etapas == 3:
        # (4.23)
        return (1 + alpha * a_tau + (alpha ** 2 * tau ** 2) / 3) * np.exp(-alpha * a_tau)
    else:
        raise ValueError("etapas debe ser 1, 2 o 3")


def S_sal(w, alpha, etapas):
    """Densidad espectral de potencia segun el numero de etapas."""
    if etapas == 1:
        # (4.17)
        return (2 * alpha) / (w ** 2 + alpha ** 2)
    elif etapas == 2:
        # (4.20)
        return (4 * alpha ** 3) / (w ** 2 + alpha ** 2) ** 2
    elif etapas == 3:
        # (4.24) corregida: 16 en lugar de 4
        return (16 * alpha ** 5) / (w ** 2 + alpha ** 2) ** 3
    else:
        raise ValueError("etapas debe ser 1, 2 o 3")


CONFIGS = [
    {"etapas": 1, "carpeta": "RC_una_etapa",   "titulo": "Circuito RC de una etapa"},
    {"etapas": 2, "carpeta": "RC_dos_etapas",  "titulo": "Circuito RC de dos etapas"},
    {"etapas": 3, "carpeta": "RC_tres_etapas", "titulo": "Circuito RC de tres etapas"},
]

print("INICIO")
print("Tiempos de muestreo:", TI)

# PROCESO PRINCIPAL
for cfg in CONFIGS:
    etapas = cfg["etapas"]
    carpeta_filtro = os.path.join(BASE_DIR, cfg["carpeta"])
    carpeta_tiempo = os.path.join(carpeta_filtro, "tiempo")
    carpeta_frecuencia = os.path.join(carpeta_filtro, "frecuencia")
    os.makedirs(carpeta_tiempo, exist_ok=True)
    os.makedirs(carpeta_frecuencia, exist_ok=True)

    print(f"\n=== {cfg['titulo']} ===")

    # defininicion de alfa con base al tiempo de covarainz y la etapas del analisis
    def alpha_desde_tc(tc, etapas):
        if etapas == 1:
            return 1 / tc
        elif etapas == 2:
            return 2 / tc
        elif etapas == 3:
            return 8 / (3 * tc)
        else:
            raise ValueError("etapas debe ser 1, 2 o 3")

    datos_tc = []
    for tc in tc_values:
        alpha = alpha_desde_tc(tc, etapas)

        # Matriz de covarianza (a partir de la R_sal correspondiente)
        kx = np.zeros((N, N))
        for i in range(N):
            for k in range(N):
                kx[i, k] = R_sal(TI[i] - TI[k], alpha, etapas)
        a = inv(kx)

        sx = S_sal(w0, alpha, etapas)  # espectro evaluado en w0 

        print(f"tc={tc}  alpha={alpha:.4f}")
        print("Matriz de covarianza:")
        print(kx)
        print("Matriz inversa:")
        print(a)

        datos_tc.append({"tc": tc, "alpha": alpha, "a": a, "sx": sx})

    tc_fraccion = [str(Fraction(tc).limit_denominator()) for tc in tc_values]

    # ------------------------------------------------------------------------------
    # Una figura por cada funcion basica B_k, tanto en tiempo como en frecuencia
    # ------------------------------------------------------------------------------
    for j in range(N):
        # ---------------------- DOMINIO DEL TIEMPO ----------------------
        fig_t, ax_t = plt.subplots(figsize=FIGSIZE_T)
        for idx, d in enumerate(datos_tc):
            alpha, a = d["alpha"], d["a"]
            Bt = np.zeros_like(t0)
            for i in range(N):
                Bt += R_sal(t0 - TI[i], alpha, etapas) * a[i, j]
            alpha_fraccion = str(Fraction(alpha).limit_denominator())
            ax_t.plot(t0, Bt, color=colors[idx], linestyle=line_styles[idx],
                      linewidth=1.5, label=f'tc={tc_fraccion[idx]}   α={alpha_fraccion}')

        ax_t.set_title(f"Función básica B_{j + 1}(t) - {cfg['titulo']}")
        ax_t.set_xlabel("t [s]")
        ax_t.set_ylabel(f"B_{j + 1}(t)")
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(loc='best')
        fig_t.tight_layout()
        ruta_t = os.path.join(carpeta_tiempo, f"B_{j + 1}_tiempo.png")
        fig_t.savefig(ruta_t, dpi=DPI)
        plt.close(fig_t)

        # ---------------------- DOMINIO DE LA FRECUENCIA ----------------------
        fig_f, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_F)
        for idx, d in enumerate(datos_tc):
            alpha, a, sx = d["alpha"], d["a"], d["sx"]
            Bj = sx * sum(a[i, j] * np.exp(-1j * w0 * TI[i]) for i in range(N))
            alpha_fraccion = str(Fraction(alpha).limit_denominator())

            ax1.plot(w0, np.abs(Bj), color=colors[idx], linestyle=line_styles[idx],
                     linewidth=1.5, label=f'tc={tc_fraccion[idx]}   α={alpha_fraccion}')
            ax2.plot(w0, np.unwrap(np.angle(Bj)), color=colors[idx], linestyle=line_styles[idx],
                     linewidth=1.5, label=f'tc={tc_fraccion[idx]}')

        ax1.set_title("Magnitud")
        ax1.set_xlabel("w [rad/s]")
        ax1.set_ylabel(f"|B_{j + 1}(jw)|")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        ax2.set_title("Fase")
        ax2.set_xlabel("w [rad/s]")
        ax2.set_ylabel("fase [rad]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        fig_f.suptitle(f"Función básica B_{j + 1}(jw) - {cfg['titulo']}")
        fig_f.tight_layout()
        ruta_f = os.path.join(carpeta_frecuencia, f"B_{j + 1}_frecuencia.png")
        fig_f.savefig(ruta_f, dpi=DPI)
        plt.close(fig_f)

    print(f"Guardado en: {carpeta_filtro}")

print("\ngraficas guardadas en", BASE_DIR)