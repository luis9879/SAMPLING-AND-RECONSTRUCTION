#Programa automatizado para generar las figuras de la familia de filtros Chebyshev tipo II,
#Los valores de la frecuencia de corte se han determinado de forma concreta, es decir, son validos para un tiempo de covarianza unitario simpre que
#se cumplan las mallas de integración,las mallas se poueden ajustar para valores muy grandes para integrar ordenes impares, este programa toma como referencia a "CHEBYSHEV_MANUAL.py"
#Se automatizado haciendo uso de modelos de AI pero la estrcutura del programa cumple con los calculos de manual, simplemente en esta versión se automatiza para el almacenaamiento de las figuras
import numpy as np
from numpy.linalg import inv
import matplotlib
MOSTRAR = False                
if not MOSTRAR:
    matplotlib.use("Agg")       
import matplotlib.pyplot as plt
from pathlib import Path
# ============================================================
# PARÁMETROS
# ============================================================
N       = 9
deltaT  = 1
epsilon = 0.5                                    # amplitud del rizo
TI      = deltaT * np.arange(N)                  # tiempos de muestreo

ordenes  = [1, 2, 3, 4, 5, 6]
wc_tabla = [4.14,3.89,3.45,3.1, 3.09, 3.1]  # w_c [rad/s] para cada orden
WC       = dict(zip(ordenes, wc_tabla))

# Mallas de integración según el orden del filtro
W_MAX_GENERAL = 5.0
W_MAX_ORDEN_1 = 50.0
N_W = 1001
def obtener_malla_frecuencia(P):
    if P == 1:
        return np.linspace(-W_MAX_ORDEN_1, W_MAX_ORDEN_1, N_W)
    else:
        return np.linspace(-W_MAX_GENERAL, W_MAX_GENERAL, N_W)
# Ventana de tiempo para las funciones básicas y el error
t_fam = np.linspace(TI[0] - 5, TI[-1] + 5, 2000)

# Imágenes: 3 órdenes por imagen
grupos = {"ordenes_1_3_5": [1, 3, 5],
          "ordenes_2_4_6": [2, 4, 6]}

usar_Hz = False      # True -> eje en f [Hz]; False -> eje en w [rad/s]
gd_ylim = None       # p. ej. (-10, 20) para recortar los picos del retardo de grupo
err_escala     = 'log'   # 'log' (recomendada) o 'lineal'
err_normalizar = False   # True -> grafica sigma^2(t)/K(0)
ERR_PISO       = 1e-12   # por debajo de esto el valor es ruido numérico
# Carpeta de salida
try:
    BASE_DIR = Path(__file__).parent
except NameError:                       # ejecución interactiva
    BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "Chevyshev_resultados_deltaT_igual_a_1"
OUT_DIR.mkdir(exist_ok=True)

# Colores y etiquetas (índice j = 1..N)
_idx     = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
COLORES  = [plt.cm.tab20((_idx[j % 20] + 0.5) / 20) for j in range(N)]
labels_B = [fr'$B_{{{j+1}}}$' for j in range(N)]

# ============================================================
# ESPECTRO DE CHEBYSHEV TIPO II
# ============================================================
def ChebyshevII_espectro(wc, w, epsilon, P):
    """S(w) = e^2/(1+e^2),  e = epsilon * C_P(wc/|w|),  S(0)=1."""
    w_abs = np.abs(w)
    S  = np.ones_like(w_abs, dtype=float)
    nz = w_abs > 0
    x  = wc / w_abs[nz]
    cp = np.where(x <= 1,
                  np.cos(P * np.arccos(np.minimum(x, 1.0))),
                  np.cosh(P * np.arccosh(np.maximum(x, 1.0))))
    e2 = (epsilon * cp) ** 2
    S[nz] = e2 / (1 + e2)
    return S

# ============================================================
# CÁLCULO COMPLETO PARA UN ORDEN P
# ============================================================
def calcular_orden(P, wc):
    w0 = obtener_malla_frecuencia(P)
    S_w = ChebyshevII_espectro(wc, w0, epsilon, P)

    def K_vec(tau):
        """K(tau) = (1/2pi) * integral S(w) cos(w tau) dw, para un arreglo tau."""
        tau = np.asarray(tau, dtype=float)
        return (1 / (2 * np.pi)) * np.trapezoid(
            S_w[None, :] * np.cos(np.outer(tau.ravel(), w0)), x=w0, axis=1
        ).reshape(tau.shape)

    # Matriz de covarianza de las muestras y su inversa
    kx  = K_vec(TI[:, None] - TI[None, :])
    a   = inv(kx)
    cnd = np.linalg.cond(kx)

    # Tiempo: B_j(t) y error de reconstrucción
    Kmat = np.array([K_vec(t_fam - TI[i]) for i in range(N)])   # (N, n_t)
    B_t  = a.T @ Kmat
    K0   = K_vec(np.array([0.0]))[0]
    err  = K0 - np.sum(Kmat * (a @ Kmat), axis=0)

    # Frecuencia: B_k(w), magnitud y fase
    E    = np.exp(-1j * np.outer(TI, w0))                       # (N, n_w)
    H    = a.T @ E                                              # sum_i a[i,k] e^{-j w T_i}
    Bf   = S_w[None, :] * H
    mag  = np.abs(Bf)
    fase = np.array([np.unwrap(np.angle(Bf[k])) for k in range(N)])

    # Retardo de grupo analítico: Re{Hd conj(H)} / |H|^2
    Hd  = a.T @ (TI[:, None] * E)
    num = np.real(Hd * np.conj(H))
    den = np.abs(H) ** 2
    tol = 1e-10 * den.max(axis=1, keepdims=True)
    gd  = np.where(den > tol, num / np.where(den > tol, den, 1.0), np.nan)

    return dict(P=P, wc=wc, S_w=S_w, K0=K0, cond=cnd,
                B_t=B_t, err=err, mag=mag, fase=fase, gd=gd, w0=w0)

# ============================================================
# CÁLCULO PARA TODOS LOS ÓRDENES
# ============================================================
resultados = {}
for P in ordenes:
    print(f"Orden P={P}, wc={WC[P]} ...", flush=True)
    resultados[P] = calcular_orden(P, WC[P])
    r = resultados[P]
    print(f"   K(0) = {r['K0']:.4f}   cond(kx) = {r['cond']:.3e}", flush=True)

# ============================================================
# FIGURAS: 4 imágenes (tiempo y frecuencia por separado)
#   tiempo_ordenes_1_3_5     | tiempo_ordenes_2_4_6
#   frecuencia_ordenes_1_3_5 | frecuencia_ordenes_2_4_6
# ============================================================
xlab = 'f [Hz]' if usar_Hz else r'$\omega$ [rad/s]'
dt_lbl = str(deltaT).replace('.', '_')

def leyenda_y_guardar(fig, ax_ref, titulo_leyenda, archivo):
    h, l = ax_ref.get_legend_handles_labels()
    fig.legend(h, l, loc='center right', bbox_to_anchor=(1.0, 0.5),
               fontsize=10, ncol=1, title=titulo_leyenda)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(OUT_DIR / archivo, dpi=200, bbox_inches='tight')
    print(f"Guardado: {archivo}", flush=True)

for nombre, grupo in grupos.items():
    ords = ", ".join(str(P) for P in grupo)
    n = len(grupo)

    # ── Imagen de TIEMPO: B_j(t) | error de reconstrucción ──
    fig_t, axt = plt.subplots(n, 2, figsize=(15, 4.8 * n))
    for fila, P in enumerate(grupo):
        r = resultados[P]
        lbl = rf'$P={P}$, $\omega_c={r["wc"]}$'
        for j in range(N):
            axt[fila, 0].plot(t_fam, r['B_t'][j], linewidth=0.8, color=COLORES[j],
                              label=labels_B[j] if fila == 0 else None)
        err_plot = r['err'] / r['K0'] if err_normalizar else r['err']
        if err_escala == 'log':
            axt[fila, 1].semilogy(t_fam, np.maximum(err_plot, ERR_PISO), color='black', linewidth=1.2)
            axt[fila, 1].set_ylim(ERR_PISO, 3 * max(err_plot.max(), ERR_PISO))
        else:
            axt[fila, 1].plot(t_fam, err_plot, color='black', linewidth=1.5)
        for Ti in TI:                                   # instantes de muestreo
            axt[fila, 1].axvline(Ti, color='gray', linestyle=':', linewidth=0.6)
        axt[fila, 0].set_title(f'Funciones básicas — {lbl}', fontsize=11)
        axt[fila, 1].set_title(f'Error de reconstrucción — {lbl}', fontsize=11)
        axt[fila, 0].set_ylabel(r'$B_j(t)$')
        axt[fila, 1].set_ylabel(r'$\tilde{\sigma}^2(t)/K(0)$' if err_normalizar else r'$\tilde{\sigma}^2(t)$')
        for c in range(2):
            axt[fila, c].set_xlabel('t [s]')
            axt[fila, c].grid(True, alpha=0.3)
    fig_t.suptitle(rf'Estudio en Tiempo — Chebyshev II, órdenes {ords} '
                   rf'($\epsilon={epsilon}$, $\Delta T={deltaT}$, $N={N}$)',
                   fontsize=14, fontweight='bold')
    leyenda_y_guardar(fig_t, axt[0, 0], r'Índice $j$', f"tiempo_{nombre}_deltaT{dt_lbl}.png")

    # ── Imagen de FRECUENCIA: magnitud | fase | retardo de grupo ──
    fig_f, axf = plt.subplots(n, 3, figsize=(20, 4.8 * n))
    for fila, P in enumerate(grupo):
        r = resultados[P]
        w0_P = r['w0']
        x = w0_P / (2 * np.pi) if usar_Hz else w0_P
        lbl = rf'$P={P}$, $\omega_c={r["wc"]}$'
        
        for k in range(N):
            kw = dict(linewidth=0.8, color=COLORES[k])
            axf[fila, 0].plot(x, r['mag'][k],  label=labels_B[k] if fila == 0 else None, **kw)
            axf[fila, 1].plot(x, r['fase'][k], **kw)
            axf[fila, 2].plot(x, r['gd'][k],   **kw)
        axf[fila, 0].set_title(f'Magnitud — {lbl}', fontsize=11)
        axf[fila, 1].set_title(f'Fase — {lbl}', fontsize=11)
        axf[fila, 2].set_title(f'Retardo de grupo — {lbl}', fontsize=11)
        axf[fila, 0].set_ylabel(r'$|B_k(\omega)|$')
        axf[fila, 1].set_ylabel(r'$\angle B_k(\omega)$ [rad]')
        axf[fila, 2].set_ylabel(r'$\tau_{g,k}(\omega)$ [s]')
        if gd_ylim is not None:
            axf[fila, 2].set_ylim(gd_ylim)
        for c in range(3):
            axf[fila, c].set_xlabel(xlab)
            axf[fila, c].grid(True, alpha=0.3)
    fig_f.suptitle(rf'Magnitud, Fase y Retardo de Grupo — Chebyshev II, órdenes {ords} '
                   rf'($\epsilon={epsilon}$, $\Delta T={deltaT}$, $N={N}$)',
                   fontsize=14, fontweight='bold')
    leyenda_y_guardar(fig_f, axf[0, 0], r'Índice $k$', f"frecuencia_{nombre}_deltaT{dt_lbl}.png")
# ============================================================
# FUNCIONES BÁSICAS INDIVIDUALES: una carpeta por orden dentro de OUT_DIR
#   orden_P{P}_wc{wc}/
#       B01_tiempo.png      -> B_1(t)
#       B01_frecuencia.png  -> magnitud | fase | retardo de grupo de B_1(jw)
#       ...                    (una pareja de imágenes por cada B_j, j = 1..N)
# ============================================================
for P in ordenes:
    r = resultados[P]
    w0_P = r['w0']
    x = w0_P / (2 * np.pi) if usar_Hz else w0_P
    wc_lbl   = str(r['wc']).replace('.', '_')
    DIR_ORD  = OUT_DIR / f"orden_P{P}_wc{wc_lbl}"
    DIR_ORD.mkdir(parents=True, exist_ok=True)
    sub_lbl  = rf'Chebyshev II, $P={P}$, $\omega_c={r["wc"]}$, $\Delta T={deltaT}$, $N={N}$'

    for j in range(N):
        pref = f"B{j+1:02d}"

        # ── B_j(t) ──
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t_fam, r['B_t'][j], color=COLORES[j], linewidth=1.2)
        ax.set_title(rf'$B_{{{j+1}}}(t)$ — {sub_lbl}', fontsize=11)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(rf'$B_{{{j+1}}}(t)$')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(DIR_ORD / f"{pref}_tiempo.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        # ── B_k(jw): magnitud | fase | retardo de grupo ──
        fig, (axm, axp, axg) = plt.subplots(1, 3, figsize=(18, 4.5))
        axm.plot(x, r['mag'][j],  color=COLORES[j], linewidth=1.2)
        axp.plot(x, r['fase'][j], color=COLORES[j], linewidth=1.2)
        axg.plot(x, r['gd'][j],   color=COLORES[j], linewidth=1.2)
        axm.set_title('Magnitud', fontsize=11)
        axp.set_title('Fase', fontsize=11)
        axg.set_title('Retardo de grupo', fontsize=11)
        axm.set_ylabel(rf'$|B_{{{j+1}}}(\omega)|$')
        axp.set_ylabel(rf'$\angle B_{{{j+1}}}(\omega)$ [rad]')
        axg.set_ylabel(rf'$\tau_{{g,{j+1}}}(\omega)$ [s]')
        if gd_ylim is not None:
            axg.set_ylim(gd_ylim)
        elif np.nanmax(r['gd'][j]) - np.nanmin(r['gd'][j]) < 1e-3:   # retardo constante (fase lineal)
            axg.set_ylim(np.nanmedian(r['gd'][j]) - 1, np.nanmedian(r['gd'][j]) + 1)
        axg.ticklabel_format(useOffset=False)
        for axx in (axm, axp, axg):
            axx.set_xlabel(xlab)
            axx.grid(True, alpha=0.3)
        fig.suptitle(rf'$B_{{{j+1}}}(j\omega)$ — {sub_lbl}', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(DIR_ORD / f"{pref}_frecuencia.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"Carpeta {DIR_ORD.name}: {2*N} imágenes", flush=True)
if MOSTRAR:
    plt.show()
else:
    plt.close('all')