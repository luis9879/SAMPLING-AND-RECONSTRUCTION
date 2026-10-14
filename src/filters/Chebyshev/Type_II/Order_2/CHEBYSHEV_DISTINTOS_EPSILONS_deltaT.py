#En el siguiente programa se hacen los calculos y gráficas de las funciones básicas en el tiempo y en la frecuencia. En el tiempo se ha gráficado el error de reconstrucción
#  y los puntos medios de cada intervalo de muestreo, en la frecuencia se ha graficado la magnitud, fase y retardo de grupo. 
#En la frecuencia se ha gráicado el espectro del filtro para cada orden; del 1 al 6, además, en las funciones básicas en la frecuencia se gráfico la magnitud, fase y retardo de grupo de cada función básica.
#Se ha considerado una familia de valores de epsilon y deltaT, para cada combinación se han generado las gráficas correspondientes.
#Para los valores de epsilon que es el rizado en la banda de rechazo se ha considerado los siguientes casos:epsilons = [2, 1, 0.5, 0.1]
#Para los valores de deltaT que es el periodo de muestreo se ha considerado los siguientes casos: deltaTs = [0.5, 1, 2]
#De esta forma se han considerado todos los parámetros en el filtro para caracterizar la dependencia estadística entre el orden del filtro, las condiciones de muestreo y el rizado en la banda de rechazo.
#Con la caracterización anterior se pueden hacer diseños que no eran considerados en la teoria de filtrado estocastico para la optimización de recursos
#y la reducción de la complejidad computacional en el diseño de filtros de Chebyshev tipo II.
#La referencia y estructura principal del programa se tomo del programa "CHEBYSHEV_MANUAL.py" que se encuentra en el repositorio de GitHub:
# https://github.com/luis9879/SAMPLING-AND-RECONSTRUCTION/blob/main/src/filters/Chebyshev/Type_II/Order_2/CHEBYSHEV_MANUAL.py
import numpy as np
from numpy.linalg import inv
import matplotlib
MOSTRAR = False # True -> abre ventanas al final (backend interactivo)
if not MOSTRAR:
    matplotlib.use("Agg") # solo guarda PNG: no crea ventanas de Tk
import matplotlib.pyplot as plt
from pathlib import Path
# ============================================================
# PARÁMETROS
# ============================================================
N = 9
epsilons = [2, 1, 0.5, 0.1] # amplitudes del rizo a barrer
deltaTs = [0.5, 1, 2] # periodos de muestreo a barrer [s]
ordenes = [1, 2, 3, 4, 5, 6]
# ------------------------------------------------------------
# w_c [rad/s] calibradas numéricamente para K_malla(0) = 1.
# Se sigue el mismo procedimiento que el programa de referencia:
#   P = 1: malla [-50, 50] rad/s con 1001 puntos.
#   Demás casos: por defecto [-5, 5] rad/s con 1001 puntos.
#   Excepción validada: epsilon=0.1, P=2 usa [-10, 10] rad/s con
#   2001 puntos para mantener delta_omega=0.01 rad/s.
# K(tau) se calcula por integración trapezoidal sobre la malla asignada.
# Si existen varias raíces, se conserva la de mayor w_c.
# Para recalcular la tabla: RECALIBRAR_WC = True.
# ------------------------------------------------------------
RECALIBRAR_WC = False
WC_TABLA = {
    2: {1: 1.0268347449, 2: 2.1929807174, 3: 0.6241215065, 4: 1.7822385805, 5: 0.4050738887, 6: 1.3096152569},
    1: {1: 2.0536694897, 2: 3.2021565580, 3: 2.2081978303, 4: 2.3339926314, 5: 2.6787404223, 6: 2.1979161521},
    0.5: {1: 4.1073389795, 2: 3.8569248578, 3: 3.4369869445, 4: 3.0919086511, 5: 3.0809305700, 6: 3.0952533328},
    0.1: {1: 20.5366948973, 2: 6.9611781191, 3: 4.8130217920, 4: 4.0804093242, 5: 3.7367566677, 6: 3.5551715079},
}
# Mallas de integración. Se conserva la malla original para los demás casos;
# solo se amplía epsilon=0.1, P=2, que tenía wc fuera de [-5, 5].
W_MAX_GENERAL = 5.0
W_MAX_ORDEN_1 = 50.0
W_MAX_P2_EPS01 = 10.0
N_W = 10001
N_W_P2_EPS01 = 2001 # conserva delta_omega = 0.01 rad/s
def obtener_malla_frecuencia(P, epsilon=None):
    if P == 1:
        return np.linspace(-W_MAX_ORDEN_1, W_MAX_ORDEN_1, N_W)
    if P == 2 and epsilon is not None and np.isclose(epsilon, 0.1):
        return np.linspace(-W_MAX_P2_EPS01, W_MAX_P2_EPS01, N_W_P2_EPS01)
    return np.linspace(-W_MAX_GENERAL, W_MAX_GENERAL, N_W)
# Ventana de retardos tau para graficar la función de covarianza K(tau)
TAU_MAX = 5.0
N_TAU = 1001
tau_cov = np.linspace(-TAU_MAX, TAU_MAX, N_TAU)
# Imágenes: 3 órdenes por imagen
grupos = {"ordenes_1_3_5": [1, 3, 5],
          "ordenes_2_4_6": [2, 4, 6]}
usar_Hz = False # True -> eje en f [Hz]; False -> eje en w [rad/s]
gd_ylim = None # p. ej. (-10, 20) para recortar los picos del retardo de grupo
err_escala = 'log' # 'log' (recomendada) o 'lineal'
err_normalizar = False # True -> grafica sigma^2(t)/K(0)
ERR_PISO = 1e-12 # por debajo de esto el valor es ruido numérico
ERR_TOL_MEDIO = 0.01 # si el máx. real en la ventana excede en más de este % relativo al de los
                     # puntos medios, la gráfica muestra también el máx. real
COND_ALERTA = 1e12 # cond(kx) por encima de esto: la inversión es poco confiable
# Carpeta de salida (raíz)
try:
    BASE_DIR = Path(__file__).parent
except NameError: # ejecución interactiva
    BASE_DIR = Path.cwd()
OUT_ROOT = BASE_DIR / "Chevyshev_resultados"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
# Colores y etiquetas (índice j = 1..N)
_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
COLORES = [plt.cm.tab20((_idx[j % 20] + 0.5) / 20) for j in range(N)]
labels_B = [fr'$B_{{{j+1}}}$' for j in range(N)]
xlab = 'f [Hz]' if usar_Hz else r'$\omega$ [rad/s]'
def etiqueta(valor):
    """0.5 -> '0_5' (apto para nombres de carpeta/archivo)."""
    return str(valor).replace('.', '_')
def muestreo(deltaT):
    """Instantes de muestreo TI y ventana de tiempo t_fam (funciones básicas y error)."""
    TI = deltaT * np.arange(N)
    t_fam = np.linspace(TI[0] - 5, TI[-1] + 5, 2000)
    return TI, t_fam
# ============================================================
# ESPECTRO DE CHEBYSHEV TIPO II
# ============================================================
def ChebyshevII_espectro(wc, w, epsilon, P):
    """S(w) = e^2/(1+e^2),  e = epsilon * C_P(wc/|w|),  S(0)=1."""
    w_abs = np.abs(w)
    S = np.ones_like(w_abs, dtype=float)
    nz = w_abs > 0
    x = wc / w_abs[nz]
    cp = np.where(x <= 1,
                  np.cos(P * np.arccos(np.minimum(x, 1.0))),
                  np.cosh(P * np.arccosh(np.maximum(x, 1.0))))
    e2 = (epsilon * cp) ** 2
    S[nz] = e2 / (1 + e2)
    return S
# ============================================================
# Integración numérica en la malla finita para todos los órdenes.
def modo_integracion(P):
    """Todos los órdenes usan integración trapezoidal en la malla asignada a P."""
    return 'malla'
# ============================================================
# CALIBRACIÓN NUMÉRICA DE w_c: K_malla(0) = 1
# ============================================================
def K0_malla(wc, epsilon, P):
    w = obtener_malla_frecuencia(P, epsilon)
    return np.trapezoid(ChebyshevII_espectro(wc, w, epsilon, P), x=w) / (2 * np.pi)
def calibrar_wc(epsilon, P):
    """Busca w_c para que la integral trapezoidal en la malla usada dé K_malla(0)=1."""
    W = obtener_malla_frecuencia(P, epsilon)[-1]
    ws = np.linspace(0.01, 3 * W, 4000)
    f = np.array([K0_malla(w, epsilon, P) - 1 for w in ws])
    idx = np.where(f[:-1] * f[1:] < 0)[0]
    if len(idx) == 0:
        print(f"   AVISO: K(0)=1 no tiene solución en la malla (eps={epsilon}, P={P}); "
              f"se usa el w_c con K(0) más cercano a 1", flush=True)
        return ws[np.argmin(np.abs(f))], 'malla'
    a, b = ws[idx[-1]], ws[idx[-1] + 1] # raíz de mayor w_c
    fa = K0_malla(a, epsilon, P) - 1
    for _ in range(100):
        m = 0.5 * (a + b)
        fm = K0_malla(m, epsilon, P) - 1
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b), 'malla'
if RECALIBRAR_WC:
    WC_TABLA = {e: {P: round(float(calibrar_wc(e, P)[0]), 10) for P in ordenes} for e in epsilons}
    print("WC_TABLA = {")
    for e in epsilons:
        print(f"    {e}: {WC_TABLA[e]},")
    print("}", flush=True)
# ============================================================
# CÁLCULO COMPLETO PARA UN ORDEN P Y UN epsilon
# ============================================================
def calcular_orden(P, wc, epsilon, TI, t_fam):
    # w0: eje de frecuencias para S(w), B_k(w) y la integración numérica de K(tau)
    w0 = obtener_malla_frecuencia(P, epsilon)
    S_w = ChebyshevII_espectro(wc, w0, epsilon, P)
    modo = modo_integracion(P)
    def K_vec(tau):
        """K_malla(tau) = (1/2pi) integral en la malla finita S(w) cos(w*tau) dw."""
        tau = np.asarray(tau, dtype=float)
        return (1 / (2 * np.pi)) * np.trapezoid(
            S_w[None, :] * np.cos(np.outer(tau.ravel(), w0)), x=w0, axis=1
        ).reshape(tau.shape)
    # Función de covarianza para graficar (como en el programa de un solo orden)
    K_cov = K_vec(tau_cov)
    # Matriz de covarianza de las muestras y su inversa
    kx = K_vec(TI[:, None] - TI[None, :])
    a = inv(kx)
    cnd = np.linalg.cond(kx)
    # Tiempo: B_j(t) y error de reconstrucción
    Kmat = np.array([K_vec(t_fam - TI[i]) for i in range(N)]) # (N, n_t)
    B_t = a.T @ Kmat
    K0 = K_vec(np.array([0.0]))[0]
    err = K0 - np.sum(Kmat * (a @ Kmat), axis=0)
    # Error en el punto medio de cada intervalo de muestreo (ahí se toma el error máximo)
    t_mid = 0.5 * (TI[:-1] + TI[1:]) # (N-1,)
    Km = np.array([K_vec(t_mid - TI[i]) for i in range(N)]) # (N, N-1)
    err_mid = K0 - np.sum(Km * (a @ Km), axis=0)
    en_ventana = (t_fam >= TI[0]) & (t_fam <= TI[-1]) # solo para contrastar con los puntos medios
    err_max_real = err[en_ventana].max()
    # Frecuencia: B_k(w), magnitud y fase
    E = np.exp(-1j * np.outer(TI, w0)) # (N, n_w)
    H = a.T @ E # sum_i a[i,k] e^{-j w T_i}
    Bf = S_w[None, :] * H
    mag = np.abs(Bf)
    fase = np.array([np.unwrap(np.angle(Bf[k])) for k in range(N)])
    # Retardo de grupo analítico: Re{Hd conj(H)} / |H|^2
    Hd = a.T @ (TI[:, None] * E)
    num = np.real(Hd * np.conj(H))
    den = np.abs(H) ** 2
    tol = 1e-10 * den.max(axis=1, keepdims=True)
    gd = np.where(den > tol, num / np.where(den > tol, den, 1.0), np.nan)
    return dict(P=P, wc=wc, epsilon=epsilon, modo=modo, S_w=S_w, K_cov=K_cov, K0=K0, cond=cnd,
                B_t=B_t, err=err, t_mid=t_mid, err_mid=err_mid, err_max_real=err_max_real, mag=mag, fase=fase, gd=gd, w0=w0)
# ============================================================
# FIGURAS
# ============================================================
def leyenda_y_guardar(fig, ax_ref, titulo_leyenda, ruta):
    h, l = ax_ref.get_legend_handles_labels()
    fig.legend(h, l, loc='center right', bbox_to_anchor=(1.0, 0.5),
               fontsize=10, ncol=1, title=titulo_leyenda)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(ruta, dpi=200, bbox_inches='tight')
    print(f"   Guardado: {ruta.name}", flush=True)
    if not MOSTRAR:
        plt.close(fig) # evita acumular figuras al barrer varios epsilon
def graficar_espectro_covarianza(r, epsilon, ruta):
    """Espectro S(w) del filtro y función de covarianza K(tau) para un orden."""
    P, wc = r['P'], r['wc']
    x = r['w0'] / (2 * np.pi) if usar_Hz else r['w0']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ax1.plot(x, r['S_w'], color='blue', linewidth=1.2)
    ax1.set_title('Espectro del filtro de Chebyshev tipo II')
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(r'$S(\omega)$')
    ax1.grid(True, alpha=0.3)
    ax2.plot(tau_cov, r['K_cov'], color='orange', linewidth=1.2)
    ax2.set_title('Función de covarianza')
    ax2.set_xlabel(r'$\tau$ [s]')
    ax2.set_ylabel(r'$K(\tau)$')
    ax2.grid(True, alpha=0.3)
    integ = (r'integración en $(-\infty,\infty)$' if r['modo'] == 'inf'
             else rf'malla finita $|\omega|\leq{r["w0"][-1]:g}$')
    fig.suptitle(rf'Chebyshev II, $P={P}$, $\omega_c={wc}$, $\epsilon={epsilon}$ — {integ}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close(fig)
def ylim_error(resultados, grupo):
    """Límites comunes del eje y (escala log) para las filas de una imagen: desde una década por
    debajo del menor error en puntos medios hasta 3x el máximo (la extrapolación tiende a K(0))."""
    k = lambda r: r['K0'] if err_normalizar else 1.0
    lo = min((resultados[P]['err_mid'] / k(resultados[P])).min() for P in grupo)
    hi = max((resultados[P]['err'] / k(resultados[P])).max() for P in grupo)
    lo = 10 ** (np.floor(np.log10(max(lo, ERR_PISO))) - 1)
    return max(lo, ERR_PISO), 3 * max(hi, ERR_PISO)
def graficar_error(ax, r, TI, t_fam, ylim=None):
    """Grafica el error, los puntos medios y etiquetas de sigma max/min sin líneas guía."""
    k = r['K0'] if err_normalizar else 1.0
    err = r['err'] / k
    em = r['err_mid'] / k
    # Máximo real dentro de la ventana de muestreo y mínimo entre los puntos medios.
    sigma_max = r['err_max_real'] / k
    sigma_min = np.min(em)
    # Instantes de muestreo; no se sombrea la ventana.
    for Ti in TI:
        ax.axvline(Ti, color='gray', linestyle=':', linewidth=0.6)
    if err_escala == 'log':
        ax.semilogy(t_fam, np.maximum(err, ERR_PISO), color='black', linewidth=1.2)
        ax.set_ylim(*ylim)
    else:
        ax.plot(t_fam, err, color='black', linewidth=1.5)
    ax.plot(r['t_mid'], np.maximum(em, ERR_PISO), 'o', color='tab:red', markersize=4,
            label='puntos medios')
    # Artistas vacíos para mostrar solo las etiquetas en la leyenda, sin líneas horizontales.
    ax.plot([], [], linestyle='none', marker=None,
            label=rf'$\tilde{{\sigma}}^2_{{\max}}$ = {sigma_max:.2e}')
    ax.plot([], [], linestyle='none', marker=None,
            label=rf'$\tilde{{\sigma}}^2_{{\min}}$ = {sigma_min:.2e}')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
def figuras_grupo(resultados, epsilon, deltaT, OUT_DIR):
    """4 imágenes: tiempo y frecuencia por separado, para cada grupo de órdenes."""
    TI, t_fam = muestreo(deltaT)
    dt_lbl = etiqueta(deltaT)
    for nombre, grupo in grupos.items():
        ords = ", ".join(str(P) for P in grupo)
        n = len(grupo)
        # ── Imagen de TIEMPO: B_j(t) | error de reconstrucción ──
        fig_t, axt = plt.subplots(n, 2, figsize=(15, 4.8 * n))
        ylim = ylim_error(resultados, grupo) if err_escala == 'log' else None # común a las filas
        for fila, P in enumerate(grupo):
            r = resultados[P]
            lbl = rf'$P={P}$, $\omega_c={r["wc"]}$'
            for j in range(N):
                axt[fila, 0].plot(t_fam, r['B_t'][j], linewidth=0.8, color=COLORES[j],
                                  label=labels_B[j] if fila == 0 else None)
            graficar_error(axt[fila, 1], r, TI, t_fam, ylim)
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
        leyenda_y_guardar(fig_t, axt[0, 0], r'Índice $j$',
                          OUT_DIR / f"tiempo_{nombre}_deltaT{dt_lbl}.png")
        # ── Imagen de FRECUENCIA: magnitud | fase | retardo de grupo ──
        fig_f, axf = plt.subplots(n, 3, figsize=(20, 4.8 * n))
        for fila, P in enumerate(grupo):
            r = resultados[P]
            x = r['w0'] / (2 * np.pi) if usar_Hz else r['w0']
            lbl = rf'$P={P}$, $\omega_c={r["wc"]}$'
            for k in range(N):
                kw = dict(linewidth=0.8, color=COLORES[k])
                axf[fila, 0].plot(x, r['mag'][k], label=labels_B[k] if fila == 0 else None, **kw)
                axf[fila, 1].plot(x, r['fase'][k], **kw)
                axf[fila, 2].plot(x, r['gd'][k], **kw)
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
        leyenda_y_guardar(fig_f, axf[0, 0], r'Índice $k$',
                          OUT_DIR / f"frecuencia_{nombre}_deltaT{dt_lbl}.png")
# ============================================================
# POR ORDEN: espectro + covarianza, y funciones básicas individuales
#   orden_P{P}_wc{wc}/
#       espectro_covarianza.png
#       funciones_basicas_tiempo/B01_tiempo.png ... B09_tiempo.png
#       funciones_basicas_frecuencia/B01_frecuencia.png ... B09_frecuencia.png
# ============================================================
def figuras_por_orden(resultados, epsilon, deltaT, OUT_DIR):
    _, t_fam = muestreo(deltaT)
    for P in ordenes:
        r = resultados[P]
        x = r['w0'] / (2 * np.pi) if usar_Hz else r['w0']
        wc_lbl = etiqueta(r['wc'])
        DIR_ORD = OUT_DIR / f"orden_P{P}_wc{wc_lbl}"
        DIR_T = DIR_ORD / "funciones_basicas_tiempo"
        DIR_F = DIR_ORD / "funciones_basicas_frecuencia"
        DIR_T.mkdir(parents=True, exist_ok=True)
        DIR_F.mkdir(parents=True, exist_ok=True)
        sub_lbl = (rf'Chebyshev II, $P={P}$, $\omega_c={r["wc"]}$, $\epsilon={epsilon}$, '
                   rf'$\Delta T={deltaT}$, $N={N}$')
        # ── Espectro del filtro y función de covarianza ──
        graficar_espectro_covarianza(r, epsilon, DIR_ORD / "espectro_covarianza.png")
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
            fig.savefig(DIR_T / f"{pref}_tiempo.png", dpi=200, bbox_inches='tight')
            plt.close(fig)
            # ── B_k(jw): magnitud | fase | retardo de grupo ──
            fig, (axm, axp, axg) = plt.subplots(1, 3, figsize=(18, 4.5))
            axm.plot(x, r['mag'][j], color=COLORES[j], linewidth=1.2)
            axp.plot(x, r['fase'][j], color=COLORES[j], linewidth=1.2)
            axg.plot(x, r['gd'][j], color=COLORES[j], linewidth=1.2)
            axm.set_title('Magnitud', fontsize=11)
            axp.set_title('Fase', fontsize=11)
            axg.set_title('Retardo de grupo', fontsize=11)
            axm.set_ylabel(rf'$|B_{{{j+1}}}(\omega)|$')
            axp.set_ylabel(rf'$\angle B_{{{j+1}}}(\omega)$ [rad]')
            axg.set_ylabel(rf'$\tau_{{g,{j+1}}}(\omega)$ [s]')
            if gd_ylim is not None:
                axg.set_ylim(gd_ylim)
            elif np.nanmax(r['gd'][j]) - np.nanmin(r['gd'][j]) < 1e-3: # retardo constante (fase lineal)
                axg.set_ylim(np.nanmedian(r['gd'][j]) - 1, np.nanmedian(r['gd'][j]) + 1)
            axg.ticklabel_format(useOffset=False)
            for axx in (axm, axp, axg):
                axx.set_xlabel(xlab)
                axx.grid(True, alpha=0.3)
            fig.suptitle(rf'$B_{{{j+1}}}(j\omega)$ — {sub_lbl}', fontsize=13, fontweight='bold')
            fig.tight_layout()
            fig.savefig(DIR_F / f"{pref}_frecuencia.png", dpi=200, bbox_inches='tight')
            plt.close(fig)
        print(f"   Carpeta {DIR_ORD.name}: espectro_covarianza + {2*N} imágenes de B_j", flush=True)
# ============================================================
# BUCLE PRINCIPAL: Chevyshev_resultados/epsilon_*/deltaT_*/
#   (w_c no depende de deltaT: K(0) = 1 solo involucra a S(w))
# ============================================================
for epsilon in epsilons:
    DIR_EPS = OUT_ROOT / f"epsilon_{etiqueta(epsilon)}"
    for deltaT in deltaTs:
        TI, t_fam = muestreo(deltaT)
        OUT_DIR = DIR_EPS / f"deltaT_{etiqueta(deltaT)}"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\n epsilon = {epsilon}   deltaT = {deltaT}  ->  {OUT_DIR}\n{'=' * 60}", flush=True)
        resultados = {}
        for P in ordenes:
            wc = WC_TABLA[epsilon][P]
            print(f"Orden P={P}, wc={wc}, integración: {modo_integracion(P)} ...", flush=True)
            resultados[P] = calcular_orden(P, wc, epsilon, TI, t_fam)
            r = resultados[P]
            print(f"   K(0) = {r['K0']:.5f}   cond(kx) = {r['cond']:.3e}", flush=True)
            em, real = r['err_mid'].max(), r['err_max_real']
            aviso = f"   <- el máx. real excede en {100 * (real - em) / real:.1f}%" if (real - em) / real > ERR_TOL_MEDIO else ""
            print(f"   error máx. (puntos medios) = {em:.3e}   máx. real en la ventana = {real:.3e}{aviso}", flush=True)
            if r['cond'] > COND_ALERTA:
                print(f"   AVISO: cond(kx) > {COND_ALERTA:.0e}; inv(kx) puede ser poco confiable", flush=True)
        figuras_grupo(resultados, epsilon, deltaT, OUT_DIR)
        figuras_por_orden(resultados, epsilon, deltaT, OUT_DIR)
if MOSTRAR:
    plt.show()
else:
    plt.close('all')