#En el siguiente programa se hacen los calculos y gráficas de las funciones básicas en el tiempo y en la frecuencia. En el tiempo se ha gráficado el error de reconstrucción
#  y los puntos medios de cada intervalo de muestreo, en la frecuencia se ha graficado la magnitud, fase y retardo de grupo.
#Se ha considerado una familia de valores de epsilon y deltaT, para cada combinación se han generado las gráficas correspondientes.
#Para los valores de epsilon que es el rizado en la banda de rechazo se ha considerado los siguientes casos: epsilons = [0.5, 0.1, 0.05, 0.01]
#Para los valores de deltaT que es el periodo de muestreo se ha considerado los siguientes casos: deltaTs = [0.5, 1, 2]
#Para el orden del filtro se han considerado los órdenes del 1 al 6.
#De esta forma se han considerado todos los parámetros en el filtro para caracterizar la dependencia estadística entre el orden del filtro, las condiciones de muestreo y el rizado en la banda de rechazo.
#Con la caracterización anterior se pueden hacer diseños que no eran considerados en la teoria de filtrado estocastico para la optimización de recursos
#y la reducción de la complejidad computacional en el diseño de filtros de Chebyshev tipo II.
#La referencia y estructura principal del programa se tomo del programa "CHEBYSHEV_MANUAL.py" que se encuentra en el repositorio de GitHub:
# https://github.com/luis9879/SAMPLING-AND-RECONSTRUCTION/blob/main/src/filters/Chebyshev/Type_II/Order_2/CHEBYSHEV_MANUAL.py
#
#ORGANIZACIÓN DE LAS GRÁFICAS (carpeta raíz "Filtros_graficas_individuales"):
#   Cada imagen corresponde a UN orden P y UN deltaT, con una fila por cada valor de epsilon (mismo orden, distintos epsilon).
#   Se guardan para todos los órdenes y todos los deltaT, y solo las familias de funciones:
#     Filtros_graficas_individuales/orden_P{P}/tiempo_P{P}_deltaT{dT}.png
#         fila = epsilon;  columna 1: funciones básicas B_j(t);  columna 2: error de reconstrucción (+ puntos medios)
#     Filtros_graficas_individuales/orden_P{P}/frecuencia_P{P}_deltaT{dT}.png
#         fila = epsilon;  columnas: magnitud | fase | retardo de grupo de las funciones básicas B_k(jw)
#     Filtros_graficas_individuales/orden_P{P}/covarianza_espectro_P{P}.png
#         fila = epsilon;  columna 1: función de covarianza K(tau);  columna 2: espectro S(w)
#   No se guardan las funciones básicas de forma individual.
#
#NOVEDADES:
# 1) Tiempo de covarianza:  T_cov = integral de 0 a infinito de |K(tau)/K(0)| d tau,
#    calculado para cada filtro (epsilon, orden P) con la misma K(tau) del programa (K(0)=1).
#    Su valor se muestra como etiqueta en cada fila de las gráficas y se imprime en consola.
#    No depende de deltaT, por lo que se calcula una sola vez por (epsilon, P).
#    El "infinito" es numérico: se integra de 0 a TAU_INT_MAX = 200 s (ver nota en la tabla de w_c).
# 2) Retardo de grupo: el eje y se reescala automáticamente (percentiles) para que los picos que
#    aparecen en los ceros de la magnitud no aplasten la curva; los picos quedan fuera de escala.
# 3) Magnitud de B_k(jw): se incluye el factor 2*pi/I0 (I0 = integral de S(w) en la malla) para que sea la
#    transformada de Fourier de B_k(t) calculada con la K(tau) normalizada (K(0)=1). Antes se graficaba S(w)*H(w),
#    que difiere de esa transformada en el factor constante K_u(0) = I0/(2*pi) (entre 0.50 y 1.05 según el caso).
#    La fase y el retardo de grupo no cambian.

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
epsilons = [0.5,0.1,0.05,0.01] # amplitudes del rizo a barrer
deltaTs = [0.5, 1, 2] # periodos de muestreo a barrer [s]
ordenes = [1, 2, 3, 4, 5, 6]
# ------------------------------------------------------------
# w_c [rad/s] calibradas numéricamente para que T_cov = int_0^tau_max |K(tau)/K(0)| dtau = 1,
# con tau_max = TAU_INT_MAX = 200 s y con la misma malla, regla trapezoidal y paso en tau que usa
# tiempo_covarianza(). Mallas de integración en w:
#   P = 1: [-50, 50] rad/s con N_W = 10001 puntos (delta_omega = 0.01 rad/s).
#   Demás casos: [-5, 5] rad/s con N_W = 10001 puntos (delta_omega = 0.001 rad/s).
#   Excepción: epsilon=0.1, P=2 usa [-10, 10] rad/s con 2001 puntos (delta_omega = 0.01 rad/s).
# K(tau) se calcula por integración trapezoidal sobre la malla asignada.
# NOTA: el valor T_cov = 1 es válido con estas condiciones (tau_max y malla); no es una propiedad
# intrínseca del filtro, sobre todo para P par (S(w) no se anula en w_max y K(tau) tiene cola ~1/tau).
# OJO: RECALIBRAR_WC = True resuelve OTRA condición, K_malla(0) = (1/2pi) int S dw = 1, y NO reproduce esta tabla.
# ------------------------------------------------------------
RECALIBRAR_WC = False
WC_TABLA = {
    0.5: {
        1: 2.027032,
        2: 2.557337,
        3: 3.004684,
        4: 2.947387,
        5: 2.977419,
        6: 3.277311
    },
    0.1: {
        1: 10.135159,
        2: 4.043377,
        3: 3.312322,
        4: 3.182843,
        5: 3.199647,
        6: 3.291138
    },
    0.05: {
        1: 20.270319,
        2: 5.474319,
        3: 3.826812,
        4: 3.434926,
        5: 3.342282,
        6: 3.355904
    },
    0.01: {
        1: 101.351594,
        2: 11.808982,
        3: 5.920747,
        4: 4.479218,
        5: 3.949431,
        6: 3.726966
    }
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
# Ventana de retardos tau para evaluar la función de covarianza K(tau)
TAU_MAX = 5.0
N_TAU = 5001
tau_cov = np.linspace(-TAU_MAX, TAU_MAX, N_TAU)
usar_Hz = False # True -> eje en f [Hz]; False -> eje en w [rad/s]
gd_ylim = None # None -> escala automática (ver abajo); p. ej. (-10, 20) para fijarla a mano
# Escala automática del retardo de grupo: el eje y se ajusta a los percentiles GD_PERCENTILES de los
# valores graficados (se descartan los picos de los ceros de la magnitud) más un margen GD_MARGEN
# (fracción del rango entre percentiles). GD_SPAN_MIN_DT * deltaT es la altura mínima del eje, para que
# un retardo (casi) constante se vea como línea plana y no como ruido amplificado.
GD_PERCENTILES = (2, 98)
GD_MARGEN = 0.10
GD_SPAN_MIN_DT = 2.0
# Tiempo de covarianza T_cov = int_0^inf |K(tau)/K(0)| d tau. El "infinito" es numérico: se integra
# hasta TAU_INT_MAX [s] (limitado a 0.9 del semiperiodo de K en la malla, pi/delta_omega).
TAU_INT_MAX = 200.0
TCOV_PASOS = 8 # puntos de tau por semiperiodo de la oscilación más rápida (pi/w_max)
TCOV_TOL_CONV = 0.02 # aviso si T_cov cambia más de este % entre tau_max/2 y tau_max
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
OUT_ROOT = BASE_DIR / "Filtros_graficas_individuales"
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
# CALIBRACIÓN NUMÉRICA ALTERNATIVA DE w_c: K_malla(0) = (1/2pi) int S dw = 1
# (NO es la condición con la que se calculó WC_TABLA; ver nota arriba)
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
# TIEMPO DE COVARIANZA  T_cov = int_0^inf |K(tau)/K(0)| d tau
# ============================================================
_CACHE_TCOV = {} # T_cov no depende de deltaT: se calcula una vez por (epsilon, P, wc)
def tiempo_covarianza(S_w, w0, tau_max=TAU_INT_MAX):
    """Integral trapezoidal de |K_n(tau)| en [0, tau_max], con K_n(tau) = K(tau)/K(0) calculada con
    la misma malla de frecuencia y la misma regla trapezoidal que K_vec (K_n(0) = 1).
    Devuelve (T_cov, tau_max usado, T_cov acumulado hasta tau_max/2)."""
    dw = w0[1] - w0[0]
    tau_max = min(tau_max, 0.9 * np.pi / dw) # K en la malla es periódica (2*pi/dw) y simétrica en pi/dw
    d_tau = np.pi / (TCOV_PASOS * w0[-1])
    taus = np.linspace(0.0, tau_max, int(np.ceil(tau_max / d_tau)) + 1)
    I0 = np.trapezoid(S_w, x=w0)
    K_n = np.empty_like(taus)
    paso = 400 # bloques de tau para no desbordar la memoria
    for i in range(0, len(taus), paso):
        t = taus[i:i + paso]
        K_n[i:i + paso] = np.trapezoid(S_w[None, :] * np.cos(np.outer(t, w0)), x=w0, axis=1) / I0
    absK = np.abs(K_n)
    mitad = taus <= 0.5 * tau_max
    return np.trapezoid(absK, x=taus), tau_max, np.trapezoid(absK[mitad], x=taus[mitad])
# ============================================================
# ESCALA AUTOMÁTICA DEL RETARDO DE GRUPO
# ============================================================
def ylim_retardo_grupo(gd, deltaT):
    """Límites del eje y para el retardo de grupo que ignoran los picos (ceros de la magnitud).
    gd: arreglo con las curvas a graficar en un mismo eje. Devuelve (ymin, ymax) o None."""
    v = np.asarray(gd, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    p_lo, p_hi = np.percentile(v, GD_PERCENTILES)
    span = p_hi - p_lo
    lo, hi = p_lo - GD_MARGEN * span, p_hi + GD_MARGEN * span
    lo, hi = max(lo, v.min() - 0.05 * span), min(hi, v.max() + 0.05 * span) # sin picos: no sobra espacio
    span_min = GD_SPAN_MIN_DT * deltaT
    if hi - lo < span_min: # retardo casi constante: se centra y se da una altura mínima
        c = 0.5 * (lo + hi)
        lo, hi = c - 0.5 * span_min, c + 0.5 * span_min
    return lo, hi
# ============================================================
# CÁLCULO COMPLETO PARA UN ORDEN P Y UN epsilon
# ============================================================
def calcular_orden(P, wc, epsilon, TI, t_fam):
    # w0: eje de frecuencias para S(w), B_k(w) y la integración numérica de K(tau)
    w0 = obtener_malla_frecuencia(P, epsilon)
    S_w = ChebyshevII_espectro(wc, w0, epsilon, P)
    I0 = np.trapezoid(S_w, x=w0) # integral de S(w) en la malla (normaliza K y B_k(jw))
    modo = modo_integracion(P)
    def K_vec(tau):
        tau = np.asarray(tau, dtype=float)
        integral = np.trapezoid(
            S_w[None, :] * np.cos(np.outer(tau.ravel(), w0)),
            x=w0,
            axis=1,
        )
        return (integral / I0).reshape(tau.shape)
    # Función de covarianza evaluada en la ventana de retardos (K(0)=1)
    K_cov = K_vec(tau_cov)
    # Tiempo de covarianza: integral de 0 a infinito de |K(tau)/K(0)| (no depende de deltaT)
    clave = (epsilon, P, wc)
    if clave not in _CACHE_TCOV:
        _CACHE_TCOV[clave] = tiempo_covarianza(S_w, w0)
    Tcov, Tcov_taumax, Tcov_mitad = _CACHE_TCOV[clave]
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
    Bf = (2 * np.pi / I0) * S_w[None, :] * H # transformada de Fourier de B_k(t) con K(0)=1
    mag = np.abs(Bf)
    fase = np.array([np.unwrap(np.angle(Bf[k])) for k in range(N)])
    # Retardo de grupo analítico: Re{Hd conj(H)} / |H|^2
    Hd = a.T @ (TI[:, None] * E)
    num = np.real(Hd * np.conj(H))
    den = np.abs(H) ** 2
    tol = 1e-10 * den.max(axis=1, keepdims=True)
    gd = np.where(den > tol, num / np.where(den > tol, den, 1.0), np.nan)
    return dict(P=P, wc=wc, epsilon=epsilon, modo=modo, S_w=S_w, K_cov=K_cov, K0=K0, cond=cnd,
                Tcov=Tcov, Tcov_taumax=Tcov_taumax, Tcov_mitad=Tcov_mitad,
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
    print(f"   Guardado: {ruta.parent.name}/{ruta.name}", flush=True)
    if not MOSTRAR:
        plt.close(fig) # evita acumular figuras al barrer varios órdenes
def ylim_error(rs):
    """Límites comunes del eje y (escala log) para las filas de una imagen (los distintos epsilon): desde una
    década por debajo del menor error en puntos medios hasta 3x el máximo (la extrapolación tiende a K(0))."""
    k = lambda r: r['K0'] if err_normalizar else 1.0
    lo = min((r['err_mid'] / k(r)).min() for r in rs)
    hi = max((r['err'] / k(r)).max() for r in rs)
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
def etiqueta_fila(r):
    """Etiqueta de una fila: orden, epsilon, w_c y tiempo de covarianza."""
    return (rf'$P={r["P"]}$, $\epsilon={r["epsilon"]}$, $\omega_c={r["wc"]}$, '
            rf'$T_{{cov}}={r["Tcov"]:.3f}$ s')
# ============================================================
# POR ORDEN P Y POR deltaT: 2 imágenes (tiempo y frecuencia), una fila por cada epsilon
#   Filtros_graficas_individuales/orden_P{P}/tiempo_P{P}_deltaT{dT}.png
#   Filtros_graficas_individuales/orden_P{P}/frecuencia_P{P}_deltaT{dT}.png
# ============================================================
def figuras_orden(P, rs, deltaT, DIR_ORD):
    """rs: lista de resultados de calcular_orden para el mismo P y los distintos epsilon (una fila c/u)."""
    TI, t_fam = muestreo(deltaT)
    dt_lbl = etiqueta(deltaT)
    n = len(rs)
    eps_txt = ", ".join(str(r['epsilon']) for r in rs)
    # ── Imagen de TIEMPO: B_j(t) | error de reconstrucción ──
    fig_t, axt = plt.subplots(n, 2, figsize=(15, 4.8 * n), squeeze=False)
    ylim = ylim_error(rs) if err_escala == 'log' else None # común a las filas
    for fila, r in enumerate(rs):
        lbl = etiqueta_fila(r)
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
    fig_t.suptitle(rf'Estudio en Tiempo — Chebyshev II, orden $P={P}$ '
                   rf'($\Delta T={deltaT}$, $N={N}$; $\epsilon$ = {eps_txt})',
                   fontsize=14, fontweight='normal')
    leyenda_y_guardar(fig_t, axt[0, 0], r'Índice $j$',
                      DIR_ORD / f"tiempo_P{P}_deltaT{dt_lbl}.png")
    # ── Imagen de FRECUENCIA: magnitud | fase | retardo de grupo ──
    fig_f, axf = plt.subplots(n, 3, figsize=(20, 4.8 * n), squeeze=False)
    for fila, r in enumerate(rs):
        x = r['w0'] / (2 * np.pi) if usar_Hz else r['w0']
        lbl = etiqueta_fila(r)
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
        lim_gd = gd_ylim if gd_ylim is not None else ylim_retardo_grupo(r['gd'], deltaT)
        if lim_gd is not None:
            axf[fila, 2].set_ylim(lim_gd)
        for c in range(3):
            axf[fila, c].set_xlabel(xlab)
            axf[fila, c].grid(True, alpha=0.3)
    fig_f.suptitle(rf'Magnitud, Fase y Retardo de Grupo — Chebyshev II, orden $P={P}$ '
                   rf'($\Delta T={deltaT}$, $N={N}$; $\epsilon$ = {eps_txt})',
                   fontsize=14, fontweight='normal')
    leyenda_y_guardar(fig_f, axf[0, 0], r'Índice $k$',
                      DIR_ORD / f"frecuencia_P{P}_deltaT{dt_lbl}.png")

def figura_covarianza_espectro(P, rs, DIR_ORD):
    """Grafica la covarianza y el espectro para un orden, con una fila por epsilon."""
    n = len(rs)
    fig, axs = plt.subplots(n, 2, figsize=(15, 4.8 * n), squeeze=False)
    for fila, r in enumerate(rs):
        lbl = rf'$P={r["P"]}$, $\epsilon={r["epsilon"]}$, $\omega_c={r["wc"]}$'
        x_w = r['w0'] / (2 * np.pi) if usar_Hz else r['w0']
        axs[fila, 0].plot(tau_cov, r['K_cov'], color='black', linewidth=1.2)
        axs[fila, 1].plot(x_w, r['S_w'], color='tab:blue', linewidth=1.2)
        axs[fila, 0].set_title(f'Función de covarianza — {lbl}', fontsize=11)
        axs[fila, 1].set_title(f'Espectro — {lbl}', fontsize=11)
        axs[fila, 0].set_ylabel(r'$K(\tau)/K(0)$')
        axs[fila, 1].set_ylabel(r'$S(\omega)$')
        axs[fila, 0].set_xlabel(r'$\tau$ [s]')
        axs[fila, 1].set_xlabel(xlab)
        for ax in axs[fila]:
            ax.grid(True, alpha=0.3)
    fig.suptitle(rf'Función de covarianza y espectro — Chebyshev II, orden $P={P}$',
                 fontsize=14, fontweight='normal')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ruta = DIR_ORD / f"covarianza_espectro_P{P}.png"
    fig.savefig(ruta, dpi=200, bbox_inches='tight')
    print(f"   Guardado: {ruta.parent.name}/{ruta.name}", flush=True)
    if not MOSTRAR:
        plt.close(fig)

# ============================================================
# BUCLE PRINCIPAL: Filtros_graficas_individuales/orden_P*/
#   Para cada orden P y cada deltaT se calculan todos los epsilon y se genera una imagen de tiempo y una
#   de frecuencia con una fila por epsilon. La imagen de covarianza/espectro se genera una vez por orden.
#   (w_c no depende de deltaT: solo involucra a S(w); T_cov tampoco)
# ============================================================
for P in ordenes:
    DIR_ORD = OUT_ROOT / f"orden_P{P}"
    DIR_ORD.mkdir(parents=True, exist_ok=True)
    for deltaT in deltaTs:
        TI, t_fam = muestreo(deltaT)
        print(f"\n{'=' * 60}\n orden P = {P}   deltaT = {deltaT}  ->  {DIR_ORD}\n{'=' * 60}", flush=True)
        rs = []
        for epsilon in epsilons:
            wc = WC_TABLA[epsilon][P]
            print(f"Orden P={P}, epsilon={epsilon}, wc={wc}, integración: {modo_integracion(P)} ...", flush=True)
            r = calcular_orden(P, wc, epsilon, TI, t_fam)
            rs.append(r)
            print(f"   K(0) = {r['K0']:.5f}   cond(kx) = {r['cond']:.3e}", flush=True)
            dT_conv = abs(r['Tcov'] - r['Tcov_mitad']) / r['Tcov']
            print(f"   T_cov = {r['Tcov']:.4f} s   (integral de |K(tau)/K(0)| hasta tau = {r['Tcov_taumax']:.0f} s)", flush=True)
            if dT_conv > TCOV_TOL_CONV:
                print(f"   AVISO: T_cov aún crece {100 * dT_conv:.1f}% entre tau={r['Tcov_taumax'] / 2:.0f} y {r['Tcov_taumax']:.0f} s: "
                      f"K(tau) decae lento (cola ~1/tau, típica de un espectro con borde abrupto o S(w_max)={r['S_w'][-1]:.3g} != 0); "
                      f"el valor depende de TAU_INT_MAX", flush=True)
            em, real = r['err_mid'].max(), r['err_max_real']
            aviso = f"   <- el máx. real excede en {100 * (real - em) / real:.1f}%" if (real - em) / real > ERR_TOL_MEDIO else ""
            print(f"   error máx. (puntos medios) = {em:.3e}   máx. real en la ventana = {real:.3e}{aviso}", flush=True)
            if r['cond'] > COND_ALERTA:
                print(f"   AVISO: cond(kx) > {COND_ALERTA:.0e}; inv(kx) puede ser poco confiable", flush=True)
        figuras_orden(P, rs, deltaT, DIR_ORD)
    figura_covarianza_espectro(P, rs, DIR_ORD)
if MOSTRAR:
    plt.show()
else:
    plt.close('all')
