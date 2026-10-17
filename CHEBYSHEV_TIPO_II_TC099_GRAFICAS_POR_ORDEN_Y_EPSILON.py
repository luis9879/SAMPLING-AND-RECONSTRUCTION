# -*- coding: utf-8 -*-
"""
CHEBYSHEV TIPO II: FUNCIONES BASICAS Y VALIDACION DE TIEMPO DE COVARIANZA

Genera las cinco imagenes por cada orden P=1,...,6 y por cada epsilon:
    epsilon = 1.0, 0.5, 0.1, 0.05

Estructura de salida:
    Resultados_Chebyshev_Tipo_II_Tc_099/
        orden_P1/
            epsilon_1/       (cinco graficas individuales)
            epsilon_0_5/
            epsilon_0_1/
            epsilon_0_05/
            comparativas_impares_tiempo_error_eps1.png
            comparativas_pares_tiempo_error_eps1.png
            comparativas_impares_funciones_basicas_frecuencia_eps1.png
            comparativas_pares_funciones_basicas_frecuencia_eps1.png
            comparativas_impares_covarianza_espectro_eps1.png
            comparativas_pares_covarianza_espectro_eps1.png
            ... (comparativas para todos los epsilon)
        orden_P2/
        ...
        orden_P6/
        graficas_por_orden/
            P1_tiempo_funciones_basicas_error_todos_epsilons.png
            P1_frecuencia_magnitud_fase_retardo_todos_epsilons.png
            ... (dos imagenes por cada orden P=1,...,6)
        validacion_Tc.csv

Las imagenes comparativas que agrupan varios ordenes se copian en cada
carpeta orden_Pn. Las nuevas imagenes en graficas_por_orden mantienen fijo P
y comparan todos los epsilon en filas separadas.

Los w_c se calibraron numericamente para obtener Tc ~= 0.99 s al integrar
|K_n(tau)| entre 0 y 9 s, usando las mallas de frecuencia de este programa.
No es la integral hasta infinito. Si w_c queda fuera del intervalo de la
malla de frecuencia, el programa lo advierte: esos casos requieren una
malla mas amplia para una validacion espectral robusta.

Requisitos:
    pip install numpy matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. VECTOR DE MUESTRAS Y PARAMETROS
# ============================================================
T = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], dtype=float)
# Alternativas del MATLAB original:
# T = np.arange(0, 2.0001, 0.25)
# T = np.arange(0, 8.0001, 1.0)
# T = np.arange(0, 16.0001, 2.0)
# T = np.array([0, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16], dtype=float)

if np.any(np.diff(T) <= 0):
    raise ValueError("El vector T debe contener instantes estrictamente crecientes.")

nm = T.size
ordenes = [1, 2, 3, 4, 5, 6]
# Frecuencias de corte para Tc(0..9 s) ~= 0.99 s.
# Se conservan las mallas nw del programa MATLAB de referencia.
epsilons = [1.0, 0.5, 0.1, 0.05]
wc_tabla_por_epsilon = {
    1.0:  {1: 1.01657,  2: 2.19754, 3: 2.38568, 4: 2.84734, 5: 2.75901, 6: 3.45612},
    0.5:  {1: 2.03314,  2: 2.45911, 3: 2.70933, 4: 2.97675, 5: 3.08460, 6: 3.65883},
    0.1:  {1: 10.16572, 2: 4.08975, 3: 3.32918, 4: 3.20386, 5: 3.22470, 6: 4.32624},
    0.05: {1: 20.33145, 2: 5.51379, 3: 3.86262, 4: 3.46604, 5: 3.36354, 6: 4.70710},
}
nw_tabla = {1: 10001, 2: 460, 3: 411, 4: 361, 5: 350, 6: 330}

# Definicion numerica usada para la calibracion de la tabla.
TC_OBJETIVO = 0.99  # segundos, 1 % por debajo de 1 s
TC_TAU_MAX = 9.0    # integrar en 0 <= tau <= 9 s
TC_DT = 0.01        # paso temporal de integracion [s]

dt = 0.01
dw = 0.01
t_inicial = -5.0

if nm == 9:
    nt = 1401
elif nm == 19:
    nt = 2601
elif nm == 29:
    nt = 3801
else:
    nt = round(((T[-1] - T[0]) + 10) / dt) + 1

t = t_inicial + np.arange(nt) * dt
delta_T = float(np.median(np.diff(T)))

# Amplitudes conservadas del programa MATLAB original.
if nm == 9:
    x_muestras = np.array([1, 1.4, 0.6, -0.2, -0.5, -1.4, 1, 1.2, 0.5], dtype=float)
elif nm == 19:
    x_muestras = np.array([
        1, 1.4, 0.6, -0.2, -0.5, -1.4, 1, 1.2, 0.5,
        1.2, 1, 1.4, 0.6, -0.2, -0.5, -1.4, 1, 1.2, 0.5
    ], dtype=float)
elif nm == 29:
    x_muestras = np.array([
        1, 1.4, 0.6, -0.2, -0.5, -1.4, 1, 1.2, 0.5,
        1.2, 1, 1.4, 0.6, -0.2, -0.5, -1.4, 1, 1.2, 0.5,
        1.2, 1, 1.4, 0.6, -0.2, -0.5, -1.4, 1, 1.2, 0.5
    ], dtype=float)
else:
    raise ValueError("T debe contener 9, 19 o 29 instantes, como el programa original.")


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================
def espectro_chebyshev_II(w, epsilon, orden, wc):
    """
    Espectro Chebyshev tipo II usado en el TXT:
    S(w) = q^2/(1+q^2), q = epsilon*C_P(wc/|w|).

    Para |w| <= wc se usa cosh(P*acosh(wc/|w|)).
    Para |w| > wc se usa la continuacion real cos(P*acos(wc/|w|)).
    En w=0, el limite del espectro es 1.
    """
    w = np.asarray(w, dtype=float)
    wa = np.abs(w)
    S = np.ones_like(wa)
    no_cero = wa > 0
    x = wc / wa[no_cero]
    C = np.empty_like(x)
    interior = x >= 1.0
    exterior = ~interior
    C[interior] = np.cosh(orden * np.arccosh(x[interior]))
    C[exterior] = np.cos(orden * np.arccos(np.clip(x[exterior], 0.0, 1.0)))
    q2 = (epsilon * C) ** 2
    S[no_cero] = q2 / (1.0 + q2)
    return S


def covarianza_normalizada(tau, w, Sw, I0, bloque=100):
    """
    K_n(tau) = integral S(w)*cos(w*tau) dw / integral S(w) dw.
    Integra por bloques para limitar el uso de memoria.
    """
    tau_array = np.asarray(tau, dtype=float)
    shape_original = tau_array.shape
    tau_flat = tau_array.ravel()
    K_flat = np.empty_like(tau_flat)

    for inicio in range(0, tau_flat.size, bloque):
        fin = min(inicio + bloque, tau_flat.size)
        tau_b = tau_flat[inicio:fin]
        C = np.cos(tau_b[:, None] * w[None, :])
        numerador = np.trapezoid(C * Sw[None, :], x=w, axis=1)
        K_flat[inicio:fin] = numerador / I0

    K = K_flat.reshape(shape_original)
    if K.ndim == 0:
        return float(K)
    return K


def crear_evaluador_covarianza(w, Sw, I0, tau_min, tau_max, paso_tau):
    """Precalcula K_n en una malla fina y reutiliza sus valores.

    En la configuracion predeterminada, t, T, los puntos medios y tau_cov se
    alinean con la malla de 0.005 s. Si se solicita un retardo fuera de esa
    malla, se calcula directamente por integracion trapezoidal.
    """
    tau_cache = np.arange(tau_min, tau_max + 0.5 * paso_tau, paso_tau)
    valores_cache = covarianza_normalizada(tau_cache, w, Sw, I0, bloque=250)

    def evaluar(tau):
        tau_array = np.asarray(tau, dtype=float)
        shape_original = tau_array.shape
        escalados = (tau_array.ravel() - tau_cache[0]) / paso_tau
        indices = np.rint(escalados).astype(int)
        alineados = (
            np.all(np.abs(escalados - indices) < 1e-6)
            and np.all(indices >= 0)
            and np.all(indices < valores_cache.size)
        )
        if alineados:
            resultado = valores_cache[indices].reshape(shape_original)
        else:
            resultado = covarianza_normalizada(tau_array, w, Sw, I0)
        if np.ndim(resultado) == 0:
            return float(resultado)
        return resultado

    return evaluar


def guardar_figura(fig, carpeta, nombre):
    """Guarda y cierra la figura; la carpeta se crea si hace falta."""
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. CARPETA RAIZ DE RESULTADOS
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

carpeta_raiz = BASE_DIR / "Resultados_Chebyshev_Tipo_II_Tc_099"
carpeta_raiz.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. BUCLE PRINCIPAL: PROCESAR LOS SEIS ORDENES
# ============================================================
registros_tc = []
# Se guardan los arreglos necesarios para construir, al final, comparativas
# por epsilon y por grupos de ordenes impares (1,3,5) y pares (2,4,6).
resultados_graficas = {}

for epsilon in epsilons:
    eps_tag_loop = f"{epsilon:g}".replace(".", "_")
    print("\n" + "#" * 72)
    print(f"PROCESANDO epsilon={epsilon:g}")
    print("#" * 72)

    for orden in ordenes:
        wc = wc_tabla_por_epsilon[epsilon][orden]
        nw = nw_tabla[orden]

        print("\n" + "=" * 64)
        print(f"Chebyshev tipo II: orden P={orden}, epsilon={epsilon:g}, "
              f"wc={wc:g} rad/s")
        print("=" * 64)

        # --------------------------------------------------------
        # Malla frecuencial simetrica basada en nw e inw del MATLAB
        # --------------------------------------------------------
        Wmax = (nw - 1) * dw
        w = np.linspace(-Wmax, Wmax, 2 * nw - 1)
        Sw = espectro_chebyshev_II(w, epsilon, orden, wc)

        # --------------------------------------------------------
        # Varianza espectral y covarianza normalizada
        # --------------------------------------------------------
        I0 = np.trapezoid(Sw, x=w)
        if not np.isfinite(I0) or I0 <= 0:
            print(f"ADVERTENCIA: integral espectral no valida para P={orden}; se omite.")
            continue

        K_eval = crear_evaluador_covarianza(
            w, Sw, I0,
            tau_min=float(t[0] - T[-1]),
            tau_max=float(t[-1] - T[0]),
            paso_tau=dt / 2.0,
        )
        K0 = K_eval(np.array([0.0]))[0]

        # Validacion del tiempo de covarianza en la misma ventana usada para
        # construir la tabla w_c: integral de |K_n(tau)| entre 0 y 9 s.
        tau_tc = np.arange(0.0, TC_TAU_MAX + 0.5 * TC_DT, TC_DT)
        K_tc = K_eval(tau_tc)
        Tc_calculado = float(np.trapezoid(np.abs(K_tc), x=tau_tc))
        registros_tc.append({
            "epsilon": epsilon,
            "orden": orden,
            "wc_rad_s": wc,
            "Wmax_malla_rad_s": Wmax,
            "Tc_0_a_9_s": Tc_calculado,
            "objetivo_s": TC_OBJETIVO,
            "error_absoluto_s": abs(Tc_calculado - TC_OBJETIVO),
            "wc_dentro_malla": bool(wc <= Wmax),
        })
        print(f"Tc(0..{TC_TAU_MAX:g} s) = {Tc_calculado:.7f} s; "
              f"objetivo = {TC_OBJETIVO:.2f} s")
        if wc > Wmax:
            print("ADVERTENCIA: wc excede Wmax. La calibracion usa un espectro "
                  "truncado por la malla; ampliar la malla antes de interpretar "
                  "este caso como un resultado fisico robusto.")

        print(f"Malla: [{w[0]:.4f}, {w[-1]:.4f}] rad/s; "
              f"Nw={w.size}; delta_w={w[1]-w[0]:.6g} rad/s")
        print(f"I0 = {I0:.12g}; K(0) normalizada = {K0:.12g}")

        # --------------------------------------------------------
        # Matriz de covarianza e inversa
        # --------------------------------------------------------
        retardos_muestras = T[:, None] - T[None, :]
        R = K_eval(retardos_muestras)
        R = 0.5 * (R + R.T)
        cond_R = np.linalg.cond(R)
        try:
            A = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            print(f"ADVERTENCIA: matriz R singular para P={orden}; se omite.")
            continue

        print(f"Numero de condicion cond(R) = {cond_R:.6e}")
        if cond_R > 1e12:
            print("ADVERTENCIA: R esta mal condicionada; revise la malla y la precision.")

        # --------------------------------------------------------
        # Funciones basicas en el tiempo
        # --------------------------------------------------------
        Kmat = np.zeros((nm, nt), dtype=float)
        for j in range(nm):
            Kmat[j, :] = K_eval(t - T[j])
        B_t = A.T @ Kmat

        # --------------------------------------------------------
        # Funcion de reconstruccion con las amplitudes originales
        # --------------------------------------------------------
        FR = x_muestras @ B_t

        # --------------------------------------------------------
        # Error de reconstruccion y puntos medios
        # --------------------------------------------------------
        Var = K0 - np.sum(Kmat * (A @ Kmat), axis=0)
        t_medios = 0.5 * (T[:-1] + T[1:])
        Kmat_medios = np.zeros((nm, t_medios.size), dtype=float)
        for j in range(nm):
            Kmat_medios[j, :] = K_eval(t_medios - T[j])
        Var_medios = K0 - np.sum(Kmat_medios * (A @ Kmat_medios), axis=0)

        mascara_ventana = (t >= T[0]) & (t <= T[-1])
        Var_ventana = Var[mascara_ventana]
        t_ventana = t[mascara_ventana]
        i_max_real = int(np.argmax(Var_ventana))
        i_max_medio = int(np.argmax(Var_medios))
        i_min_medio = int(np.argmin(Var_medios))
        max_error_real = float(Var_ventana[i_max_real])
        max_error_medios = float(Var_medios[i_max_medio])
        min_error_medios = float(Var_medios[i_min_medio])

        print(f"Error maximo real = {max_error_real:.6e} en t={t_ventana[i_max_real]:.6g} s")
        print(f"Error maximo en puntos medios = {max_error_medios:.6e} "
              f"en t={t_medios[i_max_medio]:.6g} s")
        print(f"Error minimo en puntos medios = {min_error_medios:.6e} "
              f"en t={t_medios[i_min_medio]:.6g} s")
        if np.min(Var) < -1e-8:
            print("ADVERTENCIA: Var(t) contiene valores negativos apreciables.")

        # --------------------------------------------------------
        # Funciones basicas en frecuencia: modulo, fase y retardo
        # --------------------------------------------------------
        E = np.exp(-1j * T[:, None] * w[None, :])
        H = A.T @ E
        B_w = (2.0 * np.pi / I0) * H * Sw[None, :]
        modulo_B = np.abs(B_w)
        fase_B = np.unwrap(np.angle(B_w), axis=1)

        Hd = A.T @ (T[:, None] * E)
        den_gd = np.abs(H) ** 2
        num_gd = np.real(Hd * np.conj(H))
        retardo_grupo = np.full(den_gd.shape, np.nan, dtype=float)
        for k in range(nm):
            tolerancia = 1e-10 * np.max(den_gd[k, :])
            valido = den_gd[k, :] > tolerancia
            retardo_grupo[k, valido] = num_gd[k, valido] / den_gd[k, valido]

        # --------------------------------------------------------
        # Covarianza para graficar
        # --------------------------------------------------------
        tau_cov = np.linspace(-5.0, 5.0, 2001)
        K_cov = K_eval(tau_cov)

        # --------------------------------------------------------
        # Carpeta por orden y subcarpeta individual por epsilon.
        # Las comparativas impares/pares se guardan despues directamente
        # dentro de cada carpeta orden_Pn.
        # --------------------------------------------------------
        carpeta_orden = carpeta_raiz / f"orden_P{orden}"
        carpeta_orden.mkdir(parents=True, exist_ok=True)
        eps_tag = f"{epsilon:g}".replace(".", "_")
        carpeta_epsilon = carpeta_orden / f"epsilon_{eps_tag}"
        carpeta_epsilon.mkdir(parents=True, exist_ok=True)
        prefijo = f"TipoII_P{orden}_eps{eps_tag}_N{nm}"

        # Cachear solo los arreglos necesarios para las comparativas.
        resultados_graficas[(epsilon, orden)] = {
            "epsilon": epsilon,
            "orden": orden,
            "wc": wc,
            "t": t.copy(),
            "B_t": B_t.copy(),
            "Var": Var.copy(),
            "t_medios": t_medios.copy(),
            "Var_medios": Var_medios.copy(),
            "t_ventana": t_ventana.copy(),
            "max_error_real": max_error_real,
            "w": w.copy(),
            "modulo_B": modulo_B.copy(),
            "fase_B": fase_B.copy(),
            "retardo_grupo": retardo_grupo.copy(),
            "tau_cov": tau_cov.copy(),
            "K_cov": K_cov.copy(),
            "Sw": Sw.copy(),
            "Tc": Tc_calculado,
        }

        # ========================================================
        # FIGURA 1: FUNCIONES BASICAS EN EL TIEMPO
        # ========================================================
        fig, ax = plt.subplots(figsize=(11, 6))
        for k in range(nm):
            ax.plot(t, B_t[k, :], linewidth=1.05, label=f"$B_{{{k+1}}}(t)$")
        ax.grid(True)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("$B_k(t)$")
        ax.set_title(f"Funciones basicas en tiempo - Chebyshev II, P={orden}, "
                     rf"$\epsilon={epsilon:g}$")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        guardar_figura(fig, carpeta_epsilon, f"{prefijo}_funciones_basicas_tiempo.png")

        # ========================================================
        # FIGURA 2: FUNCION DE RECONSTRUCCION
        # ========================================================
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(t, FR, "b-", linewidth=1.4, label="$FR(t)$")
        ax.plot(T, x_muestras, "ro", markerfacecolor="r", label="Muestras")
        ax.grid(True)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Amplitud")
        ax.set_title(f"Funcion de reconstruccion - Chebyshev II, P={orden}, "
                     rf"$\epsilon={epsilon:g}$")
        ax.legend(loc="best")
        guardar_figura(fig, carpeta_epsilon, f"{prefijo}_reconstruccion.png")

        # ========================================================
        # FIGURA 3: ERROR DE RECONSTRUCCION Y PUNTOS MEDIOS
        # ========================================================
        fig, ax = plt.subplots(figsize=(10, 5.5))
        piso_error = 1e-12
        ax.semilogy(t, np.maximum(Var, piso_error), "k-", linewidth=1.25,
                    label="$Var(t)$")
        for ti in T:
            ax.axvline(ti, linestyle=":", color="0.65", linewidth=0.8)
        ax.semilogy(t_medios, np.maximum(Var_medios, piso_error), "ro",
                    markerfacecolor="r", label="Puntos medios")
        ax.plot(t_ventana[i_max_real], max(max_error_real, piso_error), "mp",
                markerfacecolor="m", markersize=10,
                label=f"Maximo real = {max_error_real:.3e}")
        ax.plot(t_medios[i_min_medio], max(min_error_medios, piso_error), "bv",
                markerfacecolor="b", markersize=8,
                label=f"Minimo puntos medios = {min_error_medios:.3e}")
        ax.grid(True, which="both")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("$Var(t)$")
        ax.set_title(f"Error de reconstruccion - Chebyshev II, P={orden}, "
                     rf"$\epsilon={epsilon:g}$")
        ax.legend(loc="best")
        guardar_figura(fig, carpeta_epsilon, f"{prefijo}_error_reconstruccion.png")

        # ========================================================
        # FIGURA 4: MODULO, FASE Y RETARDO DE GRUPO EN FRECUENCIA
        # ========================================================
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for k in range(nm):
            axes[0].plot(w, modulo_B[k, :], linewidth=1.0, label=f"$B_{{{k+1}}}$")
        axes[0].grid(True)
        axes[0].set_ylabel(r"$|B_k(\omega)|$")
        axes[0].set_title(f"Magnitud de las funciones basicas - P={orden}, "
                          rf"$\epsilon={epsilon:g}$")
        axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

        for k in range(nm):
            axes[1].plot(w, fase_B[k, :], linewidth=1.0)
        axes[1].grid(True)
        axes[1].set_ylabel("Fase [rad]")
        axes[1].set_title("Fase de las funciones basicas")

        for k in range(nm):
            axes[2].plot(w, retardo_grupo[k, :], linewidth=1.0)
        axes[2].grid(True)
        axes[2].set_xlabel(r"$\omega$ [rad/s]")
        axes[2].set_ylabel(r"$\tau_g$ [s]")
        axes[2].set_title("Retardo de grupo")

        # Escala de retardo basada en percentiles para evitar que picos aislados
        # dominen la visualizacion, como en el programa MATLAB.
        valores_gd = retardo_grupo[np.isfinite(retardo_grupo)]
        if valores_gd.size:
            q02, q98 = np.percentile(valores_gd, [2, 98])
            rango = q98 - q02
            if rango > 0:
                lim_inf = max(q02 - 0.10 * rango, np.min(valores_gd) - 0.05 * rango)
                lim_sup = min(q98 + 0.10 * rango, np.max(valores_gd) + 0.05 * rango)
                rango_minimo = 2.0 * delta_T
                if lim_sup - lim_inf < rango_minimo:
                    centro = 0.5 * (lim_sup + lim_inf)
                    lim_inf = centro - rango_minimo / 2
                    lim_sup = centro + rango_minimo / 2
                if lim_sup > lim_inf:
                    axes[2].set_ylim(lim_inf, lim_sup)

        fig.tight_layout()
        guardar_figura(fig, carpeta_epsilon, f"{prefijo}_funciones_basicas_frecuencia.png")

        # ========================================================
        # FIGURA 5: COVARIANZA Y DENSIDAD ESPECTRAL
        # ========================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(tau_cov, K_cov, "k-", linewidth=1.25)
        axes[0].grid(True)
        axes[0].set_xlabel(r"$\tau$ [s]")
        axes[0].set_ylabel("$K(\\tau)/K(0)$")
        axes[0].set_title(f"Covarianza normalizada; Tc(0..{TC_TAU_MAX:g} s)={Tc_calculado:.4f} s")

        axes[1].plot(w, Sw, "b-", linewidth=1.15)
        axes[1].grid(True)
        axes[1].set_xlabel(r"$\omega$ [rad/s]")
        axes[1].set_ylabel("$S(\\omega)$")
        axes[1].set_title("Densidad espectral")
        fig.tight_layout()
        guardar_figura(fig, carpeta_epsilon, f"{prefijo}_covarianza_espectro.png")

        print(f"Resultados P={orden}, epsilon={epsilon:g} guardados en:\n{carpeta_epsilon}")


# ============================================================
# 5. COMPARATIVAS POR EPSILON: ORDENES IMPARES Y PARES
# ============================================================
def guardar_comparativa_en_cada_orden(fig, nombre_archivo):
    """Guarda la misma comparativa en el directorio raiz de cada orden_Pn.

    Estas figuras comparan varios ordenes; se replican en las seis carpetas
    orden_Pn tal como se solicita, mientras que las figuras individuales
    permanecen dentro de orden_Pn/epsilon_x/.
    """
    for P_destino in ordenes:
        destino = carpeta_raiz / f"orden_P{P_destino}" / nombre_archivo
        destino.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, dpi=140, bbox_inches="tight")
    plt.close(fig)


def estilo_eje(ax):
    ax.grid(True, alpha=0.28)
    ax.tick_params(axis="both", labelsize=8)


def generar_comparativa_tiempo_error(epsilon, grupo_ordenes, etiqueta_grupo):
    """Una fila por orden; columnas: funciones basicas en tiempo y error."""
    fig, axes = plt.subplots(len(grupo_ordenes), 2,
                             figsize=(15, 4.4 * len(grupo_ordenes)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for fila, P in enumerate(grupo_ordenes):
        r = resultados_graficas[(epsilon, P)]
        ax_b, ax_e = axes[fila]
        for j in range(nm):
            ax_b.plot(r["t"], r["B_t"][j], lw=0.9,
                      color=colores[j % len(colores)], label=f"$B_{{{j+1}}}(t)$")
        ax_b.set_title(f"Funciones basicas — P={P}, wc={r['wc']:.5g}")
        ax_b.set_xlabel("t [s]")
        ax_b.set_ylabel("$B_j(t)$")
        estilo_eje(ax_b)

        piso = 1e-12
        ax_e.semilogy(r["t"], np.maximum(r["Var"], piso), color="black",
                      lw=1.15, label="$Var(t)$")
        ax_e.semilogy(r["t_medios"], np.maximum(r["Var_medios"], piso), "o",
                      color="tab:red", ms=4, label="Puntos medios")
        for ti in T:
            ax_e.axvline(ti, linestyle=":", color="0.65", linewidth=0.6)
        mask = (r["t"] >= T[0]) & (r["t"] <= T[-1])
        if np.any(mask):
            idx_max = int(np.argmax(r["Var"][mask]))
            t_max = r["t"][mask][idx_max]
            y_max = max(float(r["Var"][mask][idx_max]), piso)
            ax_e.plot(t_max, y_max, "mp", ms=8,
                      label=f"Max. real={r['max_error_real']:.2e}")
        idx_min = int(np.argmin(r["Var_medios"]))
        ax_e.plot(r["t_medios"][idx_min],
                  max(float(r["Var_medios"][idx_min]), piso), "bv", ms=7,
                  label=f"Min. medios={r['Var_medios'][idx_min]:.2e}")
        ax_e.set_title(f"Error de reconstruccion — P={P}")
        ax_e.set_xlabel("t [s]")
        ax_e.set_ylabel("$Var(t)$")
        estilo_eje(ax_e)
        ax_e.legend(loc="best", fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Funciones basicas", loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"Tiempo y error de reconstruccion — ordenes {etiqueta_grupo}; "
                 f"epsilon={epsilon:g}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    eps_tag = f"{epsilon:g}".replace(".", "_")
    nombre = f"comparativas_{etiqueta_grupo}_tiempo_error_eps{eps_tag}.png"
    guardar_comparativa_en_cada_orden(fig, nombre)


def generar_comparativa_funciones_frecuencia(epsilon, grupo_ordenes, etiqueta_grupo):
    """Una fila por orden; columnas: magnitud, fase y retardo de grupo."""
    fig, axes = plt.subplots(len(grupo_ordenes), 3,
                             figsize=(20, 4.3 * len(grupo_ordenes)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for fila, P in enumerate(grupo_ordenes):
        r = resultados_graficas[(epsilon, P)]
        w = r["w"]
        for j in range(nm):
            color = colores[j % len(colores)]
            axes[fila, 0].plot(w, r["modulo_B"][j], lw=0.85, color=color,
                               label=f"$B_{{{j+1}}}$")
            axes[fila, 1].plot(w, r["fase_B"][j], lw=0.85, color=color)
            axes[fila, 2].plot(w, r["retardo_grupo"][j], lw=0.85, color=color)
        axes[fila, 0].set_title(f"Magnitud — P={P}, wc={r['wc']:.5g}")
        axes[fila, 1].set_title(f"Fase — P={P}")
        axes[fila, 2].set_title(f"Retardo de grupo — P={P}")
        axes[fila, 0].set_ylabel("$|B_k(\\omega)|$")
        axes[fila, 1].set_ylabel("Fase [rad]")
        axes[fila, 2].set_ylabel("$\\tau_g$ [s]")
        for ax in axes[fila]:
            ax.set_xlabel(r"$\omega$ [rad/s]")
            estilo_eje(ax)
        values = r["retardo_grupo"][np.isfinite(r["retardo_grupo"])]
        if values.size:
            q02, q98 = np.percentile(values, [2, 98])
            span = q98 - q02
            if span > 0:
                lo = max(q02 - 0.1 * span, values.min() - 0.05 * span)
                hi = min(q98 + 0.1 * span, values.max() + 0.05 * span)
                if hi > lo:
                    axes[fila, 2].set_ylim(lo, hi)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Indice k", loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"Funciones basicas en frecuencia — ordenes {etiqueta_grupo}; "
                 f"epsilon={epsilon:g}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    eps_tag = f"{epsilon:g}".replace(".", "_")
    nombre = f"comparativas_{etiqueta_grupo}_funciones_basicas_frecuencia_eps{eps_tag}.png"
    guardar_comparativa_en_cada_orden(fig, nombre)


def generar_comparativa_covarianza_espectro(epsilon, grupo_ordenes, etiqueta_grupo):
    """Una fila por orden; columna izquierda covarianza y derecha espectro."""
    fig, axes = plt.subplots(len(grupo_ordenes), 2,
                             figsize=(15, 4.1 * len(grupo_ordenes)), squeeze=False)

    for fila, P in enumerate(grupo_ordenes):
        r = resultados_graficas[(epsilon, P)]
        axes[fila, 0].plot(r["tau_cov"], r["K_cov"], color="black", lw=1.15)
        axes[fila, 0].axvline(0, color="0.65", linestyle=":", lw=0.7)
        axes[fila, 0].set_title(f"Covarianza — P={P}, wc={r['wc']:.5g}; "
                                f"Tc(0–9 s)={r['Tc']:.4f} s")
        axes[fila, 0].set_xlabel(r"$\tau$ [s]")
        axes[fila, 0].set_ylabel(r"$K(\tau)/K(0)$")
        axes[fila, 1].plot(r["w"], r["Sw"], color="tab:blue", lw=1.1)
        axes[fila, 1].set_title(f"Espectro — P={P}")
        axes[fila, 1].set_xlabel(r"$\omega$ [rad/s]")
        axes[fila, 1].set_ylabel("$S(\\omega)$")
        for ax in axes[fila]:
            estilo_eje(ax)

    fig.suptitle(f"Funcion de covarianza y espectro — ordenes {etiqueta_grupo}; "
                 f"epsilon={epsilon:g}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    eps_tag = f"{epsilon:g}".replace(".", "_")
    nombre = f"comparativas_{etiqueta_grupo}_covarianza_espectro_eps{eps_tag}.png"
    guardar_comparativa_en_cada_orden(fig, nombre)


# Las comparativas se preparan por separado para cada epsilon para no mezclar
# espectros, covarianzas o errores correspondientes a distintos niveles de rizo.
grupo_ordenes = {
    "impares": [1, 3, 5],
    "pares": [2, 4, 6],
}
for epsilon in epsilons:
    for etiqueta_grupo, ordenes_grupo in grupo_ordenes.items():
        generar_comparativa_tiempo_error(epsilon, ordenes_grupo, etiqueta_grupo)
        generar_comparativa_funciones_frecuencia(epsilon, ordenes_grupo, etiqueta_grupo)
        generar_comparativa_covarianza_espectro(epsilon, ordenes_grupo, etiqueta_grupo)
    print(f"Comparativas de ordenes impares y pares creadas para epsilon={epsilon:g}.")


# ============================================================
# 6. COMPARATIVAS POR ORDEN FIJO Y TODOS LOS EPSILON
# ============================================================
# Estas nuevas figuras se guardan en la raiz de resultados, dentro de
# graficas_por_orden/. Cada imagen mantiene fijo P y dedica una fila a cada
# epsilon = 1, 0.5, 0.1 y 0.05.
carpeta_graficas_por_orden = carpeta_raiz / "graficas_por_orden"
carpeta_graficas_por_orden.mkdir(parents=True, exist_ok=True)


def limites_error_comunes(registros_orden):
    """Calcula limites y comunes para comparar los errores de todos los epsilon."""
    positivos = []
    maximo = 0.0
    for r in registros_orden:
        err = np.asarray(r["Var"], dtype=float)
        err_mid = np.asarray(r["Var_medios"], dtype=float)
        positivos.extend(err[np.isfinite(err) & (err > 1e-12)].tolist())
        positivos.extend(err_mid[np.isfinite(err_mid) & (err_mid > 1e-12)].tolist())
        if np.any(np.isfinite(err)):
            maximo = max(maximo, float(np.nanmax(err)))
        if np.any(np.isfinite(err_mid)):
            maximo = max(maximo, float(np.nanmax(err_mid)))
    minimo = min(positivos) if positivos else 1e-12
    y_min = max(1e-12, 10.0 ** (np.floor(np.log10(minimo)) - 1.0))
    y_max = max(3.0 * maximo, 10.0 * y_min)
    return y_min, y_max


def graficar_por_orden_tiempo_error(orden):
    """Figura por orden fijo: filas=epsilon, columnas=funciones basicas/error."""
    registros = [resultados_graficas[(epsilon, orden)] for epsilon in epsilons]
    fig, axes = plt.subplots(len(epsilons), 2,
                             figsize=(15, 4.0 * len(epsilons)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    yerr = limites_error_comunes(registros)

    for fila, r in enumerate(registros):
        epsilon = r["epsilon"]
        ax_b, ax_e = axes[fila]
        for j in range(nm):
            ax_b.plot(r["t"], r["B_t"][j], lw=0.9,
                      color=colores[j % len(colores)],
                      label=rf"$B_{{{j+1}}}(t)$" if fila == 0 else None)
        ax_b.set_title(
            rf"Funciones basicas — $P={orden}$, $\epsilon={epsilon:g}$, "
            rf"$\omega_c={r['wc']:.5g}$"
        )
        ax_b.set_xlabel("Tiempo t [s]")
        ax_b.set_ylabel(r"$B_j(t)$")
        ax_b.grid(True, alpha=0.28)

        piso = 1e-12
        ax_e.semilogy(r["t"], np.maximum(r["Var"], piso),
                      color="black", lw=1.1, label=r"$Var(t)$")
        ax_e.semilogy(r["t_medios"], np.maximum(r["Var_medios"], piso),
                      "o", color="tab:red", ms=3.5, label="Puntos medios")
        for ti in T:
            ax_e.axvline(ti, color="0.7", linestyle=":", linewidth=0.55)

        mask = (r["t"] >= T[0]) & (r["t"] <= T[-1])
        if np.any(mask):
            var_window = r["Var"][mask]
            t_window = r["t"][mask]
            idxmax = int(np.argmax(var_window))
            ax_e.plot(t_window[idxmax], max(float(var_window[idxmax]), piso),
                      "mp", ms=7, label=f"Max. real={r['max_error_real']:.2e}")
        idxmin = int(np.argmin(r["Var_medios"]))
        ax_e.plot(r["t_medios"][idxmin],
                  max(float(r["Var_medios"][idxmin]), piso),
                  "bv", ms=6, label=f"Min. medios={float(r['Var_medios'][idxmin]):.2e}")
        ax_e.set_ylim(*yerr)
        ax_e.set_title(rf"Error de reconstruccion — $\epsilon={epsilon:g}$, "
                       rf"$T_c(0\!:\!9)={r['Tc']:.4f}$ s")
        ax_e.set_xlabel("Tiempo t [s]")
        ax_e.set_ylabel(r"$Var(t)$")
        ax_e.grid(True, which="both", alpha=0.28)
        ax_e.legend(loc="best", fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Funciones basicas", loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(
        rf"Chebyshev tipo II — orden fijo $P={orden}$: funciones basicas y error "
        "para todos los valores de epsilon", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    ruta = carpeta_graficas_por_orden / f"P{orden}_tiempo_funciones_basicas_error_todos_epsilons.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparativa por orden guardada: {ruta}")


def graficar_por_orden_frecuencia(orden):
    """Figura por orden fijo: filas=epsilon, columnas=magnitud/fase/retardo."""
    registros = [resultados_graficas[(epsilon, orden)] for epsilon in epsilons]
    fig, axes = plt.subplots(len(epsilons), 3,
                             figsize=(20, 4.0 * len(epsilons)), squeeze=False)
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for fila, r in enumerate(registros):
        epsilon = r["epsilon"]
        w = r["w"]
        ax_mag, ax_fase, ax_gd = axes[fila]
        for j in range(nm):
            color = colores[j % len(colores)]
            label = rf"$B_{{{j+1}}}$" if fila == 0 else None
            ax_mag.plot(w, r["modulo_B"][j], color=color, lw=0.85, label=label)
            ax_fase.plot(w, r["fase_B"][j], color=color, lw=0.85)
            ax_gd.plot(w, r["retardo_grupo"][j], color=color, lw=0.85)

        ax_mag.set_title(rf"Magnitud — $\epsilon={epsilon:g}$, $\omega_c={r['wc']:.5g}$")
        ax_fase.set_title(rf"Fase — $\epsilon={epsilon:g}$")
        ax_gd.set_title(rf"Retardo de grupo — $\epsilon={epsilon:g}$")
        ax_mag.set_ylabel(r"$|B_k(\omega)|$")
        ax_fase.set_ylabel("Fase [rad]")
        ax_gd.set_ylabel(r"$\tau_g$ [s]")
        for ax in (ax_mag, ax_fase, ax_gd):
            ax.set_xlabel(r"Frecuencia $\omega$ [rad/s]")
            ax.grid(True, alpha=0.28)

        values = r["retardo_grupo"][np.isfinite(r["retardo_grupo"])]
        if values.size:
            q02, q98 = np.percentile(values, [2, 98])
            span = q98 - q02
            if span > 0:
                lo = max(q02 - 0.10 * span, float(values.min()) - 0.05 * span)
                hi = min(q98 + 0.10 * span, float(values.max()) + 0.05 * span)
                if hi > lo:
                    ax_gd.set_ylim(lo, hi)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Indice k", loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(
        rf"Chebyshev tipo II — orden fijo $P={orden}$: magnitud, fase y retardo de grupo "
        "para todos los valores de epsilon", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    ruta = carpeta_graficas_por_orden / f"P{orden}_frecuencia_magnitud_fase_retardo_todos_epsilons.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparativa por orden guardada: {ruta}")


# Crear las dos nuevas imagenes agregadas para cada orden P=1,...,6.
# Cada una tiene cuatro filas, una por epsilon: 1, 0.5, 0.1 y 0.05.
for orden in ordenes:
    graficar_por_orden_tiempo_error(orden)
    graficar_por_orden_frecuencia(orden)


# Guardar una tabla CSV para comprobar los T_c calculados en cada condicion.
import csv
csv_path = carpeta_raiz / "validacion_Tc.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    campos = ["epsilon", "orden", "wc_rad_s", "Wmax_malla_rad_s", "Tc_0_a_9_s",
              "objetivo_s", "error_absoluto_s", "wc_dentro_malla"]
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    writer.writerows(registros_tc)

print("\nProceso terminado.")
print(f"Ordenes procesados: {ordenes}")
print(f"Epsilons procesados: {epsilons}")
print(f"Carpeta raiz: {carpeta_raiz}")
print("Cada orden_Pn contiene subcarpetas epsilon_1, epsilon_0_5, epsilon_0_1 y epsilon_0_05.")
print("Cada subcarpeta epsilon contiene cinco imagenes individuales.")
print("En cada orden_Pn tambien se guardan las comparativas de ordenes impares y pares para cada epsilon.")
print(f"CSV de validacion: {csv_path}")
print(f"Imagenes individuales: {len(epsilons) * len(ordenes) * 5} PNG")
print(f"Comparativas distintas: {len(epsilons) * 2 * 3} PNG")
print(f"Archivos comparativos guardados al replicarlas en cada orden_Pn: "
      f"{len(epsilons) * 2 * 3 * len(ordenes)} PNG")
print(f"Nuevas comparativas por orden fijo: {2 * len(ordenes)} PNG en {carpeta_graficas_por_orden}")
