# FILTRO RC de tres etapas- en frecuencia
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from fractions import Fraction
import random as rnd
# Tiempos de covarianza a comparar
tc_values = [4/5,3/5,2/5]
N =11
deltaT =0.3
t0 = np.linspace(-5, 5, 1000)
TI = np.array([deltaT * i for i in range(1, N + 1)]) # para centrar la primer muestra en deltaT y un muestreo uniforme
#Tmin, Tmax = 0.01, 3 #muestreo no uniforme
#TI = np.sort(np.random.uniform(Tmin, Tmax, N))
print("INICIO")
print("Tiempos de muestreo:", TI)

# Colores para cada tiempo de covarianza
colors = ['black', 'orange', 'red']
line_styles = ['-', '--', '-.']

# Para cada función básica
for j in range(N):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Para cada tiempo de covarianza
    for idx, tc in enumerate(tc_values):
        alfa = 8 / (3 * tc)  # para cada tc


        # Función de covarianza
        def covarianza(t1, t2, alfa):
            return (1 + (alfa * np.abs(t1 - t2)) + (alfa ** 2 * (t1 - t2) ** 2) / 3) * np.exp(-alfa * np.abs(t1 - t2))


        # Matriz de covarianza
        kx = np.zeros((N, N))
        for i in range(N):
            for k in range(N):
                kx[i, k] = covarianza(TI[i], TI[k], alfa)

        # Matriz inversa
        a = inv(kx)

        print("matriz de covarianza")
        print(kx)
        print("Matriz inversa de covarianza")
        print(a)
        # Espectro
        f0 = np.linspace(-3, 3, 10000)
        w0 = 2 * np.pi * f0
        sx = (16 * alfa ** 5) / (alfa ** 2 + w0 ** 2) ** 3


        # Función básica en frecuencia
        def funcion_basica_fre(w0, j, TI, a, sx, N):
            return sx * sum(a[i, j] * np.exp(-1j * w0 * TI[i]) for i in range(N))


        Bj = funcion_basica_fre(w0, j, TI, a, sx, N)
        # Graficas
        # magnitud
        tc_fraccion = [str(Fraction(tc).limit_denominator()) for tc in tc_values]
        alfa_fraccion = [str(Fraction(alfa).limit_denominator()) for tc in tc_values]
        ax1.plot(w0, np.abs(Bj), color=colors[idx], linestyle=line_styles[idx],
                 linewidth=1.5, label=f'tc={tc_fraccion[idx]}   α={alfa_fraccion[idx]}')
        # fase
        ax2.plot(w0,(np.unwrap(np.angle(Bj))), color=colors[idx],
                 linestyle=line_styles[idx], linewidth=1.5, label=f'tc={tc_fraccion[idx]}')

    # TITULOS
    ax1.set_title('Magnitud')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('w [rad/s]')
    ax1.set_ylabel('|B(jw)|')
    ax1.legend(loc='best')
    # TITULOS-FASE
    ax2.set_title('Fase')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('w [rad/s]')
    ax2.set_ylabel('fase [rad]')
    ax2.legend(loc='best')
    plt.suptitle(f'Funciones básicas B_{j + 1}(f)')
    plt.tight_layout()
    plt.show()