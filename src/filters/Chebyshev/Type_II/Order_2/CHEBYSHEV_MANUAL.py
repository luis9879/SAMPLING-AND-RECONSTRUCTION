import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#Parámetros 
N=9
deltaT=0.5
w0 = np.linspace(-50,50, 1001)#Vector de frecuencias
t0 = np.linspace(-5,5, 1001) #Vector de tiempos para la integración
P=1 #orden del filtro
TI =deltaT*np.arange(N)
print("Tiempos de muestreo:", TI)
epsilon=0.5 #amplitud del rizo
#Para el filtro de Chebyshev tipo II se tiene
#Calculo del espectro
def ChebyshevII_espectro(wc, w0, epsilon, P):
    espectro = np.ones_like(w0, dtype=float)
    for i, w in enumerate(w0):
        w_abs = abs(w)
        if w_abs == 0:
            espectro[i] = 1.0
        else:
            x = wc / w_abs
            if x <= 1:
                cp = np.cos(P * np.arccos(x))
            else:
                cp = np.cosh(P * np.arccosh(x))
            e2 = (epsilon * cp)**2
            espectro[i] = e2 / (1 + e2)
    return espectro

wc=4.14#estaa parte se debe ajustar a los valores definidos en la tesis, cuando se definan todos los ordenes se debe ajustaar a los valores de frecuencia d ecorte en la tesis
S_w = ChebyshevII_espectro(wc, w0, epsilon, P)
#calculo de la función de covarianza mediante integraación numerica

# Calculo de la funcion de covarianza mediante integracion numerica
def K_tau(tau):
    integrando = S_w * np.cos(w0 * tau)
    return (1 / (2 * np.pi)) * np.trapezoid(integrando, x=w0)
K = np.zeros_like(t0)
for i, tau in enumerate(t0):
    K[i] = K_tau(tau)
print("Funcion de covarianza:")
print(K)
#gráficos del espectro y funcion de covarainza
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

#Graáfico del espectro
ax1.plot(w0, S_w, color='blue')
ax1.set_title('Espectro del filtro de Chebyshev tipo II')
ax1.set_xlabel('Frecuencia (rad/s)')
ax1.set_ylabel('Magnitud')
ax1.grid(True)
#Gráfico de la función de covarianza
ax2.plot(t0, K, color='orange')
ax2.set_title('Función de covarianza')
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Covarianza')
ax2.grid(True)
#Calculo de la matriz de covarianza y su inversa para cada función básica
kx=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        kx[i,j]=K_tau(TI[i]-TI[j])
print("Matriz de funciones de covarianza \n",kx)
a=inv(kx)
print("Matriz inversa de funciones de covarianza \n",a)
#Definición de la función básica en el tiempo
def funcion_basica(t,TI,a,j):
    return sum(
        K_tau(t - TI[i]) * a[i, j]
        for i in range(len(TI))
    )
plt.figure()
for j in range(N):
    funcion_tiempo = [funcion_basica(t, TI, a, j) for t in t0]
    plt.plot(t0,funcion_tiempo, label=f'B_{j}(t)')
plt.grid(True)
plt.legend()
plt.xlabel("tiempo[seg]")
plt.ylabel("funcion Básica B(t)")
#Calculos en la frecuencia 
########################################################################
#Estudio em frecuencia
#la funcion de densidad espectral sx(w)=4*alfa^3/(alfa^2+w^2)^2     la cual tiene como máximo 4/alfa
#por lo tanto se define sx(w)
sx=S_w
#la función básica en la frecuencia esta dada por
def funcion_basica_freq(w,j,TI,a,sx):
    return sx * sum(
        a[i,j] * np.exp(-1j * w * TI[i])
        for i in range(N)
    )
#Retardo de grupo
def group_delay_analitico(a, w0, Ti):
    N_  = len(Ti)
    n_w = len(w0)
    # Diferencias de tiempo entre muestras: dT[i,l] = Ti[i] - Ti[l]
    dT = Ti[:, None] - Ti[None, :]                          # (N, N)
    # cos(ω · dT) para todos los ω a la vez
    cos_term = np.cos(dT[:, :, None] * w0[None, None, :]) 
    gd = np.zeros((N_, n_w))
    for k in range(N_):
        # Producto exterior de la columna k de a
        A = np.outer(a[:, k], a[:, k])                     
        # Pesos ponderados: A[i,l] · cos(ω·(Ti[i]-Ti[l]))
        weighted = A[:, :, None] * cos_term                 
        # Numerador:   Σ_il Ti[i] · A[i,l] · cos(...)
        num = np.sum(Ti[:, None, None] * weighted, axis=(0, 1))
        # Denominador: Σ_il A[i,l] · cos(...)
        den = np.sum(weighted, axis=(0, 1))
        # para numeradores nulos
        gd[k] = np.where(np.abs(den) > 1e-12, num / den, np.nan)

    return gd
#Una figura por cada B_j en frecuencia
for j in range(N):
    Bj = funcion_basica_freq(w0, j, TI, a, sx)
    fig_f, (ax_m, ax_p, ax_gd) = plt.subplots(1, 3, figsize=(18, 5))
    ax_m.plot(w0, np.abs(Bj))
    ax_m.set_title('Magnitud')
    ax_m.set_xlabel('w [rad/s]')
    ax_m.set_ylabel('|B(w)|')
    ax_m.grid(True)
    ax_p.plot(w0, np.unwrap(np.angle(Bj)))
    ax_p.set_title('Fase')
    ax_p.set_xlabel('w [rad/s]')
    ax_p.set_ylabel('fase [rad]')
    ax_p.grid(True)
    ax_gd.plot(w0, group_delay_analitico(a, w0, TI)[j])
    ax_gd.set_title('Retardo de grupo')
    ax_gd.set_xlabel('w [rad/s]')
    ax_gd.set_ylabel('τ_g [s]')
    ax_gd.grid(True)
    fig_f.suptitle(f'Función básica B_{j+1}(w)')
    fig_f.tight_layout()
#Pruebas

#Plots
plt.show()