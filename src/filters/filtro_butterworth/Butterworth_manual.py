#Programa para el calculo de la función de covarianza cpara el filtro butterworth de orden n considerando un T_c unitario con las formulas del maestro lozada 
#y otras determinadas manualmente.
import numpy as np
import matplotlib.pyplot as plt

#Definiciones y calculos
n=5 #orden del filtro
w_c=n*np.sin(np.pi/(2*n)) #frecuencia de corte para un T_c=1
tetta=np.pi*(2*np.arange(1,n+1)-1)/(2*n) #angulos de los polos del filtro butterworth
s_k=w_c*(-np.sin(tetta)+1j*np.cos(tetta)) #polos del filtro butterworth
w=np.linspace(-10,10,1000) #vector de frecuencias para el calculo de la respuesta en frecuencia
tao=np.linspace(-10,10,1000) #vector de tiempos para el calculo de la funcion de covarianza
s_n=(2*w_c**2)/(w**2+w_c**2) #polo del filtro butterworth de orden n=1
# Funcion de covarianza
R_x = (-1 / n) * np.sum(
    s_k[:, np.newaxis] * np.exp(s_k[:, np.newaxis] * np.abs(tao)),
    axis=0,
)
# Densidad espectral de potencia bilateral
S_x = (2 * w_c ** (2 * n)) / (w ** (2 * n) + w_c ** (2 * n))

#plots
# Graficas
fig, (ax_cov, ax_psd) = plt.subplots(2, 1, figsize=(9, 7))

ax_cov.plot(tao, R_x.real)
ax_cov.set_title("Funcion de covarianza")
ax_cov.set_xlabel("Retardo, tao")
ax_cov.set_ylabel("R_x(tao)")
ax_cov.grid(True)

ax_psd.plot(w, S_x)
ax_psd.set_title("Densidad espectral de potencia")
ax_psd.set_xlabel("Frecuencia angular, w")
ax_psd.set_ylabel("S_x(w)")
ax_psd.grid(True)

fig.tight_layout()
plt.show()