#FILTRO RC DE UNA ETAPA estudio en frecuencia
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
tc=1#tiempo de covarianza
alfa=1/tc
N=10
delta=0.2
TI=np.array([delta*i for i in range(N)])#para un muestreo uniforme que varía con el valor de delta
def convarianza(t1,t2):
    return np.exp(-alfa * np.abs(t1 - t2))
kt=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        kt[i,j]=convarianza(TI[i],TI[j])
print("matriz de covarianza")
print(kt)
print("matriz inversa")
a=inv(kt)
print(a)
#ESTUDIO EN FRECUENCIA
# Se define que la fución básica en terminos de la frecuencia como
#B(jw)=suma, desde i=1, hasta i=N (funcion de coriazna en frecuencia*aij)
#donde la función de covarianza en la frecuencia es igual a Rx(t-TI)=Sx(w)*e^-2j*pi*f*(t-TI)
w0=np.linspace(-100,100,10000)#vector frecuencia- para verlo en frecuencia, cambiar w0 por f0 y descomentar w0 y editar los plots
#w0=2*np.pi*f0 #vector frecuencia angular
sx=(2*alfa)/(w0**2+alfa**2)#espectro del filtro RC de una etapa normalizado

def funcion_basica_freq(w, j, TI, a, sx):
    B = np.zeros_like(w, dtype=complex)
    for i in range(N):
        B += sx * a[i, j] * np.exp(-1j * w * TI[i])
    return B


for j in range(N):
    Bj = funcion_basica_freq(w0, j, TI, a, sx)
    plt.subplot(1,2,1)
    plt.plot(w0, np.abs(Bj))
    plt.title('Magnitud')
    plt.grid(True)
    plt.xlabel('f [RAD/SEG]')
    plt.ylabel('|B(f)|')

    plt.subplot(1,2,2)
    plt.plot(w0, np.unwrap(np.angle(Bj)))
    plt.title('Fase')
    plt.grid(True)
    plt.xlabel('f [RAD/SEG]')
    plt.ylabel('fase [rad]')
    plt.suptitle(f'Función básica B_{j+1}(f)')
    plt.show()

