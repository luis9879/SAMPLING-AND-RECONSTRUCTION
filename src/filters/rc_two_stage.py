from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
###########################################
#FILTRO RC DE DOS ETAPAS ESTUDIOO EN FRECUENCIA
#para un filtro RC de dos etapas se definen las siguientes funciones
#La función de covarianza normalizada R(x)=(1-alfa*(valor_absoluto(tau)))e^-alfa*(valor_absoluto(tau))
#la funcion de densidad espectral sx(w)=4*alfa^3/(alfa^2+w^2)^2     la cual tiene como máximo 4/alfa
N=20
t_c=2#tiempo de covarianza
alfa=2/(t_c)
deltat=1#espacio entre las muestras
TI=np.array([deltat*i for i in range(0,N+1)])#para un muestreo uniforme
N=len(TI)
print(TI)
muestra = np.array([1,5,1,3])
def covarianza(t1,t2):
    return (1 + alfa * np.abs(t1 - t2))*np.exp(-alfa * np.abs(t1 - t2))

kv=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        kv[i,j]=covarianza(TI[i],TI[j])

print("MATRIZ DE COVARIANZA")
print(kv)
print("MATRIZ INVERSA DE COVARIANZA")
a=inv(kv)
print(a)

########################################################################
#Estudio em frecuencia
#la funcion de densidad espectral sx(w)=4*alfa^3/(alfa^2+w^2)^2     la cual tiene como máximo 4/alfa
#por lo tanto se define sx(w)
f0=np.linspace(-alfa*3,alfa*3,1000)
w0=2*np.pi*f0
sx=(4*alfa**3)/(alfa**2+w0**2)**2
#la función básica en la frecuencia esta dada por
def funcion_basica_freq(w,j,TI,a,sx):
    return sx * sum(
        a[i,j] * np.exp(-1j * w * TI[i])
        for i in range(N)
    )

for j in range(N):
    Bj = funcion_basica_freq(w0,j,TI, a,sx)
    plt.subplot(1,2,1)
    plt.plot(w0, np.abs(Bj))
    plt.title('Magnitud')
    plt.grid(True)
    plt.xlabel('w [rad/s]')
    plt.ylabel('|B(f)|')
    plt.subplot(1,2,2)
    plt.plot(w0, np.unwrap(np.angle(Bj)))
    plt.title('Fase')
    plt.grid(True)
    plt.xlabel('w [RA/SEG]')
    plt.ylabel('fase [rad]')
    plt.suptitle(f'Función básica B_{j+1}(f)')
    plt.show()

