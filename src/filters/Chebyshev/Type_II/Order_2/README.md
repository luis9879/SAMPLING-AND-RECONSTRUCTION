# Chebyshev Type II — Order 2
## 1. Descripción
Este directorio contiene programas para el análisis de un filtro Chebyshev tipo II de segundo orden mediante muestreo y reconstrucción de funciones asociadas a su caracterización estadística.
Se estudia la influencia de los parámetros del filtro y del intervalo de muestreo sobre la reconstrucción, tanto en el dominio del tiempo como en el dominio de la frecuencia.
### 2. Análisis en frecuencia
El análisis frecuencial permite estudiar la distribución espectral de las funciones básicas y evaluar las diferencias entre la representación original y la reconstruida.
Las integrales utilizadas para calcular magnitudes espectrales se aproximan numéricamente sobre una malla de frecuencias finita. Por tanto, los resultados dependen tanto de las propiedades del filtro como de la resolución y extensión de dicha malla.
## 3. Programas
- `CHEBYSHEV_Familia_graficas.py`: genera las gráficas correspondientes a la familia de funciones estudiada para un único vvalor de rizo definido.
- `CHEBYSHEV_MANUAL.py`: realiza el análisis de las funciones básicas y su comportamiento en tiempo y frecuencia, es la referencia de los demás programas.
- `CHEBYSHEV_DISTINTOS_EPSILONS_deltaT.py`: evalúa diferentes valores de epsilon e intervalos de muestreo, organiza los resultados y permite comparar los errores de reconstrucción.
Los nombres y las descripciones deben mantenerse sincronizados con las funciones que realmente implementa cada programa.
## 4. Parámetros de simulación
Los parámetros utilizados deben consultarse directamente en cada script para garantizar la reproducibilidad.
| Parámetro | Descripción |
|---|---|
| Orden del filtro | Segundo orden. |
| Tipo de filtro | Chebyshev tipo II. |
| epsilon | Parámetro para definir el rizo en la banda de rechazo|
| \(\Delta T\) | Intervalo de muestreo. |
| \(N\) | Número de muestras empleadas en la reconstrucción. |
| Malla de frecuencias | Intervalo y resolución utilizados para aproximar las integrales espectrales. |
| \(\omega_c\) | Frecuencia de corte o frecuencia característica, según la definición empleada en el programa. |
Los valores concretos de estos parámetros, así como los criterios de cálculo del error, deben registrarse a partir de la configuración efectiva de cada ejecución.
## 5. Limitaciones de la malla de integración
La integración numérica en frecuencia se realiza sobre un intervalo finito y con una resolución determinada. Esto introduce dos fuentes principales de error.
### 5.1. Extensión insuficiente de la malla
Si la frecuencia de interés \(\omega_c\) supera la frecuencia máxima de la malla, la integración no cubre el intervalo necesario para evaluar adecuadamente la contribución espectral.
En estas condiciones, las integrales pueden quedar truncadas y las estimaciones de energía, errores o magnitudes espectrales pueden resultar sesgadas.
Por ello, debe verificarse que:
\[
\omega_c\leq\omega_{\max},
\]
donde \(\omega_{\max}\) es la frecuencia máxima considerada en la integración.
Esta condición es necesaria, pero no garantiza por sí sola la precisión del resultado.
### 5.2. Resolución insuficiente
Aunque la malla cubra la frecuencia de interés, una separación excesiva entre puntos puede impedir representar adecuadamente las variaciones de las funciones integradas.
Es necesario comprobar la convergencia numérica mediante el refinamiento de la malla y la comparación de los resultados obtenidos con diferentes resoluciones.
## 6. Resultados
La carpeta `Chevyshev_resultados/` contiene los resultados generados durante las simulaciones.
Estos archivos permiten examinar la variación de los errores de reconstrucción y las características de las funciones analizadas bajo diferentes configuraciones de los parámetros.
## 7. Reproducibilidad
Para reproducir los experimentos:
1. Instalar las dependencias de Python requeridas por el proyecto.
2. Verificar los parámetros configurados en el programa que se desea ejecutar.
3. Ejecutar el script correspondiente.
4. Confirmar que la malla de integración cubre el intervalo de frecuencias requerido.
## 8. Alcance
El objetivo es caracterizar la relación entre los parámetros del filtro, el muestreo, la reconstrucción y los errores numéricos obtenidos en los dominios temporal y frecuencial con relación al procedimiento de muestreo y reconstrucción mediante la regla de la esperanza matemática condicional (REMC) para filtros lineales de orden n, el análisis dual tiempo-frecuencia.
Mediante los programas realizados se relacionará con la dependencia estadística a partir de la estructura del filtro y una entrada, como la estrcutura módifica esta pero también la influencia estadística toma 
ciertas características y con ello formular algoritmos cuasi-óptimos con los errores deseados o como la estructura y laa entrada módifican el error de reconstrucción.
