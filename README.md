# Laboratorio4

# Laboratorio 4 Fatiga muscular

### Este laboratorio tiene como propósito analizar el comportamiento de los materiales cuando son sometidos a cargas cíclicas. Se estudiará el fenómeno de la fatiga, que ocurre debido a la aplicación repetitiva de esfuerzos, lo que puede llevar a la falla del material sin necesidad de alcanzar su resistencia última

## Objetivos
● Comprender el fenómeno de fatiga en materiales sometidos a cargas cíclicas.

● Utilizar un sistema de adquisición de datos (DAQ) para registrar y analizar la actividad eléctrica del músculo.

● Analizar la influencia de factores como la amplitud del esfuerzo y la cantidad de ciclos en la resistencia a la fatiga

## Requisitos

Para ejecutar este código en tu computadora, necesitas instalar lo siguiente:

- Python 3.x (versión recomendada 3.9 o superior)
- Sistema de adquisición de datos (DAQ) para el registro de señales musculares.
- Electrodos para la medición de actividad electromiográfica (EMG).
- Objeto para inducir fatiga muscular (por ejemplo, una pelota de estrés).
- Software para análisis de datos (Excel, MATLAB o equivalente).

# Procedimiento

# *1.Explicacion del codigo*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy import stats
```

- **numpy (np):** Biblioteca utilizada para trabajar con arreglos de números y realizar cálculos matemáticos avanzados de manera eficiente.

- **matplotlib.pyplot (plt):** Herramienta para generar gráficos que permiten visualizar datos de forma clara.

- **scipy.signal**  Implementación de filtros digitales (butter, lfilter, freqz).Net.
- **scipy.stats** Análisis estadístico de señales.

# *Lectura de datos del archivo de texto*

```python
tiempo = []
voltaje = []
with open('emg_signal_120s_2.txt', 'r') as archivo:
```
- Se crean dos listas vacías donde guardaremos los datos.

- ```tiempo``` almacenará los valores de la primera columna del archivo.

- ```voltaje``` almacenará los valores de la segunda columna.
- ```archivo ``` será el identificador con el que accederemos a su contenido.


```python
next(archivo)  # Saltar la primera línea (encabezados)
for linea in archivo:
    t, v = linea.split()
    tiempo.append(float(t))
    voltaje.append(float(v))
```
- La primera línea del archivo suele contener los títulos de las columnas,entonces ```next(archivo)``` ignora esa línea y salta directamente a los valores numéricos.
- ```for linea in archivo``` Recorre el archivo línea por línea.
- ```linea.split()```Separa la línea en dos valores: t(tiempo) y v(voltaje)
- ```float(t) y float(v) ``` Convierte los valores en números reales (float).
- ``` tiempo.append(float(t)) y voltaje.append(float(v))```Guarda los valores en sus respectivas listas.


```python
tiempo = np.array(tiempo)
voltaje = np.array(voltaje)
```
- Nota: Las listas en Python son más lentas para cálculos numéricos.
- NumPy permite hacer operaciones matemáticas de manera más rápida y eficiente.

 # *Configuracion de Filtros*

 ```python
fs = 1 / np.mean(np.diff(tiempo))  # Frecuencia de muestreo
def filtrado_combinado(signal, fs, low_cut=10, high_cut=150):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
  if low_cut >= nyquist:
        raise ValueError(f"La frecuencia de corte baja ({low_cut}Hz) debe ser menor que Nyquist ({nyquist}Hz)")
```
-  ```np.diff(tiempo): ``` Calcula la diferencia entre cada par de muestras de tiempo consecutivas, es decir, obtiene el intervalo entre muestras.

-  ```np.mean(...): ``` Calcula el promedio de estos intervalos.

-  ```1 / ...: ``` Al tomar el inverso, se obtiene la frecuencia de muestreo fs, que indica cuántas muestras por segundo tiene la señal.
- **La frecuencia de Nyquist es la mitad de la frecuencia de muestreo y representa el límite superior de frecuencias que pueden analizarse sin aliasing.**
- ```low_cut``` es la frecuencia de corte del filtro pasa-altas. Si es mayor o igual a Nyquist, el filtro no tiene sentido.

```python
 if high_cut >= nyquist:
        high_cut = nyquist * 0.99  # Ajuste para evitar aliasing
        print(f"Advertencia: Se ajustó high_cut a {high_cut}Hz")
```
- ```high_cut``` es la frecuencia de corte del filtro pasa-bajas. Si supera Nyquist, se ajusta automáticamente a 0.99 * nyquist para evitar aliasing.

# *Diseño de los filtros*

```python
try:
        b_high, a_high = butter(4, low_cut / nyquist, btype='high')
        b_low, a_low = butter(4, high_cut / nyquist, btype='low')
filtered = lfilter(b_high, a_high, signal)
        filtered = lfilter(b_low, a_low, filtered)
  return filtered
    except Exception as e:
        print(f"Error en diseño de filtros: {str(e)}")
        raise
```
- ```butter(4, ...)``` crea un filtro Butterworth de orden 4, que ofrece una respuesta suave sin oscilaciones.

- Se normalizan las frecuencias de corte dividiéndolas entre nyquist, ya que ```butter()``` trabaja con valores entre 0 y 1.

- ```btype='high':``` Diseña un filtro pasa-altas con low_cut Hz.

- ```btype='low':``` Diseña un filtro pasa-bajas con high_cut Hz.

- ```lfilter(b, a, signal):``` Aplica el filtro digital definido por b y a a la señal.

- Primero se aplica el pasa-altas para eliminar frecuencias bajas no deseadas.

- Luego se aplica el pasa-bajas para eliminar frecuencias altas no deseadas.


# *Visualizacion de los Resultados (Grafica)*

Visualizacion del Filtrado
```python
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(tiempo, voltaje, 'b', alpha=0.5, label='Original')
plt.plot(tiempo, signal_filtrada, 'r', label='Filtrada')
plt.title('Señal EMG original vs filtrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.legend()
plt.grid()
```
Visualizacion Espectro de la señal Filtrada

```python
N = len(signal_filtrada)
frequencies = np.fft.fftfreq(N, 1 / fs)[:N // 2]
spectrum = np.abs(np.fft.fft(signal_filtrada)[:N // 2]) * 2 / N

plt.subplot(1, 2, 2)
plt.semilogy(frequencies, spectrum)
plt.title('Espectro de la señal filtrada')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.grid()
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/451656b5-2075-4822-b488-504222ebca9c)

## **Panel izquierdo: Señal EMG original vs. filtrada**
- Interpretación: Se observa que la señal filtrada mantiene la estructura general de la original, pero con menor ruido y variabilidad extrema. Esto sugiere que se ha aplicado un filtro, probablemente un paso banda, que conserva la información útil mientras elimina interferencias.

## **Panel derecho: Espectro de la señal filtrada**

-  La energía de la señal se concentra en frecuencias bajas y medias, mientras que en las altas frecuencias la magnitud disminuye considerablemente. Esto indica que el filtrado ha eliminado componentes de alta frecuencia no deseadas.


 # *Aventanamiento*

 ```python
def aplicar_ventaneo(signal, fs, window_size=1.0, overlap=0.5, window_type='hamming', plot_ventanas=True):

```
- signal: La señal de entrada a la que se le aplicará el ventaneo.

- fs: La frecuencia de muestreo de la señal, necesaria para convertir segundos a muestras.

- window_size: Tamaño de la ventana en segundos (por defecto, 1.0 s).

- overlap: Porcentaje de solapamiento entre ventanas, un valor entre 0 y 1 (por defecto, 50%).

- window_type: Tipo de ventana a aplicar ('hamming' por defecto).

- plot_ventanas: Un booleano para decidir si se grafican las ventanas.


## **Cálculo del Tamaño de Ventana y Paso*

```python
samples_per_window = int(window_size * fs)
overlap_samples = int(samples_per_window * overlap)
step_size = samples_per_window - overlap_samples
```
-  ```samples_per_window: ```Convierte el tamaño de la ventana de segundos a muestras.

- ```overlap_samples: ``` Calcula cuántas muestras de la ventana anterior se reutilizan.

- ```step_size: ``` Número de muestras que se avanza en cada ventana.

## **Seleccion Tipo de Ventana*

```python
if window_type.lower() == 'hamming':
    window = np.hamming(samples_per_window)
elif window_type.lower() == 'hanning':
    window = np.hanning(samples_per_window)
else:
    raise ValueError("Tipo de ventana no soportado")
```
- se usa ```np.hamming()``` o ```np.hanning()``` según el tipo de ventana:

      Ventana de Hamming: Reduce los efectos de fugas espectrales mejor que la rectangular.

      Ventana de Hanning: Similar a la de Hamming pero con más atenuación en los extremos.

- ```lower():``` Asegura que el usuario pueda escribir "Hamming" o "hamming" sin error.

**Inicialización de Listas para Guardar los Resultados**

```python
ventanas = []
tiempos_ventana = []
```

- ventanas: Lista donde se almacenarán los segmentos de la señal.

- tiempos_ventana: Lista con los tiempos centrales de cada ventana.


 






  
