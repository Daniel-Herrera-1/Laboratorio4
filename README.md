# Laboratorio 4 Fatiga muscular

### Este laboratorio tiene como propósito analizar el comportamiento de el musculo cuando este ces sometido a esfuerzos. Se estudiará el fenómeno de la fatiga, que ocurre debido a la aplicación repetitiva de esfuerzos

## Objetivos
● Comprender el fenómeno de fatiga en materiales sometidos a cargas cíclicas.

● Utilizar un sistema de adquisición de datos (DAQ) para registrar y analizar la actividad eléctrica del músculo.

● Analizar la influencia de factores como la amplitud del esfuerzo y la cantidad de ciclos en la resistencia a la fatiga

## Requisitos

Para ejecutar este código en tu computadora, necesitas instalar lo siguiente:

- Python 3.x (versión recomendada 3.9 o superior)
- Sistema de adquisición de datos (DAQ) para el registro de señales musculares.
- Electrodos para la medición de actividad electromiográfica (EMG).
- Objeto para inducir fatiga muscular (por ejemplo, una pelota antiestres).
- Software para análisis de datos (Excel, MATLAB o equivalente).

# Procedimiento

# *Explicacion del codigo*
```python
import nidaqmx
import time as time_lib  # Para medir el tiempo de adquisición
# Parámetros de adquisición
fs = 1000  # Frecuencia de muestreo en Hz
duracion = 120  # Duración en segundos
buffer_size = 1000  # Número de muestras a leer por iteración

# Crear tarea de adquisición
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
    task.timing.cfg_samp_clk_timing(fs, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

    # Variables para almacenar datos
    all_data = []
    start_time = time_lib.time()

    print("Adquiriendo datos...")

    while time_lib.time() - start_time < duracion:
        # Leer datos en bloques
        data = task.read(number_of_samples_per_channel=buffer_size, timeout=10)
        all_data.extend(data)  # Agregar los datos a la lista

    print("Adquisición completada.")

# Convertir datos a numpy array
all_data = np.array(all_data)

# Crear eje de tiempo
time = np.linspace(0, duracion, len(all_data))

# Guardar los datos en un archivo TXT
filename = "emg_signal_120s_3.txt"
np.savetxt(filename, np.column_stack((time, all_data)), delimiter="\t", header="Tiempo(s)\tVoltaje(V)", comments="")

print(f"Señal guardada en {filename}")
```
- Este fragmento de código adquiere y guarda datos de una señal electromiográfica (EMG) utilizando una tarjeta de adquisición de datos NI DAQ a través de la biblioteca nidaqmx
  
        nidaqmx → Permite la comunicación con dispositivos NI DAQ para la adquisición de señales.
        time_lib (renombrado de time) → Se usa para medir el tiempo de adquisición.

- fs = 1000 → Se adquirirá 1000 muestras por segundo (1 kHz).
-  duracion = 120 → La adquisición durará 120 segundos.
- uffer_size = 1000 → Se leerán bloques de 1000 muestras por iteración, lo que equivale a 1

 - ```nidaqmx.Task()``` Crea una tarea de adquisición de datos.
   
         add_ai_voltage_chan("Dev1/ai0") → Agrega un canal de voltaje analógico en la entrada "Dev1/ai0".
         cfg_samp_clk_timing(fs, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS) → Configura la frecuencia de muestreo (fs) y el modo de adquisición continua.

### Nota: "Dev1/ai0" representa la entrada analógica del dispositivo de adquisición.



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


# **APLICACION DEL VENTANEO**

```python
for i in range(0, len(signal) - samples_per_window + 1, step_size):
    ventana = signal[i:i + samples_per_window] * window
    ventanas.append(ventana)
    tiempos_ventana.append((i + samples_per_window / 2) / fs)  # Tiempo central
```
Paso a paso:

- ```for i in range(0, len(signal) - samples_per_window + 1, step_size)```

      i es la posición inicial de cada ventana.

      Se detiene antes de que la ventana exceda el tamaño de la señal.

      Se avanza en step_size, que es el tamaño de la ventana menos el solapamiento.

  - ```ventana = signal[i:i + samples_per_window] * window```

        Se extrae un segmento de la señal desde i hasta i + samples_per_window.

        Se multiplica por la función de ventana (window), que puede ser Hamming o Hanning.

- ```ventanas.append(ventana)```

       Se guarda la ventana generada en la lista ventanas.

- ```tiempos_ventana.append((i + samples_per_window / 2) / fs)```

       Se calcula el tiempo central de la ventana en segundos.

- ```(i + samples_per_window / 2) / fs da el punto medio en tiempo.```


# **Grafico de la Señal con ventanas**

```python
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(signal)) / fs, signal, label='Señal filtrada')
for i in range(0, len(signal) - samples_per_window + 1, step_size):
    plt.axvspan(i / fs, (i + samples_per_window) / fs, color='green', alpha=0.1)
plt.title('Señal con ventanas de análisis')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
```
- Señal completa

      np.arange(len(signal)) / fs convierte índices a tiempo.

      plt.plot(..., label='Señal filtrada') grafica la señal.

- Ventanas marcadas en verde:

      plt.axvspan(inicio, fin, color='green', alpha=0.1)

      alpha=0.1 hace que las ventanas sean semitransparentes.

  ![image](https://github.com/user-attachments/assets/ca0007c2-2620-4ea4-b5ef-af6456e76145)


#  **Gráfico 2: Ventanas aplicadas a la señal**

```python
plt.subplot(3, 1, 2)
num_show = min(3, len(ventanas))
colors = ['red', 'blue', 'green']
for j in range(num_show):
    inicio = j * step_size
    segmento = signal[inicio:inicio + samples_per_window] * window
    plt.plot(np.arange(samples_per_window) / fs, segmento,
             color=colors[j], label=f'Ventana {j + 1}')
plt.title('Ejemplo de ventanas aplicadas')
plt.xlabel('Tiempo en ventana [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
```
- Se seleccionan hasta 3 ventanas ```(num_show = min(3, len(ventanas)))```.

- Cada ventana se grafica con un color diferente (```red```, ```blue```, ```green```).

- ```plt.plot(..., label=f'Ventana {j + 1}')``` muestra las ventanas superpuestas.

![image](https://github.com/user-attachments/assets/72cd07ff-64ba-46e4-b16d-b35d3ea2e501)


#  **Gráfico 3: Función de Ventana Utilizada**

```python
plt.subplot(3, 1, 3)
plt.plot(window, 'k', label=f'Función {window_type}')
plt.title('Función de ventana utilizada')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
```
- Se grafica la función de ventana (Hamming o Hanning).

- ```plt.plot(window, 'k', label=...)``` muestra la curva en negro.

  ![image](https://github.com/user-attachments/assets/a7128a08-43f0-4150-955e-52b60d91bcfe)

# **Aplicar Ventaneo a Señal Filtrada**

```python
window_size = 2.0  # segundos
overlap = 0.5  # 50% de solapamiento
ventanas, tiempos_ventana = aplicar_ventaneo(
    signal_filtrada, fs, window_size, overlap, 'hamming')
```
- Se define el tamaño de ventana (window_size = 2.0 segundos).

- Se especifica un solapamiento del 50% (overlap = 0.5).

- Se llama a la función aplicar_ventaneo(), que devuelve:

      ventanas: Lista de segmentos de la señal filtrada con la ventana aplicada.

      tiempos_ventana: Lista de los tiempos centrales de cada ventana.

-Esto divide la señal en fragmentos solapados que serán analizados en el dominio de la frecuencia.

# Funcion Analisis ESPECTRAL

```python
def analisis_espectral_ventanas(ventanas, fs):
```
- ventanas: Lista de segmentos de la señal.
- fs: Frecuencia de muestreo.
  **Y devuelve tres métricas espectrales:**
  
- median_freqs: Lista de frecuencias medianas por ventana.
- mean_freqs: Lista de frecuencias medias por ventana.
- spectral_entropy: Lista de valores de entropía espectral por ventana.

# Inicialización de listas vacías
  ```python
  edian_freqs = []
mean_freqs = []
spectral_entropy = []
```
- Se crean tres listas vacías para almacenar los resultados de cada ventana.

# Cálculo del espectro con FFT

```python
n = len(ventana)
spectrum = np.abs(np.fft.fft(ventana)[:n // 2]) * 2 / n
freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]
psd = spectrum ** 2
```
- np.fft.fft(ventana): Calcula la Transformada Rápida de Fourier (FFT).

- [:n // 2]: Se queda solo con la mitad positiva del espectro.

- np.abs(... ) * 2 / n: Se normaliza la amplitud de la FFT.

- np.fft.fftfreq(n, 1 / fs)[:n // 2]: Calcula las frecuencias asociadas a la FFT.

- psd = spectrum ** 2: Se obtiene la Densidad Espectral de Potencia (PSD), que representa la distribución de energía en el dominio de la frecuencia.

# Cálculo de la frecuencia mediana

```python
cumsum = np.cumsum(psd)
median_freq = freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]
median_freqs.append(median_freq)
```
- np.cumsum(psd): Calcula la suma acumulada de la PSD.

- np.searchsorted(cumsum, cumsum[-1] / 2): Encuentra el índice donde la suma acumulada alcanza el 50% de la energía total.

- median_freqs.append(median_freq): Se guarda la frecuencia mediana.

# Calculo Frecuencia Media

```python
mean_freq = np.sum(freqs * psd) / np.sum(psd)
mean_freqs.append(mean_freq)
```
Nota: La frecuencia media indica el "centro de masa" del espectro.

- np.sum(freqs * psd) / np.sum(psd): Calcula la media ponderada de las frecuencias con la PSD.

- mean_freqs.append(mean_freq): Se guarda la frecuencia media.

# Cálculo de la entropía espectral

```python
psd_norm = psd / np.sum(psd)
entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
spectral_entropy.append(entropy)
```
- psd_norm = psd / np.sum(psd): Se normaliza la PSD para que su suma sea 1.

- np.sum(psd_norm * np.log(psd_norm + 1e-10)): Se aplica la fórmula de la entropía de Shannon.

- spectral_entropy.append(entropy): Se guarda la entropía espectral.

## La entropía espectral mide qué tan disperso está el espectro.

- Valores altos → Energía distribuida en muchas frecuencias (ruido, señal compleja).

- Valores bajos → Energía concentrada en pocas frecuencias (tonos puros).

# **Ejecucion analisis espectral**

```python
median_freqs, mean_freqs, spectral_entropy = analisis_espectral_ventanas(ventanas, fs)
```
- ```median_freqs:``` Frecuencia mediana de cada ventana.

- ```mean_freqs:``` Frecuencia media de cada ventana.

- ```spectral_entropy:``` Entropía espectral de cada ventana.

## Divison de Segmentos 
```python

split_idx = len(tiempos) // 2
early_t = tiempos[:split_idx]
late_t = tiempos[split_idx:]

```
- ```split_idx:``` Índice que divide la señal en dos mitades.
-  ```early_t:``` Contiene los tiempos centrales de las ventanas en la primera mitad del registro.
-  ```late_t:``` Contiene los tiempos centrales en la segunda mitad.


## **Extracción de métricas espectrales en cada segmento**

```python
early_median = median_freqs[:split_idx]
late_median = median_freqs[split_idx:]

early_mean = mean_freqs[:split_idx]
late_mean = mean_freqs[split_idx:]

early_entropy = spectral_entropy[:split_idx]
late_entropy = spectral_entropy[split_idx:]
```
### **se separan los valores de frecuencia mediana, media y entropía en dos conjuntos:**

- ```early_*:``` Valores en la primera mitad (sin fatiga).

- ```late_*:``` Valores en la segunda mitad (posible fatiga)


## **Pruebas estadísticas para detección de fatiga**
```pythom
t_median, p_median = stats.ttest_ind(early_median, late_median)
t_mean, p_mean = stats.ttest_ind(early_mean, late_mean)
t_entropy, p_entropy = stats.ttest_ind(early_entropy, late_entropy)
```

- t_*: Estadístico de la prueba (magnitud de la diferencia).

- p_*: Valor p (indica si la diferencia es significativa).


## Cálculo de los valores medios y cambio absoluto
```python
'median_freq': {
    'early': np.mean(early_median),
    'late': np.mean(late_median),
    'change': np.mean(late_median) - np.mean(early_median),
```
- early: Promedio de la frecuencia mediana en la primera mitad de la señal.

- late: Promedio en la segunda mitad.

- change: Diferencia entre el valor tardío y el temprano.

  **Nota1: change negativo → Disminución de la frecuencia mediana → Fatiga muscular presente.**

  ```python
  'mean_freq': {
    'early': np.mean(early_mean),
    'late': np.mean(late_mean),
    'change': np.mean(late_mean) - np.mean(early_mean),
  ```
**Nota2: change negativo → Disminución de la frecuencia media → Indicación de fatiga.**


```python
'spectral_entropy': {
    'early': np.mean(early_entropy),
    'late': np.mean(late_entropy),
    'change': np.mean(late_entropy) - np.mean(early_entropy),
```
**Nota3: change positivo → Aumento de la entropía → Mayor aleatoriedad en la señal → Indicio de fatiga.**


## Evaluacion estadistica

```python
'p_value': p_median,
'significant': p_median < 0.05
```

- ``` Si significant: True```, la disminución de la frecuencia mediana es estadísticamente significativa → evidencia de fatiga.

- ```Si significant: False```, no hay suficiente evidencia estadística para afirmar la presencia de fatiga.


# **Detección de Fatiga**

```python
resultados_fatiga = detectar_fatiga(tiempos_ventana, median_freqs, mean_freqs, spectral_entropy)
```
- Llama a la función detectar_fatiga para analizar la evolución de las características espectrales y determinar si hay fatiga muscular.
  
      Entrada

      - tiempos_ventana: Lista con los tiempos centrales de cada ventana de análisis.

       - median_freqs: Frecuencia mediana de cada ventana.

      - mean_freqs: Frecuencia media de cada ventana.

      - spectral_entropy: Entropía espectral de cada ventana.
  
      Salida:

      - resultados_fatiga: Diccionario con los valores promedio temprano/tardío, cambios y significancia estadística.


  # **Visualizacion Resultados**

  ### Gráfico 1: Evolución de la Frecuencia Mediana

  ```python
  plt.subplot(3, 1, 1)
  plt.plot(tiempos_ventana, median_freqs, 'b-o', label='Frecuencia mediana')
  plt.axvline(tiempos_ventana[len(tiempos_ventana) // 2], color='r', linestyle='--', label='División temprano/tardío')
  plt.title('Evolución de la Frecuencia Mediana')
   plt.xlabel('Tiempo [s]')
  plt.ylabel('Frecuencia [Hz]')
  plt.grid()
  plt.legend()
  ```
![image](https://github.com/user-attachments/assets/85e44cb9-98be-4b4d-a8de-3393f10a97c4)

-  Una disminución en la frecuencia mediana en la segunda mitad del tiempo sugiere fatiga muscular.

  ### Gráfico 2: Evolución de la Frecuencia Media

  ```python
plt.subplot(3, 1, 2)
plt.plot(tiempos_ventana, mean_freqs, 'g-o', label='Frecuencia media')
plt.axvline(tiempos_ventana[len(tiempos_ventana) // 2], color='r', linestyle='--')
plt.title('Evolución de la Frecuencia Media')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.grid()
plt.legend()

```
![image](https://github.com/user-attachments/assets/d05a472b-361e-422a-8885-745bf168c606)

- Si la frecuencia media disminuye con el tiempo, es una señal de fatiga.


### Gráfico 3: Evolución de la Entropía Espectral

```python
plt.subplot(3, 1, 3)
plt.plot(tiempos_ventana, spectral_entropy, 'm-o', label='Entropía espectral')
plt.axvline(tiempos_ventana[len(tiempos_ventana) // 2], color='r', linestyle='--')
plt.title('Evolución de la Entropía Espectral')
plt.xlabel('Tiempo [s]')
plt.ylabel('Entropía')
plt.grid()
plt.legend()

```
![image](https://github.com/user-attachments/assets/d734bca4-e28a-4900-b7d1-be37dff459c6)

- Si la entropía espectral aumenta con el tiempo, indica mayor aleatoriedad y pérdida de estructura en la señal → posible fatiga muscular.

# **Reporte de Resultados**

```python
print("\n=== RESULTADOS DEL ANÁLISIS DE FATIGA ===")
print(f"\nFrecuencia Mediana:")
print(f"  Temprano: {resultados_fatiga['median_freq']['early']:.2f} Hz")
print(f"  Tardío: {resultados_fatiga['median_freq']['late']:.2f} Hz")
print(f"  Cambio: {resultados_fatiga['median_freq']['change']:.2f} Hz")
print(
    f"  Significativo (p<0.05): {'Sí' if resultados_fatiga['median_freq']['significant'] else 'No'} (p={resultados_fatiga['median_freq']['p_value']:.4f})")

print(f"\nFrecuencia Media:")
print(f"  Temprano: {resultados_fatiga['mean_freq']['early']:.2f} Hz")
print(f"  Tardío: {resultados_fatiga['mean_freq']['late']:.2f} Hz")
print(f"  Cambio: {resultados_fatiga['mean_freq']['change']:.2f} Hz")
print(
    f"  Significativo (p<0.05): {'Sí' if resultados_fatiga['mean_freq']['significant'] else 'No'} (p={resultados_fatiga['mean_freq']['p_value']:.4f})")

print(f"\nEntropía Espectral:")
print(f"  Temprano: {resultados_fatiga['spectral_entropy']['early']:.2f}")
print(f"  Tardío: {resultados_fatiga['spectral_entropy']['late']:.2f}")
print(f"  Cambio: {resultados_fatiga['spectral_entropy']['change']:.2f}")
print(
    f"  Significativo (p<0.05): {'Sí' if resultados_fatiga['spectral_entropy']['significant'] else 'No'} (p={resultados_fatiga['spectral_entropy']['p_value']:.4f})")

# Interpretación
if resultados_fatiga['median_freq']['significant'] and resultados_fatiga['median_freq']['change'] < 0:
    print("\nCONCLUSIÓN: Se detectó fatiga muscular significativa (disminución en frecuencia mediana)")
elif resultados_fatiga['mean_freq']['significant'] and resultados_fatiga['mean_freq']['change'] < 0:
    print("\nCONCLUSIÓN: Se detectó fatiga muscular (disminución en frecuencia media)")
else:
    print("\nCONCLUSIÓN: No se detectó fatiga muscular significativa")
```
    === RESULTADOS DEL ANÁLISIS DE FATIGA ===
    
    Frecuencia Mediana:
      Temprano: 61.89 Hz
      Tardío: 61.99 Hz
      Cambio: 0.10 Hz
      Significativo (p<0.05): No (p=0.9674)
    
    Frecuencia Media:
      Temprano: 70.60 Hz
      Tardío: 70.99 Hz
      Cambio: 0.39 Hz
      Significativo (p<0.05): No (p=0.7845)
    
    Entropía Espectral:
      Temprano: 5.24
      Tardío: 5.23
      Cambio: -0.01
      Significativo (p<0.05): No (p=0.6972)
    
    CONCLUSIÓN: No se detectó fatiga muscular significativa

## Este bloque de código imprime un reporte detallado del análisis de fatiga muscular con base en tres métricas clave: frecuencia mediana, frecuencia media y entropía espectral. 

-  Interpretación de los resultados:

Si la frecuencia mediana disminuyó significativamente → Fatiga detectada.

Si la frecuencia media disminuyó significativamente → Fatiga probable.

Si ninguna métrica muestra cambios significativos → No hay fatiga detec




  

 









 






  
