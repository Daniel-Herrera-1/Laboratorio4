import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy import stats

# 1. Lectura de datos del archivo de texto
tiempo = []
voltaje = []

with open('emg_signal_120s_2.txt', 'r') as archivo:
    next(archivo)  # Saltar la primera línea (encabezados)
    for linea in archivo:
        t, v = linea.split()
        tiempo.append(float(t))
        voltaje.append(float(v))

# Convertir a arrays de numpy
tiempo = np.array(tiempo)
voltaje = np.array(voltaje)

# 2. Configuración de filtros
fs = 1 / np.mean(np.diff(tiempo))  # Frecuencia de muestreo


# Diseño de filtros (combinación pasa altas y pasa bajas)
def filtrado_combinado(signal, fs, low_cut=10, high_cut=150):
    """Aplica filtro pasa altas y pasa bajas combinado con verificaciones"""
    nyquist = 0.5 * fs

    # Verificar que las frecuencias de corte sean válidas
    if low_cut >= nyquist:
        raise ValueError(
            f"La frecuencia de corte baja ({low_cut}Hz) debe ser menor que la frecuencia de Nyquist ({nyquist}Hz)")
    if high_cut >= nyquist:
        high_cut = nyquist * 0.99  # Ajustamos a un valor justo debajo de Nyquist
        print(f"Advertencia: Se ajustó high_cut a {high_cut}Hz para que sea menor que Nyquist")

    # Diseño de filtros
    try:
        # Filtro pasa altas (10Hz)
        b_high, a_high = butter(4, low_cut / nyquist, btype='high')
        # Filtro pasa bajas (150Hz por defecto)
        b_low, a_low = butter(4, high_cut / nyquist, btype='low')

        # Aplicar ambos filtros en cascada
        filtered = lfilter(b_high, a_high, signal)
        filtered = lfilter(b_low, a_low, filtered)
        return filtered
    except Exception as e:
        print(f"Error en diseño de filtros: {str(e)}")
        raise


# Verificar la frecuencia de muestreo
print(f"\nFrecuencia de muestreo (fs): {fs} Hz")
print(f"Frecuencia de Nyquist: {0.5 * fs} Hz")

# Aplicar filtrado combinado con valores seguros
try:
    signal_filtrada = filtrado_combinado(voltaje, fs, low_cut=10, high_cut=150)
    print("Filtrado aplicado correctamente")
except Exception as e:
    print(f"Error al aplicar filtros: {str(e)}")
    # Si falla, intentar con valores más conservadores
    try:
        print("Intentando con frecuencias de corte más bajas...")
        signal_filtrada = filtrado_combinado(voltaje, fs, low_cut=10, high_cut=100)
    except Exception as e:
        print(f"Error crítico: {str(e)}")
        exit()

# Aplicar filtrado combinado
signal_filtrada = filtrado_combinado(voltaje, fs)

# 3. Visualización de filtrado
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(tiempo, voltaje, 'b', alpha=0.5, label='Original')
plt.plot(tiempo, signal_filtrada, 'r', label='Filtrada')
plt.title('Señal EMG original vs filtrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.legend()
plt.grid()

# Espectro de la señal filtrada
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


# 4. Aventanamiento mejorado con visualización
def aplicar_ventaneo(signal, fs, window_size=1.0, overlap=0.5, window_type='hamming', plot_ventanas=True):
    """
    Aplica ventaneo a la señal con visualización opcional

    Args:
        signal: Señal de entrada
        fs: Frecuencia de muestreo
        window_size: Tamaño de ventana en segundos
        overlap: Porcentaje de solapamiento (0-1)
        window_type: Tipo de ventana ('hamming' o 'hanning')
        plot_ventanas: Mostrar gráficos de ventanas

    Returns:
        ventanas: Lista de ventanas de señal
        tiempos_ventana: Tiempos centrales de cada ventana
    """
    samples_per_window = int(window_size * fs)
    overlap_samples = int(samples_per_window * overlap)
    step_size = samples_per_window - overlap_samples

    # Selección de ventana
    if window_type.lower() == 'hamming':
        window = np.hamming(samples_per_window)
    elif window_type.lower() == 'hanning':
        window = np.hanning(samples_per_window)
    else:
        raise ValueError("Tipo de ventana no soportado")

    ventanas = []
    tiempos_ventana = []

    # Aplicar ventaneo
    for i in range(0, len(signal) - samples_per_window + 1, step_size):
        ventana = signal[i:i + samples_per_window] * window
        ventanas.append(ventana)
        tiempos_ventana.append((i + samples_per_window / 2) / fs)  # Tiempo central

    # Visualización
    if plot_ventanas:
        # Gráfico de señal con ventanas
        plt.figure(figsize=(15, 10))

        # Señal completa con ventanas marcadas
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(len(signal)) / fs, signal, label='Señal filtrada')
        for i in range(0, len(signal) - samples_per_window + 1, step_size):
            plt.axvspan(i / fs, (i + samples_per_window) / fs, color='green', alpha=0.1)
        plt.title('Señal con ventanas de análisis')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.grid()

        # Detalle de ventanas consecutivas
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

        # Función de ventana
        plt.subplot(3, 1, 3)
        plt.plot(window, 'k', label=f'Función {window_type}')
        plt.title('Función de ventana utilizada')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

    return ventanas, tiempos_ventana


# Aplicar ventaneo a la señal filtrada
window_size = 2.0  # segundos
overlap = 0.5  # 50% de solapamiento
ventanas, tiempos_ventana = aplicar_ventaneo(
    signal_filtrada, fs, window_size, overlap, 'hamming')


# 5. Análisis espectral por ventanas
def analisis_espectral_ventanas(ventanas, fs):
    """
    Realiza análisis espectral para cada ventana

    Args:
        ventanas: Lista de ventanas de señal
        fs: Frecuencia de muestreo

    Returns:
        median_freqs: Frecuencias medianas por ventana
        mean_freqs: Frecuencias medias por ventana
        spectral_entropy: Entropía espectral por ventana
    """
    median_freqs = []
    mean_freqs = []
    spectral_entropy = []

    for ventana in ventanas:
        n = len(ventana)
        spectrum = np.abs(np.fft.fft(ventana)[:n // 2]) * 2 / n
        freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]
        psd = spectrum ** 2

        # Frecuencia mediana
        cumsum = np.cumsum(psd)
        median_freq = freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]
        median_freqs.append(median_freq)

        # Frecuencia media
        mean_freq = np.sum(freqs * psd) / np.sum(psd)
        mean_freqs.append(mean_freq)

        # Entropía espectral (indicador de fatiga)
        psd_norm = psd / np.sum(psd)
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
        spectral_entropy.append(entropy)

    return np.array(median_freqs), np.array(mean_freqs), np.array(spectral_entropy)


# Realizar análisis espectral
median_freqs, mean_freqs, spectral_entropy = analisis_espectral_ventanas(ventanas, fs)


# 6. Detección de fatiga muscular
def detectar_fatiga(tiempos, median_freqs, mean_freqs, spectral_entropy):
    """
    Analiza indicadores de fatiga y realiza prueba estadística

    Args:
        tiempos: Tiempos centrales de las ventanas
        median_freqs: Frecuencias medianas
        mean_freqs: Frecuencias medias
        spectral_entropy: Entropía espectral

    Returns:
        resultados: Diccionario con resultados del análisis
    """
    # Dividir en segmentos temprano y tardío
    split_idx = len(tiempos) // 2
    early_t = tiempos[:split_idx]
    late_t = tiempos[split_idx:]

    early_median = median_freqs[:split_idx]
    late_median = median_freqs[split_idx:]

    early_mean = mean_freqs[:split_idx]
    late_mean = mean_freqs[split_idx:]

    early_entropy = spectral_entropy[:split_idx]
    late_entropy = spectral_entropy[split_idx:]

    # Pruebas estadísticas
    t_median, p_median = stats.ttest_ind(early_median, late_median)
    t_mean, p_mean = stats.ttest_ind(early_mean, late_mean)
    t_entropy, p_entropy = stats.ttest_ind(early_entropy, late_entropy)

    # Resultados
    resultados = {
        'median_freq': {
            'early': np.mean(early_median),
            'late': np.mean(late_median),
            'change': np.mean(late_median) - np.mean(early_median),
            'p_value': p_median,
            'significant': p_median < 0.05
        },
        'mean_freq': {
            'early': np.mean(early_mean),
            'late': np.mean(late_mean),
            'change': np.mean(late_mean) - np.mean(early_mean),
            'p_value': p_mean,
            'significant': p_mean < 0.05
        },
        'spectral_entropy': {
            'early': np.mean(early_entropy),
            'late': np.mean(late_entropy),
            'change': np.mean(late_entropy) - np.mean(early_entropy),
            'p_value': p_entropy,
            'significant': p_entropy < 0.05
        }
    }

    return resultados


# Detección de fatiga
resultados_fatiga = detectar_fatiga(tiempos_ventana, median_freqs, mean_freqs, spectral_entropy)

# 7. Visualización de resultados
plt.figure(figsize=(15, 10))

# Frecuencia mediana
plt.subplot(3, 1, 1)
plt.plot(tiempos_ventana, median_freqs, 'b-o', label='Frecuencia mediana')
plt.axvline(tiempos_ventana[len(tiempos_ventana) // 2], color='r', linestyle='--',
            label='División temprano/tardío')
plt.title('Evolución de la Frecuencia Mediana')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.grid()
plt.legend()

# Frecuencia media
plt.subplot(3, 1, 2)
plt.plot(tiempos_ventana, mean_freqs, 'g-o', label='Frecuencia media')
plt.axvline(tiempos_ventana[len(tiempos_ventana) // 2], color='r', linestyle='--')
plt.title('Evolución de la Frecuencia Media')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.grid()
plt.legend()

# Entropía espectral
plt.subplot(3, 1, 3)
plt.plot(tiempos_ventana, spectral_entropy, 'm-o', label='Entropía espectral')
plt.axvline(tiempos_ventana[len(tiempos_ventana) // 2], color='r', linestyle='--')
plt.title('Evolución de la Entropía Espectral')
plt.xlabel('Tiempo [s]')
plt.ylabel('Entropía')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# 8. Reporte de resultados
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