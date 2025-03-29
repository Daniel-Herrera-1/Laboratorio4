import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
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

# Graficar la señal
plt.figure(figsize=(10, 5))
plt.plot(time, all_data, label="Señal EMG", linewidth=0.8)
plt.title("Señal de Electromiografía (EMG) - 120s")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()