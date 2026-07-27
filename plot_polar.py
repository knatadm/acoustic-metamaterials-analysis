import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


data_pairs = {
    "1500 Hz": ["average_1.5_data_str.txt", "average_1.5_data_without_str.txt", "#1f77b4"],
    "1850 Hz": ["average_1.85_data_str.txt", "average_1.85_data_without_str.txt", "#ff7f0e"],
    "2000 Hz": ["average_2_data_str.txt", "average_2_data_without_str.txt", "#d62728"],
    "2010 Hz": ["average_2.01_data_str.txt", "average_2.01_data_without_str.txt", "#16ff0e"],
    "2200 Hz": ["average_2.2_data_str.txt", "average_2.2_data_without_str.txt", "#c750c7"],
}


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)


for label, (str_file, no_str_file, color) in data_pairs.items():
    try:
        data_str = np.genfromtxt(str_file, skip_header=1, encoding='latin-1')
        data_no_str = np.genfromtxt(no_str_file, skip_header=1, encoding='latin-1')
       
        angles_str = data_str[:, 0]
        vals_str = data_str[:, 1]
        angles_no_str = data_no_str[:, 0]
        vals_no_str = data_no_str[:, 1]
       
        # Интерполяция
        f_interp = interp1d(angles_no_str, vals_no_str, kind='linear', fill_value="extrapolate")
        vals_no_str_aligned = f_interp(angles_str)
       
        # Расчет ДБ
        normalized = (vals_str + 1e-9) / (vals_no_str_aligned + 1e-9)
        db_values = 20 * np.log10(np.abs(normalized))
        # Сглаживание: для применения нужно расскоментировать следующую строчку
        # db_values_smoothed = savgol_filter(db_values, window_length=5, polyorder=2)
       
        # Рисуем
        angles_rad = np.radians(angles_str)
        idx = np.argsort(angles_rad)
        angles_sorted = np.append(angles_rad[idx], angles_rad[idx][0])
        db_sorted = np.append(db_values_smoothed[idx], db_values_smoothed[idx][0])
       
        ax.plot(angles_sorted, db_sorted, label=label, color=color, linewidth=2)
       
    except Exception as e:
        print(f"Ошибка в файле {label}: {e}")


ax.set_ylim(-20, 20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.title("Относительная ДН", pad=20)
plt.show()
