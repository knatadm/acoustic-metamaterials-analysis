import os
import re
import numpy as np
from scipy.io import wavfile

# --- НАСТРОЙКИ (ПРОПИШИ СВОЁ) ---
SOURCE_DIR = "data_chirp_without_str"       # Папка, где лежат wav-файлы чирпа по углам
OUTPUT_FILE = "average_4_data_without_str.txt"  # Итоговый текстовый файл с данными

F_MIN = 3999.0
F_MAX = 4001.0
# --------------------------------

def extract_angle(filename):
    """Ищет угол в названии файла."""
    match = re.search(r"(-?\d+)", filename)
    return int(match.group(1)) if match else None

def get_chirp_energy(filepath, f_min, f_max):
    """Считает чистую энергию чирпа в полосе частот БЕЗ нормировки на длину."""
    try:
        sample_rate, data = wavfile.read(filepath)

        # Если запись стерео, берем один канал
        if len(data.shape) > 1:
            data = data[:, 0]

        # Переводим в float
        if data.dtype == np.int16:
            data = data.astype(float) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(float) / 2147483648.0
        else:
            data = data.astype(float)

        # Считаем БПФ
        n = len(data)
        len_fft = 2 ** (int(np.ceil(np.log2(n))) - 1)
        
        fft_data = np.fft.fft(data, n=len_fft)
        frequencies = np.fft.fftfreq(len_fft, d=1/sample_rate)
        
        pos_idx = np.where(frequencies >= 0)[0]
        frequencies = frequencies[pos_idx]
        
        # ВАЖНО: берем просто модуль спектра (амплитуду) БЕЗ деления на len_fft!
        amplitudes = np.abs(fft_data[pos_idx]) * 2

        # Вырезаем наш диапазон частот
        search_zone = np.where((frequencies >= f_min) & (frequencies <= f_max))[0]
        
        if len(search_zone) == 0:
            closest_idx = np.abs(frequencies - 4000.0).argmin()
            return amplitudes[closest_idx]
            
        # Возвращаем среднее значение энергии в этой полосе
        return np.mean(amplitudes[search_zone])
        
    except Exception as e:
        print(f"Ошибка с файлом {filepath}: {e}")
        return None

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Папка '{SOURCE_DIR}' не найдена!")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.wav')]
    if not files:
        print(f"В папке '{SOURCE_DIR}' нет wav-файлов.")
        return

    print(f"=== ОБРАБОТКА ЧИРПА ({F_MIN} - {F_MAX} Гц) ===")
    results = []

    for f in files:
        filepath = os.path.join(SOURCE_DIR, f)
        angle = extract_angle(f)
        if angle is None:
            continue

        energy = get_chirp_energy(filepath, F_MIN, F_MAX)
        if energy is not None:
            results.append((angle, energy))
            print(f"Угол: {angle:4}° | Накопленная энергия: {energy:.4f}")

    # Сортируем по углам и сохраняем
    results.sort(key=lambda x: x[0])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# Данные чирп-сигнала (энергия спектра вокруг 4000 Гц)\n")
        out.write("# Угол \t Энергия\n")
        for angle, eng in results:
            out.write(f"{angle} \t {eng:.6f}\n")

    print(f"\nГотово! Данные сохранены в '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()