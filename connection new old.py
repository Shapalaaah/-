import cv2
import os
import numpy as np
import math
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image

def load_template_image():
    path = filedialog.askopenfilename(
        title="Выберите аэрофотоснимок",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        messagebox.showerror("Ошибка", "Шаблон не выбран.")
        return None
    try:
        img_color = Image.open(path).convert("RGB")
        img_gray = img_color.convert("L")
        cv2.imshow("Selected aerial photograph", np.array(img_color))
        return np.array(img_color), np.array(img_gray)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить аэрофотоснимок:\n{e}")
        return None

def load_map_images_from_folder():
    folder_path = filedialog.askdirectory(title="Выберите папку с тайлами карты")
    if not folder_path:
        messagebox.showerror("Ошибка", "Не выбрана папка с тайлами")
        return []

    supported_formats = (".png", ".jpg", ".jpeg")
    map_images = []
    failed_files = []
    supported_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(supported_formats):
                supported_count += 1
                file_path = os.path.join(root, file_name)
                try:
                    image = Image.open(file_path).convert("RGB")
                    np_color = np.array(image)
                    np_gray = np.array(image.convert("L"))
                    map_images.append((np_color, np_gray, file_path))
                except Exception as e:
                    failed_files.append(f"{file_name} — {str(e)}")

    if not map_images:
        if supported_count == 0:
            messagebox.showerror("Ошибка", "В папке и подпапках нет тайлов подходящего формата (.jpg, .jpeg, .png)")
        else:
            messagebox.showerror("Ошибка", "Не удалось загрузить ни один тайл (возможно, повреждены)")
        return []

    if failed_files:
        messagebox.showwarning("Предупреждение", f"Некоторые тайлы не загружены:\n" + "\n".join(failed_files[:10]))

    messagebox.showinfo("Готово", f"Загружено тайлов: {len(map_images)} из {supported_count} подходящих файлов.")
    return map_images

def extract_lonlat_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    try:
        zoom = int(parts[-3])
        x_tile = int(parts[-2])
        y_tile = int(os.path.splitext(parts[-1])[0])
    except ValueError:
        return None
    n = 2 ** zoom
    lon_deg = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
    lat_deg = math.degrees(lat_rad)
    return round(lat_deg, 6), round(lon_deg, 6)

def match_template_to_tiles(template_color, template_gray, tiles):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template_gray, None)

    best_score = 0
    best_img_color = None
    best_path = ""
    best_H = None

    for color, gray, path in tiles:
        kp2, des2 = sift.detectAndCompute(gray, None)
        if des2 is None:
            continue
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 4:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        if len(good) > best_score and H is not None:
            best_score = len(good)
            best_img_color = color
            best_H = H
            best_path = path

    if best_img_color is not None:
        result_img = best_img_color.copy()
        h, w = template_gray.shape
        corners = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, best_H)
        cv2.polylines(result_img, [np.int32(transformed)], isClosed=True, color=(0, 255, 0), thickness=2)

        coords = extract_lonlat_from_path(best_path)
        coord_text = f"Координаты тайла:\nШирота = {coords[0]}°, Долгота = {coords[1]}°" if coords else "Координаты не определены"
        if coords:
            save_result_to_file(best_path, coords, best_score)
            
        messagebox.showinfo("Результат", f"Лучшее совпадение:\n{best_path}\nСовпадений: {best_score}\n{coord_text}")
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        messagebox.showinfo("Результат", "Совпадений не найдено.")
        
def save_result_to_file(file_path, coords, match_count):
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"Файл: {file_path}\n")
        f.write(f"Широта: {coords[0]}, Долгота: {coords[1]}\n")
        f.write(f"Совпадений: {match_count}\n")
        f.write("---\n")

def save_result_to_file(file_path, coords, match_count):
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"Файл: {file_path}\n")
        f.write(f"Широта: {coords[0]}, Долгота: {coords[1]}\n")
        f.write(f"Совпадений: {match_count}\n")
        f.write("---\n")

def start_gui():
    def on_select_template():
        nonlocal template_color, template_gray
        result = load_template_image()
        if result:
            template_color, template_gray = result

    def on_select_folder():
        nonlocal tiles
        tiles = load_map_images_from_folder()

    def on_start_matching():
        if template_color is None or template_gray is None:
            messagebox.showerror("Ошибка", "Сначала выберите аэрофотоснимок.")
            return
        if not tiles:
            messagebox.showerror("Ошибка", "Сначала выберите тайлы.")
            return
        match_template_to_tiles(template_color, template_gray, tiles)

    template_color = None
    template_gray = None
    tiles = []

    root = Tk()
    root.title("Поиск координат по аэрофотоснимку")
    root.geometry("400x250")

    Button(root, text="Выбрать аэрофотоснимок", command=on_select_template, width=30).pack(pady=10)
    Button(root, text="Выбрать папку с тайлами", command=on_select_folder, width=30).pack(pady=10)
    Button(root, text="Поиск", command=on_start_matching, width=30).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
