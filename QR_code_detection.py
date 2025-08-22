import sys
sys.path.append(r"D:\pip_hub")

from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np
import os
import shutil
import tkinter as tk
from tkinter import messagebox

# 图片文件夹路径
image_folder = r'C:\Users\LHB\Pictures\OCR_Captures'
result_folder = r'C:\Users\LHB\Pictures\OCR_Results'
processed_folder = r'C:\Users\LHB\Pictures\Processed_Images'
temp_folder = r'C:\Users\LHB\Pictures\Temp_Results'

# 创建结果文件夹、已处理文件夹和临时文件夹
os.makedirs(result_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# 检查是否有图片文件
if not image_files:
    print(f"在文件夹 {image_folder} 中没有找到任何图片文件。")
    sys.exit(1)

# 初始化 PaddleOCR
try:
    ocr = PaddleOCR(use_textline_orientation=True, lang='ch') # 使用 `use_textline_orientation=True` 替代已弃用的 use_angle_cls，`lang='ch'` 指定中文
    print("PaddleOCR 初始化成功")
except Exception as e:
    print(f"PaddleOCR 初始化失败: {e}")
    sys.exit(1)

# 记录需要覆盖的文件
files_to_overwrite = []

# 处理每个图片文件
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"正在处理图片文件: {image_path}")
    
    # 打开图片
    try:
        with Image.open(image_path) as image:
            print(f"图片成功打开: {image_path}")

            # 进行文字识别
            try:
                ocr_result = ocr.ocr(image_path, cls=True)
                print(f"文字识别成功: {image_path}")
                print("识别结果:", ocr_result)
            except Exception as e:
                print(f"文字识别失败: {e}")
                continue

            # 进行二维码和条形码识别
            qr_results = []
            try:
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detector = cv2.QRCodeDetector()
                retval, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)
                if retval:
                    for barcode_data in decoded_info:
                        if barcode_data:
                            print(f"识别到的二维码/条形码: {barcode_data}")
                            qr_results.append(barcode_data)
                else:
                    print("未识别到二维码/条形码")
            except Exception as e:
                print(f"二维码/条形码识别失败: {e}")

            # 保存识别结果到临时文件夹
            temp_result_file_path = os.path.join(temp_folder, f"{os.path.splitext(image_file)[0]}_result.txt")
            try:
                with open(temp_result_file_path, 'w', encoding='utf-8') as result_file:
                    # 保存文字识别结果
                    for res in ocr_result:
                        for line in res:
                            result_file.write(line[1][0] + '\n')
                    # 保存二维码/条形码识别结果
                    for qr_result in qr_results:
                        result_file.write(qr_result + '\n')
                print(f"识别结果已保存到临时文件: {temp_result_file_path}")
            except Exception as e:
                print(f"保存识别结果失败: {e}")
                continue

            # 检查结果文件是否已存在
            result_file_path = os.path.join(result_folder, f"{os.path.splitext(image_file)[0]}_result.txt")
            if os.path.exists(result_file_path):
                files_to_overwrite.append((result_file_path, temp_result_file_path))
                print(f"结果文件已存在，需要覆盖: {result_file_path}")
            else:
                try:
                    shutil.move(temp_result_file_path, result_file_path)
                    print(f"结果文件已移动到: {result_file_path}")
                except Exception as e:
                    print(f"移动临时结果文件失败: {e}")

    except Exception as e:
        print(f"处理图片失败: {e}")
        continue

    # 移动已处理的图片到新的文件夹
    try:
        shutil.copy(image_path, os.path.join(processed_folder, image_file))
        print(f"图片已复制到: {os.path.join(processed_folder, image_file)}")
    except Exception as e:
        print(f"复制图片失败: {e}")

# 处理需要覆盖的文件
root = tk.Tk()
root.withdraw()  # 隐藏主窗口
for result_file_path, temp_result_file_path in files_to_overwrite:
    if messagebox.askyesno("文件已存在", f"{result_file_path} 已存在。是否覆盖？"):
        try:
            shutil.move(temp_result_file_path, result_file_path)
            print(f"文件已覆盖: {result_file_path}")
        except Exception as e:
            print(f"覆盖文件失败: {e}")
    else:
        try:
            os.remove(temp_result_file_path)
            print(f"跳过文件: {result_file_path}")
        except Exception as e:
            print(f"删除临时文件失败: {e}")

# 删除临时文件夹
try:
    shutil.rmtree(temp_folder)
    print(f"临时文件夹已删除: {temp_folder}")
except Exception as e:
    print(f"删除临时文件夹失败: {e}")