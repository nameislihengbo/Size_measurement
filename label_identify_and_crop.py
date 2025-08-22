import cv2
import numpy as np
import os
from PIL import Image
import argparse
from tkinter import filedialog, Tk
import sys

def detect_and_crop_labels(input_path, output_folder, min_margin_mm=3):
    """
    自动识别标签区域并裁切，保留指定的白边
    
    Args:
        input_path: 输入图片路径或图片文件夹路径
        output_folder: 输出裁切后图片文件夹路径
        min_margin_mm: 保留的最小白边（毫米），默认3mm
    """
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 判断输入路径是文件还是文件夹
    if os.path.isfile(input_path):
        # 处理单个文件
        image_files = [os.path.basename(input_path)]
        input_folder = os.path.dirname(input_path)
    elif os.path.isdir(input_path):
        # 处理文件夹
        input_folder = input_path
        # 获取所有图片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    else:
        print("输入路径既不是文件也不是文件夹！")
        return
    
    if not image_files:
        print("未找到图片文件！")
        return
    
    # 处理每张图片
    for image_file in image_files:
        try:
            image_path = os.path.join(input_folder, image_file)
            # 使用OpenCV读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图片: {image_file}")
                continue
                
            # 转换为HSV色彩空间以更好地检测白色
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 定义白色范围（在HSV色彩空间中）
            # 白色在HSV中饱和度低，明度高
            lower_white = np.array([0, 0, 180])      # 调整阈值以适应不同白色
            upper_white = np.array([180, 30, 255])   # 限制饱和度，避免彩色区域被误识别
            
            # 创建白色掩码
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # 使用形态学操作清理掩码
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算连接相邻区域
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 开运算去除噪点
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 找到最大的轮廓（假设是标签）
            if contours:
                # 根据面积排序轮廓
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest_contour = contours[0]
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 获取图像DPI信息
                try:
                    pil_img = Image.open(image_path)
                    dpi = pil_img.info.get('dpi', (300, 300))
                    if isinstance(dpi, tuple):
                        dpi_x, dpi_y = dpi
                    else:
                        dpi_x = dpi_y = dpi
                except:
                    # 默认DPI
                    dpi_x = dpi_y = 300
                
                # 计算像素到毫米的转换比例
                pixels_per_mm_x = dpi_x / 25.4
                pixels_per_mm_y = dpi_y / 25.4
                
                # 计算保留的像素边距
                margin_pixels_x = int(min_margin_mm * pixels_per_mm_x)
                margin_pixels_y = int(min_margin_mm * pixels_per_mm_y)
                
                # 添加边距（确保不超出图像边界）
                x_start = max(0, x - margin_pixels_x)
                y_start = max(0, y - margin_pixels_y)
                x_end = min(img.shape[1], x + w + margin_pixels_x)
                y_end = min(img.shape[0], y + h + margin_pixels_y)
                
                # 裁切图像
                cropped_img = img[y_start:y_end, x_start:x_end]
                
                # 保存裁切后的图像
                output_path = os.path.join(output_folder, f"cropped_{image_file}")
                cv2.imwrite(output_path, cropped_img)
                print(f"成功处理: {image_file} -> 保存为: cropped_{image_file}")
            else:
                print(f"未找到标签轮廓: {image_file}")
                
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {e}")

def select_input_path():
    """
    打开文件对话框选择图片或图片文件夹
    """
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 询问用户选择文件还是文件夹
    choice = input("请选择输入类型 (1: 单个图片文件, 2: 图片文件夹): ")
    
    if choice == "1":
        # 选择单个图片文件
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        return file_path
    elif choice == "2":
        # 选择图片文件夹
        folder_path = filedialog.askdirectory(title="选择图片文件夹")
        return folder_path
    else:
        print("无效选择")
        return None

def select_output_path():
    """
    打开文件对话框选择输出文件夹
    """
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 选择输出文件夹
    folder_path = filedialog.askdirectory(title="选择输出文件夹")
    return folder_path

def main():
    parser = argparse.ArgumentParser(description='自动识别标签并裁切')
    parser.add_argument('--input', '-i', help='输入图片路径或图片文件夹路径')
    parser.add_argument('--output', '-o', help='输出裁切后图片文件夹路径')
    parser.add_argument('--margin', '-m', type=float, default=3.0, help='保留的白边（毫米），默认3mm')
    
    args = parser.parse_args()
    
    # 如果没有通过命令行参数指定输入路径，则通过对话框选择
    input_path = args.input
    if not input_path:
        input_path = select_input_path()
        if not input_path:
            print("未选择输入路径")
            sys.exit(1)
    
    # 如果没有通过命令行参数指定输出路径，则通过对话框选择
    output_path = args.output
    if not output_path:
        output_path = select_output_path()
        if not output_path:
            print("未选择输出路径")
            sys.exit(1)
    
    detect_and_crop_labels(input_path, output_path, args.margin)

if __name__ == "__main__":
    # 示例用法
    # 修改为您的指定路径
    # INPUT_FOLDER = r"C:\Users\LHB\Pictures\OCR_Captures"  # 输入图片文件夹路径
    # OUTPUT_FOLDER = r"C:\Users\LHB\Pictures\Output_Images"  # 输出文件夹路径
    
    # detect_and_crop_labels(INPUT_FOLDER, OUTPUT_FOLDER, min_margin_mm=3)
    
    main()