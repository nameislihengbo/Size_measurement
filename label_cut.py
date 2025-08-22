import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

class StableLabelExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("稳定的标签提取工具")
        self.root.geometry("750x500")
        
        # 配置变量
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.border_mm = tk.DoubleVar(value=1.0)
        self.detection_method = tk.StringVar(value="auto")  # auto或manual
        self.edge_threshold1 = tk.IntVar(value=50)
        self.edge_threshold2 = tk.IntVar(value=150)
        self.min_contour_area = tk.IntVar(value=500)
        
        # 手动选择相关变量
        self.manual_rect = None
        self.selecting = False
        self.start_x = 0
        self.start_y = 0
        self.temp_rect = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # 网格布局配置
        self.root.grid_columnconfigure(1, weight=1)
        
        # 输入文件选择
        tk.Label(self.root, text="输入图片:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        tk.Entry(self.root, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=10, pady=10, sticky="we")
        tk.Button(self.root, text="浏览...", command=self.browse_input).grid(row=0, column=2, padx=10, pady=10)
        
        # 输出目录选择
        tk.Label(self.root, text="输出目录:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        tk.Entry(self.root, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=10, pady=10, sticky="we")
        tk.Button(self.root, text="浏览...", command=self.browse_output).grid(row=1, column=2, padx=10, pady=10)
        
        # 检测方法选择
        tk.Label(self.root, text="检测方法:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        method_frame = tk.Frame(self.root)
        method_frame.grid(row=2, column=1, sticky="w")
        tk.Radiobutton(method_frame, text="自动检测", variable=self.detection_method, value="auto").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(method_frame, text="手动框选", variable=self.detection_method, value="manual").pack(side=tk.LEFT, padx=10)
        
        # 边界宽度设置
        tk.Label(self.root, text="边界宽度(mm):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        tk.Scale(self.root, variable=self.border_mm, from_=0.1, to=5.0, orient="horizontal", 
                 length=300, command=lambda v: self.update_label(self.border_label, f"{float(v):.1f} mm")).grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.border_label = tk.Label(self.root, text=f"{self.border_mm.get():.1f} mm")
        self.border_label.grid(row=3, column=1, padx=320, pady=5, sticky="w")
        
        # 边缘检测阈值1
        tk.Label(self.root, text="边缘检测阈值1:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        tk.Scale(self.root, variable=self.edge_threshold1, from_=10, to=200, orient="horizontal", 
                 length=300, command=lambda v: self.update_label(self.edge1_label, f"{v}")).grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.edge1_label = tk.Label(self.root, text=f"{self.edge_threshold1.get()}")
        self.edge1_label.grid(row=4, column=1, padx=320, pady=5, sticky="w")
        
        # 边缘检测阈值2
        tk.Label(self.root, text="边缘检测阈值2:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        tk.Scale(self.root, variable=self.edge_threshold2, from_=100, to=400, orient="horizontal", 
                 length=300, command=lambda v: self.update_label(self.edge2_label, f"{v}")).grid(row=5, column=1, padx=10, pady=5, sticky="w")
        self.edge2_label = tk.Label(self.root, text=f"{self.edge_threshold2.get()}")
        self.edge2_label.grid(row=5, column=1, padx=320, pady=5, sticky="w")
        
        # 最小轮廓面积
        tk.Label(self.root, text="最小轮廓面积:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        tk.Scale(self.root, variable=self.min_contour_area, from_=100, to=5000, orient="horizontal", 
                 length=300, command=lambda v: self.update_label(self.area_label, f"{v}")).grid(row=6, column=1, padx=10, pady=5, sticky="w")
        self.area_label = tk.Label(self.root, text=f"{self.min_contour_area.get()}")
        self.area_label.grid(row=6, column=1, padx=320, pady=5, sticky="w")
        
        # 按钮
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=7, column=1, pady=20)
        tk.Button(button_frame, text="预览", command=self.preview, width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="提取标签", command=self.process_image, width=15).pack(side=tk.LEFT, padx=10)
    
    def update_label(self, label, text):
        label.config(text=text)
    
    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="选择输入图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.input_path.set(file_path)
            # 重置手动选择区域
            self.manual_rect = None
    
    def browse_output(self):
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir.set(dir_path)
    
    def mm_to_pixels(self, mm, dpi):
        """将毫米转换为像素"""
        return int(round(mm * dpi / 25.4))
    
    def get_image_dpi(self, image_path):
        """获取图像的DPI信息"""
        try:
            with Image.open(image_path) as img:
                dpi = img.info.get('dpi', (300, 300))
                return (int(dpi[0]), int(dpi[1]))
        except Exception as e:
            print(f"获取DPI信息失败: {e}，使用默认300DPI")
            return (300, 300)
    
    def preprocess_image(self, img):
        """图像预处理，提高后续检测稳定性"""
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 去噪
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return denoised
    
    def get_best_label_contour(self, contours, img_shape):
        """优化的轮廓筛选：选择最可能是标签的轮廓"""
        if not contours:
            return None
            
        best_contour = None
        best_score = -1
        img_area = img_shape[0] * img_shape[1]
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤面积过小或过大的轮廓
            if area < self.min_contour_area.get() or area > img_area * 0.8:
                continue
                
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            bounding_area = w * h
            
            # 计算矩形度（面积与边界框面积的比值，越接近1越可能是矩形标签）
            rectangularity = area / bounding_area if bounding_area > 0 else 0
            
            # 计算纵横比（标签通常不会太狭长）
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            aspect_score = 1.0 / aspect_ratio  # 纵横比越小得分越高
            
            # 计算紧凑度（圆形度）- 标签通常是矩形，这个值会较低
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            compactness = (4 * np.pi * area) / (perimeter * perimeter)
            compactness_score = 1 - compactness  # 矩形更符合，所以这个值高更好
            
            # 计算位置分数（中心区域的轮廓更可能是标签）
            center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
            contour_center_x = x + w // 2
            contour_center_y = y + h // 2
            distance_from_center = np.sqrt((contour_center_x - center_x)**2 + 
                                         (contour_center_y - center_y)** 2)
            position_score = 1 / (1 + distance_from_center / max(img_shape[0], img_shape[1]))
            
            # 综合评分
            score = (0.3 * rectangularity + 
                    0.2 * aspect_score + 
                    0.2 * compactness_score + 
                    0.1 * position_score + 
                    0.2 * (area / img_area))
            
            # 更新最佳轮廓
            if score > best_score:
                best_score = score
                best_contour = contour
                
        return best_contour
    
    def detect_label_auto(self, img):
        """自动检测标签"""
        # 预处理
        preprocessed = self.preprocess_image(img)
        
        # 边缘检测
        edges = cv2.Canny(preprocessed, self.edge_threshold1.get(), self.edge_threshold2.get())
        
        # 边缘增强
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 找到所有轮廓
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选最佳轮廓
        best_contour = self.get_best_label_contour(contours, img.shape)
        
        return best_contour, edges
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于手动框选"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_x, self.start_y = x, y
            self.temp_rect = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.temp_rect = (self.start_x, self.start_y, x - self.start_x, y - self.start_y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            if x != self.start_x and y != self.start_y:
                self.manual_rect = (min(self.start_x, x), min(self.start_y, y), 
                                   abs(x - self.start_x), abs(y - self.start_y))
    
    def detect_label_manual(self, img):
        """手动框选标签"""
        if self.manual_rect is not None:
            return self.manual_rect
            
        # 创建窗口并设置鼠标回调
        cv2.namedWindow("手动框选标签 - 拖动鼠标选择区域，按Enter确认")
        cv2.setMouseCallback("手动框选标签 - 拖动鼠标选择区域，按Enter确认", self.mouse_callback)
        
        temp_img = img.copy()
        while True:
            display_img = temp_img.copy()
            
            if self.temp_rect:
                x, y, w, h = self.temp_rect
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if self.manual_rect:
                x, y, w, h = self.manual_rect
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_img, "按Enter确认", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("手动框选标签 - 拖动鼠标选择区域，按Enter确认", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter键确认
                break
            elif key == 27:  # ESC键取消
                self.manual_rect = None
                break
        
        cv2.destroyAllWindows()
        return self.manual_rect
    
    def preview(self):
        """预览检测结果"""
        input_path = self.input_path.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("错误", "请选择有效的输入图片")
            return
        
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("无法读取图像")
            
            method = self.detection_method.get()
            preview_img = img.copy()
            info_text = ""
            
            if method == "auto":
                # 自动检测
                contour, edges = self.detect_label_auto(img)
                
                if contour is not None:
                    # 绘制轮廓
                    cv2.drawContours(preview_img, [contour], -1, (0, 255, 0), 2)
                    
                    # 绘制边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(preview_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    info_text = "自动检测到标签区域"
                else:
                    info_text = "未检测到标签，请调整参数或使用手动模式"
                
                # 显示边缘检测结果
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((preview_img, edges_rgb))
                cv2.putText(combined, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.namedWindow("预览 - 左: 检测结果, 右: 边缘检测", cv2.WINDOW_NORMAL)
                cv2.imshow("预览 - 左: 检测结果, 右: 边缘检测", combined)
                
            else:  # manual
                # 手动框选预览
                rect = self.detect_label_manual(img)
                if rect:
                    x, y, w, h = rect
                    cv2.rectangle(preview_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    info_text = "手动选择的标签区域"
                else:
                    info_text = "未选择区域"
                
                cv2.putText(preview_img, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.namedWindow("手动选择预览", cv2.WINDOW_NORMAL)
                cv2.imshow("手动选择预览", preview_img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            messagebox.showerror("错误", f"预览失败: {str(e)}")
    
    def process_image(self):
        """处理图像并提取标签"""
        input_path = self.input_path.get()
        output_dir = self.output_dir.get()
        
        # 验证输入
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("错误", "请选择有效的输入图片")
            return
        
        if not output_dir or not os.path.exists(output_dir):
            messagebox.showerror("错误", "请选择有效的输出目录")
            return
        
        try:
            # 读取图像
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("无法读取图像")
            
            # 获取DPI
            dpi_x, dpi_y = self.get_image_dpi(input_path)
            border_x = self.mm_to_pixels(self.border_mm.get(), dpi_x)
            border_y = self.mm_to_pixels(self.border_mm.get(), dpi_y)
            
            # 检测标签
            method = self.detection_method.get()
            x_min, y_min, x_max, y_max = 0, 0, img.shape[1], img.shape[0]
            
            if method == "auto":
                contour, _ = self.detect_label_auto(img)
                if contour is None:
                    raise ValueError("自动检测失败，请调整参数或使用手动模式")
                
                # 获取旋转边界框
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # 计算边界框的最小和最大坐标
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
            else:  # manual
                rect = self.detect_label_manual(img)
                if not rect:
                    raise ValueError("未选择标签区域")
                
                x, y, w, h = rect
                x_min, y_min = x, y
                x_max, y_max = x + w, y + h
            
            # 扩展边界
            height, width = img.shape[:2]
            x_start = max(0, x_min - border_x)
            y_start = max(0, y_min - border_y)
            x_end = min(width, x_max + border_x)
            y_end = min(height, y_max + border_y)
            
            # 裁剪图像
            cropped_img = img[y_start:y_end, x_start:x_end]
            
            # 保存结果
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_label{ext}")
            cv2.imwrite(output_path, cropped_img)
            
            messagebox.showinfo("成功", f"标签提取完成，已保存至:\n{output_path}")
            # 重置手动选择区域，以便下次处理新图片
            self.manual_rect = None
            
        except Exception as e:
            messagebox.showerror("处理失败", f"发生错误: {str(e)}")
            print(f"错误: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.option_add("*Font", "SimHei 10")  # 设置中文字体
    app = StableLabelExtractor(root)
    root.mainloop()
    