import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

class FixedRatioLabelExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("固定界面比例的标签提取工具")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 600)
        
        # 配置变量
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.border_mm = tk.DoubleVar(value=1.0)
        self.detection_method = tk.StringVar(value="auto")  # auto或manual
        self.edge_threshold1 = tk.IntVar(value=50)
        self.edge_threshold2 = tk.IntVar(value=150)
        self.min_contour_area = tk.IntVar(value=500)
        self.preview_mode = tk.StringVar(value="original")  # original, detected, edges
        # 新增：自动检测范围调整参数
        self.auto_expand_ratio = tk.DoubleVar(value=1.0)  # 自动检测范围扩大比例
        
        # 图像相关变量
        self.original_img = None  # 原始图像
        self.processed_img = None  # 处理后的图像
        self.edges_img = None  # 边缘检测图像
        self.display_img = None  # 显示用图像
        self.photo_img = None  # Tkinter显示用照片对象
        self.scale_factor = 1.0  # 缩放因子
        self.offset_x = 0  # 平移X偏移
        self.offset_y = 0  # 平移Y偏移
        self.last_x = 0  # 上次鼠标X位置
        self.last_y = 0  # 上次鼠标Y位置
        self.dragging = False  # 是否正在拖动
        self.preview_updating = False  # 预览更新标志，防止递归更新
        
        # 检测结果变量
        self.detected_contour = None  # 检测到的轮廓
        self.detected_rect = None  # 检测到的矩形
        
        # 手动选择相关变量
        self.manual_rect = None
        self.selecting = False
        self.start_x = 0
        self.start_y = 0
        self.temp_rect = None
        
        # 创建界面
        self.create_widgets()
        
        # 绑定事件
        self.preview_canvas.bind("<Button-1>", self.on_canvas_click)
        self.preview_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.preview_canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.preview_canvas.bind("<Button-4>", self.on_mouse_wheel)   # Linux
        self.preview_canvas.bind("<Button-5>", self.on_mouse_wheel)   # Linux
        self.root.bind("<Configure>", self.on_window_resize)  # 窗口大小变化事件
        
        # 参数变化时自动更新预览，但添加延迟防止频繁更新
        self.edge_threshold1.trace_add("write", lambda *args: self.schedule_preview_update())
        self.edge_threshold2.trace_add("write", lambda *args: self.schedule_preview_update())
        self.min_contour_area.trace_add("write", lambda *args: self.schedule_preview_update())
        self.detection_method.trace_add("write", lambda *args: self.reset_manual_selection())
        self.border_mm.trace_add("write", lambda *args: self.update_border_label())
        # 新增：自动扩展比例变化时更新预览
        self.auto_expand_ratio.trace_add("write", lambda *args: self.schedule_preview_update())
    
    def create_widgets(self):
        # 主框架 - 设置权重以固定比例
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=2)  # 控制面板占2份
        main_frame.grid_columnconfigure(1, weight=5)  # 预览区域占5份
        main_frame.grid_rowconfigure(0, weight=1)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="设置", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # 输入输出设置
        io_frame = ttk.LabelFrame(control_frame, text="文件设置", padding="5")
        io_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(io_frame, text="输入图片:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(io_frame, textvariable=self.input_path, width=25).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(io_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(io_frame, text="输出目录:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(io_frame, textvariable=self.output_dir, width=25).grid(row=1, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(io_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        io_frame.grid_columnconfigure(1, weight=1)
        
        # 检测方法设置
        method_frame = ttk.LabelFrame(control_frame, text="检测设置", padding="5")
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(method_frame, text="检测方法:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        method_subframe = ttk.Frame(method_frame)
        method_subframe.grid(row=0, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(method_subframe, text="自动检测", variable=self.detection_method, value="auto").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_subframe, text="手动框选", variable=self.detection_method, value="manual").pack(side=tk.LEFT, padx=5)
        
        # 新增：自动检测范围调整 - 改进布局和标签
        auto_detect_settings_frame = ttk.LabelFrame(control_frame, text="自动检测参数", padding="5")
        auto_detect_settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(auto_detect_settings_frame, text="检测范围扩大比例:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(auto_detect_settings_frame, variable=self.auto_expand_ratio, from_=1.0, to=2.0, orient="horizontal", 
                 length=150, command=lambda v: self.update_label(self.expand_label, f"{float(v):.2f}x")).grid(row=1, column=0, padx=5, pady=5, sticky="we")
        self.expand_label = ttk.Label(auto_detect_settings_frame, text=f"{self.auto_expand_ratio.get():.2f}x")
        self.expand_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # 边界设置 - 显示小数点后三位
        ttk.Label(method_frame, text="边界宽度(mm):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(method_frame, variable=self.border_mm, from_=0.1, to=5.0, orient="horizontal", 
                 length=150).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.border_label = ttk.Label(method_frame, text=f"{self.border_mm.get():.3f} mm")
        self.border_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        
        # 边缘检测参数
        ttk.Label(method_frame, text="边缘阈值1:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(method_frame, variable=self.edge_threshold1, from_=10, to=200, orient="horizontal", 
                 length=150, command=lambda v: self.update_label(self.edge1_label, f"{int(float(v))}")).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.edge1_label = ttk.Label(method_frame, text=f"{self.edge_threshold1.get()}")
        self.edge1_label.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(method_frame, text="边缘阈值2:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(method_frame, variable=self.edge_threshold2, from_=100, to=400, orient="horizontal", 
                 length=150, command=lambda v: self.update_label(self.edge2_label, f"{int(float(v))}")).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.edge2_label = ttk.Label(method_frame, text=f"{self.edge_threshold2.get()}")
        self.edge2_label.grid(row=4, column=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(method_frame, text="最小轮廓面积:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(method_frame, variable=self.min_contour_area, from_=100, to=5000, orient="horizontal", 
                 length=150, command=lambda v: self.update_label(self.area_label, f"{int(float(v))}")).grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.area_label = ttk.Label(method_frame, text=f"{self.min_contour_area.get()}")
        self.area_label.grid(row=5, column=2, padx=5, pady=5, sticky="w")
        
        method_frame.grid_columnconfigure(1, weight=1)
        
        # 预览模式设置
        preview_frame = ttk.LabelFrame(control_frame, text="预览设置", padding="5")
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(preview_frame, text="预览模式:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        preview_subframe = ttk.Frame(preview_frame)
        preview_subframe.grid(row=0, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(preview_subframe, text="原图", variable=self.preview_mode, value="original", command=self.update_preview).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(preview_subframe, text="检测结果", variable=self.preview_mode, value="detected", command=self.update_preview).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(preview_subframe, text="边缘图像", variable=self.preview_mode, value="edges", command=self.update_preview).pack(side=tk.LEFT, padx=5)
        
        # 操作按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="更新预览", command=self.update_preview, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="提取标签", command=self.process_image, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="适应窗口", command=self.fit_to_window, width=15).pack(side=tk.LEFT, padx=5)
        
        # 右侧预览区域
        preview_container = ttk.LabelFrame(main_frame, text="预览区域", padding="10")
        preview_container.grid(row=0, column=1, sticky="nsew")
        
        # 预览画布 - 设置固定的初始滚动区域以防止跳动
        self.preview_canvas = tk.Canvas(preview_container, bg="#f0f0f0", cursor="cross", scrollregion=(0, 0, 1000, 1000))
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        scrollbar_x = ttk.Scrollbar(preview_container, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        scrollbar_y = ttk.Scrollbar(preview_container, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        self.preview_canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
        
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 状态栏
        status_frame = ttk.Frame(self.root, height=20)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(status_frame, text="就绪", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=2)
    
    def update_label(self, label, text):
        """更新标签文本"""
        label.config(text=text)
    
    def update_border_label(self):
        """更新边界宽度标签，显示小数点后三位"""
        self.border_label.config(text=f"{self.border_mm.get():.3f} mm")
    
    def browse_input(self):
        """浏览选择输入图片"""
        file_path = filedialog.askopenfilename(
            title="选择输入图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.input_path.set(file_path)
            self.load_image(file_path)
            self.reset_manual_selection()
            self.fit_to_window()  # 加载图片后自动适应窗口
            self.status_label.config(text=f"已加载图片: {os.path.basename(file_path)}")
    
    def browse_output(self):
        """浏览选择输出目录"""
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir.set(dir_path)
            self.status_label.config(text=f"输出目录: {dir_path}")
    
    def load_image(self, file_path):
        """加载图像"""
        try:
            self.original_img = cv2.imread(file_path)
            if self.original_img is None:
                raise ValueError("无法读取图像文件")
            
            # 转换为RGB用于显示
            self.processed_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            self.edges_img = None  # 重置边缘图像
            self.detected_contour = None  # 重置检测结果
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
            self.status_label.config(text=f"加载图片失败: {str(e)}")
    
    def schedule_preview_update(self):
        """延迟调度预览更新，避免参数调整时频繁更新"""
        if self.original_img is None:
            return
            
        # 取消之前的调度，只执行最后一次
        try:
            self.root.after_cancel(self.update_after_id)
        except:
            pass
            
        # 延迟50毫秒更新，避免滑块拖动时频繁刷新
        self.update_after_id = self.root.after(50, self.update_preview)
    
    def fit_to_window(self):
        """调整图像大小以适应窗口"""
        if self.original_img is None or self.preview_updating:
            return
            
        # 获取画布尺寸
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # 如果画布还没渲染，使用窗口尺寸估算
        if canvas_width < 100 or canvas_height < 100:
            canvas_width = int(self.root.winfo_width() * 5 / 7) - 50  # 根据比例计算
            canvas_height = self.root.winfo_height() - 100  # 减去边距和状态栏
        
        # 获取图像尺寸
        img_height, img_width = self.processed_img.shape[:2]
        
        # 计算适应窗口的缩放因子
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        self.scale_factor = min(scale_width, scale_height) * 0.95  # 留5%的边距
        
        # 居中显示
        self.offset_x = max(0, (canvas_width - img_width * self.scale_factor) / 2)
        self.offset_y = max(0, (canvas_height - img_height * self.scale_factor) / 2)
        
        self.update_preview()
    
    def on_window_resize(self, event):
        """窗口大小变化时自动调整，保持固定比例"""
        # 避免在窗口初始化时触发，避免预览更新时递归触发
        if self.original_img is not None and event.widget == self.root and not self.preview_updating:
            # 检查是否是显著的尺寸变化
            if abs(event.width - self.root.winfo_width()) > 50 or abs(event.height - self.root.winfo_height()) > 50:
                self.fit_to_window()
    
    def reset_manual_selection(self):
        """重置手动选择区域"""
        self.manual_rect = None
        self.temp_rect = None
        if self.detection_method.get() == "manual":
            self.status_label.config(text="请在预览区域拖动鼠标框选标签")
        else:
            self.update_preview()
    
    def on_canvas_click(self, event):
        """处理画布点击事件 - 修复判断条件"""
        if self.original_img is None:  # 正确判断图像是否加载
            return
            
        if self.detection_method.get() == "manual":
            # 手动框选模式
            self.selecting = True
            x, y = self.screen_to_image(event.x, event.y)
            self.start_x, self.start_y = x, y
            self.temp_rect = None
        else:
            # 自动模式下允许拖动图像
            self.dragging = True
            self.last_x = event.x
            self.last_y = event.y
    
    def on_canvas_drag(self, event):
        """处理画布拖动事件 - 修复判断条件"""
        if self.original_img is None:  # 正确判断图像是否加载
            return
            
        if self.detection_method.get() == "manual" and self.selecting:
            # 手动框选拖动
            x, y = self.screen_to_image(event.x, event.y)
            self.temp_rect = (
                min(self.start_x, x), 
                min(self.start_y, y),
                abs(x - self.start_x),
                abs(y - self.start_y)
            )
            self.update_preview()
        elif self.dragging:
            # 图像拖动
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.offset_x += dx
            self.offset_y += dy
            self.last_x = event.x
            self.last_y = event.y
            self.update_preview()
    
    def on_canvas_release(self, event):
        """处理画布释放事件 - 修复判断条件"""
        if self.original_img is None:  # 正确判断图像是否加载
            return
            
        if self.detection_method.get() == "manual" and self.selecting:
            # 完成手动框选
            self.selecting = False
            if self.temp_rect and self.temp_rect[2] > 10 and self.temp_rect[3] > 10:  # 确保区域足够大
                self.manual_rect = self.temp_rect
                self.status_label.config(text="已选择标签区域")
            else:
                self.temp_rect = None
                self.status_label.config(text="选择区域过小，请重新选择")
            self.update_preview()
        else:
            # 结束拖动
            self.dragging = False
    
    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件（缩放） - 修复判断条件"""
        if self.original_img is None:  # 正确判断图像是否加载
            return
            
        # 获取鼠标在图像上的位置
        x, y = event.x, event.y
        img_x, img_y = self.screen_to_image(x, y)
        
        # 处理滚轮事件
        if event.delta > 0 or event.num == 4:  # 放大
            self.scale_factor *= 1.1
        elif event.delta < 0 or event.num == 5:  # 缩小
            self.scale_factor /= 1.1
            self.scale_factor = max(0.1, self.scale_factor)  # 限制最小缩放
        
        # 调整偏移量，使鼠标指向的点保持在同一位置
        new_x, new_y = self.image_to_screen(img_x, img_y)
        self.offset_x += x - new_x
        self.offset_y += y - new_y
        
        self.update_preview()
    
    def image_to_screen(self, x, y):
        """将图像坐标转换为屏幕坐标"""
        return int(x * self.scale_factor + self.offset_x), int(y * self.scale_factor + self.offset_y)
    
    def screen_to_image(self, x, y):
        """将屏幕坐标转换为图像坐标"""
        return int((x - self.offset_x) / self.scale_factor), int((y - self.offset_y) / self.scale_factor)
    
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
        if img is None:
            return None, None
            
        # 预处理
        preprocessed = self.preprocess_image(img)
        
        # 应用自动检测范围扩大比例
        if self.auto_expand_ratio.get() > 1.0:
            # 创建一个比原图更大的画布来扩展检测区域
            h, w = preprocessed.shape[:2]
            new_w = int(w * self.auto_expand_ratio.get())
            new_h = int(h * self.auto_expand_ratio.get())
            
            # 创建扩展后的图像（用黑色填充）
            expanded = np.zeros((new_h, new_w), dtype=preprocessed.dtype)
            
            # 将原图像放置在中心
            y_offset = (new_h - h) // 2
            x_offset = (new_w - w) // 2
            expanded[y_offset:y_offset+h, x_offset:x_offset+w] = preprocessed
            
            # 更新处理后的图像
            preprocessed = expanded
        
        # 边缘检测
        edges = cv2.Canny(preprocessed, self.edge_threshold1.get(), self.edge_threshold2.get())
        
        # 边缘增强
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 找到所有轮廓
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选最佳轮廓
        best_contour = self.get_best_label_contour(contours, preprocessed.shape)
        
        # 如果使用了扩展检测，需要将轮廓坐标转换回原始图像坐标
        if self.auto_expand_ratio.get() > 1.0 and best_contour is not None:
            # 计算偏移量
            h, w = img.shape[:2]
            y_offset = (preprocessed.shape[0] - h) // 2
            x_offset = (preprocessed.shape[1] - w) // 2
            
            # 调整轮廓坐标
            best_contour[:, :, 0] = best_contour[:, :, 0] - x_offset
            best_contour[:, :, 1] = best_contour[:, :, 1] - y_offset
            
            # 确保轮廓坐标在有效范围内
            best_contour[:, :, 0] = np.clip(best_contour[:, :, 0], 0, w-1)
            best_contour[:, :, 1] = np.clip(best_contour[:, :, 1], 0, h-1)
        
        # 转换边缘图像为RGB用于显示
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return best_contour, edges_rgb
    
    def update_preview(self):
        """更新预览区域显示，保持视图稳定"""
        if self.original_img is None or self.preview_updating:
            return
            
        # 设置更新标志，防止递归更新
        self.preview_updating = True
        
        try:
            # 保存当前画布尺寸
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # 根据预览模式选择要显示的图像
            if self.preview_mode.get() == "original":
                display_img = self.processed_img.copy()
            elif self.preview_mode.get() == "edges":
                # 如果还没有计算边缘，先计算
                if self.edges_img is None:
                    _, self.edges_img = self.detect_label_auto(self.original_img)
                display_img = self.edges_img.copy() if self.edges_img is not None else self.processed_img.copy()
            else:  # detected
                display_img = self.processed_img.copy()
                # 自动检测模式下绘制检测结果
                if self.detection_method.get() == "auto":
                    self.detected_contour, self.edges_img = self.detect_label_auto(self.original_img)
                    if self.detected_contour is not None:
                        # 绘制轮廓
                        cv2.drawContours(display_img, [self.detected_contour], -1, (0, 255, 0), 2)
                        # 绘制边界框
                        x, y, w, h = cv2.boundingRect(self.detected_contour)
                        
                        # 新增：根据auto_expand_ratio调整显示的边界框
                        if self.auto_expand_ratio.get() > 1.0:
                            # 计算新的边界框尺寸
                            center_x = x + w // 2
                            center_y = y + h // 2
                            new_w = int(w * self.auto_expand_ratio.get())
                            new_h = int(h * self.auto_expand_ratio.get())
                            
                            # 调整边界框坐标
                            x = max(0, center_x - new_w // 2)
                            y = max(0, center_y - new_h // 2)
                            w = min(display_img.shape[1] - x, new_w)
                            h = min(display_img.shape[0] - y, new_h)
                        
                        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        self.detected_rect = (x, y, w, h)
                    else:
                        self.detected_rect = None
                        cv2.putText(display_img, "未检测到标签，请调整参数", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 手动模式下绘制选择框
            if self.detection_method.get() == "manual":
                if self.temp_rect:
                    x, y, w, h = self.temp_rect
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if self.manual_rect:
                    x, y, w, h = self.manual_rect
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(display_img, "标签区域", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 调整图像大小用于显示（保持当前缩放比例）
            h, w = display_img.shape[:2]
            scaled_h, scaled_w = int(h * self.scale_factor), int(w * self.scale_factor)
            resized_img = cv2.resize(display_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            
            # 转换为PhotoImage
            self.photo_img = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
            
            # 更新画布，但保持当前视图位置
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(self.offset_x, self.offset_y, image=self.photo_img, anchor=tk.NW)
            
            # 维持滚动区域稳定，只在必要时调整
            if canvas_width > 0 and canvas_height > 0:
                scroll_width = max(scaled_w + self.offset_x * 2, canvas_width)
                scroll_height = max(scaled_h + self.offset_y * 2, canvas_height)
                self.preview_canvas.config(scrollregion=(0, 0, scroll_width, scroll_height))
                
        finally:
            # 重置更新标志
            self.preview_updating = False
    
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
                if self.detected_contour is None:
                    # 重新尝试检测
                    self.detected_contour, _ = self.detect_label_auto(img)
                    
                if self.detected_contour is None:
                    raise ValueError("自动检测失败，请调整参数或使用手动模式")
                
                # 获取旋转边界框
                rect = cv2.minAreaRect(self.detected_contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)  # 修改:将np.int32替换为np.intp
                
                # 计算边界框的最小和最大坐标
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 新增：根据设置扩大检测范围
                if self.auto_expand_ratio.get() > 1.0:
                    width = x_max - x_min
                    height = y_max - y_min
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    
                    # 按比例扩大
                    new_width = int(width * self.auto_expand_ratio.get())
                    new_height = int(height * self.auto_expand_ratio.get())
                    
                    x_min = max(0, center_x - new_width // 2)
                    x_max = min(img.shape[1], center_x + new_width // 2)
                    y_min = max(0, center_y - new_height // 2)
                    y_max = min(img.shape[0], center_y + new_height // 2)
            
            else:  # manual
                if not self.manual_rect:
                    raise ValueError("请先在预览区域框选标签")
                
                x, y, w, h = self.manual_rect
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
            self.status_label.config(text=f"标签提取完成: {os.path.basename(output_path)}")
            
        except Exception as e:
            messagebox.showerror("处理失败", f"发生错误: {str(e)}")
            self.status_label.config(text=f"处理失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    # 设置中文字体支持
    root.option_add("*Font", "SimHei 10")
    app = FixedRatioLabelExtractor(root)
    root.mainloop()
    