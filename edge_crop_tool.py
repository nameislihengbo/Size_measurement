import cv2
import numpy as np
import os
import argparse
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class EdgeCropTool:
    def __init__(self, root):
        """
        初始化边缘裁切工具
        :param root: tkinter根窗口
        """
        self.root = root
        self.root.title("图像边缘裁切工具")
        self.root.geometry("1200x750")
        self.root.resizable(True, True)

        # 设置中文字体
        self.style = ttk.Style()
        self.style.configure("TLabel", font=('SimHei', 10))
        self.style.configure("TButton", font=('SimHei', 10))

        # 图像相关变量
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.output_dir = None  # 输出路径
        self.mm_to_pixel_ratio = 1.0  # 毫米到像素的转换比率
        self.margin_mm = 3  # 保留的边缘宽度(mm)
        self.show_preview = True  # 默认显示预览
        self.current_contour = None  # 当前检测到的轮廓
        
        # 轮廓识别参数
        self.canny_low_threshold = 50  # Canny边缘检测低阈值
        self.canny_high_threshold = 150  # Canny边缘检测高阈值
        self.blur_kernel_size = 5  # 高斯模糊核大小
        self.dilate_kernel_size = 5  # 膨胀核大小
        self.dilate_iterations = 1  # 膨胀迭代次数
        
        # 颜色掩膜参数
        self.use_color_mask = False  # 是否使用颜色掩膜
        # 默认粉红色阈值范围 (HSV)
        self.hue_low = 140  # 色相低值
        self.hue_high = 180  # 色相高值
        self.saturation_low = 50  # 饱和度低值
        self.saturation_high = 255  # 饱和度高值
        self.value_low = 50  # 亮度低值
        self.value_high = 255  # 亮度高值

        # 创建UI
        self._create_widgets()

    def _create_widgets(self):
        """
        创建UI组件
        """
        # 顶部按钮区域
        btn_frame = ttk.Frame(self.root, padding="10")
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="打开图片", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="裁切边缘", command=self.crop_edges).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="设置输出路径", command=self.set_output_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存结果", command=self.save_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="恢复默认设置", command=self.reset_to_defaults).pack(side=tk.LEFT, padx=5)
        
        # 预览选项
        self.preview_var = tk.BooleanVar(value=self.show_preview)  # 设置为默认显示预览
        ttk.Checkbutton(btn_frame, text="显示预览", variable=self.preview_var, command=self.toggle_preview).pack(side=tk.LEFT, padx=5)

        # 参数设置区域
        params_frame = ttk.LabelFrame(self.root, text="识别参数设置", padding="10")
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        # 第一行参数 - 比率设置
        row1_frame = ttk.Frame(params_frame)
        row1_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row1_frame, text="毫米到像素比率:").pack(side=tk.LEFT, padx=5)
        self.ratio_var = tk.StringVar(value="1.0")
        ttk.Entry(row1_frame, textvariable=self.ratio_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1_frame, text="应用比率", command=self.update_ratio).pack(side=tk.LEFT, padx=5)
        
        # 比率滑动条
        self.ratio_scale = tk.Scale(row1_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                   length=200, command=self._on_ratio_scale_change)
        self.ratio_scale.set(self.mm_to_pixel_ratio)
        self.ratio_scale.pack(side=tk.LEFT, padx=5)

        # 第二行参数 - Canny低阈值
        row2_frame = ttk.Frame(params_frame)
        row2_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row2_frame, text="Canny低阈值:").pack(side=tk.LEFT, padx=5)
        self.canny_low_var = tk.StringVar(value=str(self.canny_low_threshold))
        ttk.Entry(row2_frame, textvariable=self.canny_low_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Canny低阈值滑动条
        self.canny_low_scale = tk.Scale(row2_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, 
                                       command=self._on_canny_low_scale_change)
        self.canny_low_scale.set(self.canny_low_threshold)
        self.canny_low_scale.pack(side=tk.LEFT, padx=5)

        # 第三行参数 - Canny高阈值
        row3_frame = ttk.Frame(params_frame)
        row3_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row3_frame, text="Canny高阈值:").pack(side=tk.LEFT, padx=5)
        self.canny_high_var = tk.StringVar(value=str(self.canny_high_threshold))
        ttk.Entry(row3_frame, textvariable=self.canny_high_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Canny高阈值滑动条
        self.canny_high_scale = tk.Scale(row3_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, 
                                        command=self._on_canny_high_scale_change)
        self.canny_high_scale.set(self.canny_high_threshold)
        self.canny_high_scale.pack(side=tk.LEFT, padx=5)

        # 第四行参数 - 模糊核大小
        row4_frame = ttk.Frame(params_frame)
        row4_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row4_frame, text="模糊核大小:").pack(side=tk.LEFT, padx=5)
        self.blur_kernel_var = tk.StringVar(value=str(self.blur_kernel_size))
        ttk.Entry(row4_frame, textvariable=self.blur_kernel_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 模糊核大小滑动条
        self.blur_kernel_scale = tk.Scale(row4_frame, from_=1, to=21, orient=tk.HORIZONTAL, length=200, 
                                        command=self._on_blur_kernel_scale_change)
        self.blur_kernel_scale.set(self.blur_kernel_size)
        self.blur_kernel_scale.pack(side=tk.LEFT, padx=5)

        # 第五行参数 - 膨胀核大小
        row5_frame = ttk.Frame(params_frame)
        row5_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row5_frame, text="膨胀核大小:").pack(side=tk.LEFT, padx=5)
        self.dilate_kernel_var = tk.StringVar(value=str(self.dilate_kernel_size))
        ttk.Entry(row5_frame, textvariable=self.dilate_kernel_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 膨胀核大小滑动条
        self.dilate_kernel_scale = tk.Scale(row5_frame, from_=1, to=21, orient=tk.HORIZONTAL, length=200, 
                                           command=self._on_dilate_kernel_scale_change)
        self.dilate_kernel_scale.set(self.dilate_kernel_size)
        self.dilate_kernel_scale.pack(side=tk.LEFT, padx=5)

        # 第六行参数 - 膨胀迭代次数
        row6_frame = ttk.Frame(params_frame)
        row6_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row6_frame, text="膨胀迭代次数:").pack(side=tk.LEFT, padx=5)
        self.dilate_iter_var = tk.StringVar(value=str(self.dilate_iterations))
        ttk.Entry(row6_frame, textvariable=self.dilate_iter_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 膨胀迭代次数滑动条
        self.dilate_iter_scale = tk.Scale(row6_frame, from_=1, to=10, orient=tk.HORIZONTAL, length=200, 
                                         command=self._on_dilate_iter_scale_change)
        self.dilate_iter_scale.set(self.dilate_iterations)
        self.dilate_iter_scale.pack(side=tk.LEFT, padx=5)
        
        # 颜色掩膜选项
        mask_frame = ttk.LabelFrame(params_frame, text="颜色掩膜设置", padding="10")
        mask_frame.pack(fill=tk.X, pady=5)
        
        # 启用颜色掩膜
        self.use_mask_var = tk.BooleanVar(value=self.use_color_mask)
        ttk.Checkbutton(mask_frame, text="启用颜色掩膜", variable=self.use_mask_var, command=self._on_use_mask_change).pack(anchor=tk.W, pady=5)
        
        # 预设背景颜色按钮
        preset_frame = ttk.Frame(mask_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Label(preset_frame, text="预设背景:").pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="粉色背景", command=lambda: self.load_background_preset("pink")).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="白色背景", command=lambda: self.load_background_preset("white")).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="暗色背景", command=lambda: self.load_background_preset("dark")).pack(side=tk.LEFT, padx=5)
        
        # 色相设置
        hue_frame = ttk.Frame(mask_frame)
        hue_frame.pack(fill=tk.X, pady=3)
        ttk.Label(hue_frame, text="色相范围:").pack(side=tk.LEFT, padx=5)
        self.hue_low_var = tk.StringVar(value=str(self.hue_low))
        self.hue_high_var = tk.StringVar(value=str(self.hue_high))
        ttk.Entry(hue_frame, textvariable=self.hue_low_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(hue_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(hue_frame, textvariable=self.hue_high_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 饱和度设置
        saturation_frame = ttk.Frame(mask_frame)
        saturation_frame.pack(fill=tk.X, pady=3)
        ttk.Label(saturation_frame, text="饱和度范围:").pack(side=tk.LEFT, padx=5)
        self.saturation_low_var = tk.StringVar(value=str(self.saturation_low))
        self.saturation_high_var = tk.StringVar(value=str(self.saturation_high))
        ttk.Entry(saturation_frame, textvariable=self.saturation_low_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(saturation_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(saturation_frame, textvariable=self.saturation_high_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 亮度设置
        value_frame = ttk.Frame(mask_frame)
        value_frame.pack(fill=tk.X, pady=3)
        ttk.Label(value_frame, text="亮度范围:").pack(side=tk.LEFT, padx=5)
        self.value_low_var = tk.StringVar(value=str(self.value_low))
        self.value_high_var = tk.StringVar(value=str(self.value_high))
        ttk.Entry(value_frame, textvariable=self.value_low_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(value_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(value_frame, textvariable=self.value_high_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 添加绑定事件，当输入框内容变化时自动更新
        self.canny_low_var.trace_add("write", self._on_canny_low_var_change)
        self.canny_high_var.trace_add("write", self._on_canny_high_var_change)
        self.blur_kernel_var.trace_add("write", self._on_blur_kernel_var_change)
        self.dilate_kernel_var.trace_add("write", self._on_dilate_kernel_var_change)
        self.dilate_iter_var.trace_add("write", self._on_dilate_iter_var_change)
        
        # 添加颜色参数的绑定事件
        self.hue_low_var.trace_add("write", self._on_color_param_change)
        self.hue_high_var.trace_add("write", self._on_color_param_change)
        self.saturation_low_var.trace_add("write", self._on_color_param_change)
        self.saturation_high_var.trace_add("write", self._on_color_param_change)
        self.value_low_var.trace_add("write", self._on_color_param_change)
        self.value_high_var.trace_add("write", self._on_color_param_change)

        # 图像显示区域 - 使用左右布局
        self.image_display_frame = ttk.Frame(self.root, padding="10")
        self.image_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：原图显示
        self.original_frame = ttk.LabelFrame(self.image_display_frame, text="原图", padding="5")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_canvas = tk.Canvas(self.original_frame, bg="gray")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧：预览/结果显示
        self.preview_frame = ttk.LabelFrame(self.image_display_frame, text="预览/结果", padding="5")
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="gray")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        
        # 初始化预览窗口显示提示信息
        self.preview_canvas.create_text(self.preview_canvas.winfo_width()//2 if self.preview_canvas.winfo_width() > 1 else 250, 
                                       self.preview_canvas.winfo_height()//2 if self.preview_canvas.winfo_height() > 1 else 200, 
                                       text="请先打开图像", 
                                       fill="white", font=('SimHei', 12))

    def open_image(self):
        """
        打开图像文件
        """
        file_types = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        file_path = filedialog.askopenfilename(title="选择图像文件", filetypes=[("图像文件", " ".join(file_types))])

        if file_path:
            try:
                self.image_path = file_path
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                self.status_var.set(f"已打开图像: {os.path.basename(file_path)}")
                
                # 显示原图
                self.display_original_image(self.original_image)
                
                # 如果开启了预览，立即更新预览
                if self.show_preview:
                    self.update_preview()
                else:
                    # 清空预览窗口
                    self.preview_canvas.delete("all")
                    self.preview_canvas.create_text(self.preview_canvas.winfo_width()//2, 
                                                   self.preview_canvas.winfo_height()//2, 
                                                   text="请点击'显示预览'按钮查看实时预览", 
                                                   fill="white", font=('SimHei', 12))
            except Exception as e:
                messagebox.showerror("错误", f"打开图像失败: {str(e)}")
                self.status_var.set("打开图像失败")

    def update_ratio(self):
        """
        更新毫米到像素的转换比率
        """
        try:
            ratio = float(self.ratio_var.get())
            if ratio <= 0:
                messagebox.showwarning("警告", "比率必须大于0")
                return
            self.mm_to_pixel_ratio = ratio
            self.ratio_scale.set(ratio)  # 更新滑动条
            self.status_var.set(f"比率已更新为: {ratio}")
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self.update_preview()
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的数字")

    def _on_ratio_scale_change(self, value):
        """
        比率滑动条变化时的回调
        """
        try:
            ratio = float(value)
            self.mm_to_pixel_ratio = ratio
            self.ratio_var.set(f"{ratio:.1f}")  # 更新输入框
            self.status_var.set(f"比率已更新为: {ratio}")
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self.update_preview()
        except ValueError:
            pass
             
    def _on_canny_low_scale_change(self, value):
        """
        Canny低阈值滑动条变化时的回调
        """
        try:
            low_threshold = int(value)
            self.canny_low_threshold = low_threshold
            self.canny_low_var.set(str(low_threshold))  # 更新输入框
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_canny_high_scale_change(self, value):
        """
        Canny高阈值滑动条变化时的回调
        """
        try:
            high_threshold = int(value)
            self.canny_high_threshold = high_threshold
            self.canny_high_var.set(str(high_threshold))  # 更新输入框
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_blur_kernel_scale_change(self, value):
        """
        模糊核大小滑动条变化时的回调
        """
        try:
            kernel_size = int(value)
            # 确保是奇数
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 1:
                kernel_size = 1
            self.blur_kernel_size = kernel_size
            self.blur_kernel_var.set(str(kernel_size))  # 更新输入框
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_dilate_kernel_scale_change(self, value):
        """
        膨胀核大小滑动条变化时的回调
        """
        try:
            kernel_size = int(value)
            if kernel_size < 1:
                kernel_size = 1
            self.dilate_kernel_size = kernel_size
            self.dilate_kernel_var.set(str(kernel_size))  # 更新输入框
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_dilate_iter_scale_change(self, value):
        """
        膨胀迭代次数滑动条变化时的回调
        """
        try:
            iterations = int(value)
            if iterations < 1:
                iterations = 1
            self.dilate_iterations = iterations
            self.dilate_iter_var.set(str(iterations))  # 更新输入框
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_canny_low_var_change(self, *args):
        """
        Canny低阈值输入框变化时的回调
        """
        try:
            low_threshold = int(self.canny_low_var.get())
            # 验证阈值范围
            if low_threshold >= 0 and low_threshold <= 255:
                self.canny_low_threshold = low_threshold
                self.canny_low_scale.set(low_threshold)  # 更新滑动条
                
                # 如果开启了预览，自动更新预览
                if self.show_preview and self.original_image is not None:
                    self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_canny_high_var_change(self, *args):
        """
        Canny高阈值输入框变化时的回调
        """
        try:
            high_threshold = int(self.canny_high_var.get())
            # 验证阈值范围
            if high_threshold >= 0 and high_threshold <= 255:
                self.canny_high_threshold = high_threshold
                self.canny_high_scale.set(high_threshold)  # 更新滑动条
                
                # 如果开启了预览，自动更新预览
                if self.show_preview and self.original_image is not None:
                    self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_blur_kernel_var_change(self, *args):
        """
        模糊核大小输入框变化时的回调
        """
        try:
            kernel_size = int(self.blur_kernel_var.get())
            # 确保是奇数
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 1:
                kernel_size = 1
            self.blur_kernel_size = kernel_size
            self.blur_kernel_var.set(str(kernel_size))  # 更新输入框
            self.blur_kernel_scale.set(kernel_size)  # 更新滑动条
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_dilate_kernel_var_change(self, *args):
        """
        膨胀核大小输入框变化时的回调
        """
        try:
            kernel_size = int(self.dilate_kernel_var.get())
            if kernel_size < 1:
                kernel_size = 1
            self.dilate_kernel_size = kernel_size
            self.dilate_kernel_var.set(str(kernel_size))  # 更新输入框
            self.dilate_kernel_scale.set(kernel_size)  # 更新滑动条
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def _on_dilate_iter_var_change(self, *args):
        """
        膨胀迭代次数输入框变化时的回调
        """
        try:
            iterations = int(self.dilate_iter_var.get())
            if iterations < 1:
                iterations = 1
            self.dilate_iterations = iterations
            self.dilate_iter_var.set(str(iterations))  # 更新输入框
            
            # 如果开启了预览，自动更新预览
            if self.show_preview and self.original_image is not None:
                self._validate_and_update_preview()
        except ValueError:
            pass
             
    def set_output_directory(self):
        """
        设置输出目录
        """
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir = directory
            self.status_var.set(f"输出目录已设置: {directory}")

    def reset_to_defaults(self):
        """
        恢复默认设置
        """
        # 重置所有参数到默认值
        self.mm_to_pixel_ratio = 1.0
        self.margin_mm = 3
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.blur_kernel_size = 5
        self.dilate_kernel_size = 5
        self.dilate_iterations = 1
        self.use_color_mask = False
        self.hue_low = 140
        self.hue_high = 180
        self.saturation_low = 50
        self.saturation_high = 255
        self.value_low = 50
        self.value_high = 255
        
        # 更新UI显示
        self.ratio_var.set("1.0")
        self.ratio_scale.set(self.mm_to_pixel_ratio)
        self.canny_low_var.set(str(self.canny_low_threshold))
        self.canny_low_scale.set(self.canny_low_threshold)
        self.canny_high_var.set(str(self.canny_high_threshold))
        self.canny_high_scale.set(self.canny_high_threshold)
        self.blur_kernel_var.set(str(self.blur_kernel_size))
        self.blur_kernel_scale.set(self.blur_kernel_size)
        self.dilate_kernel_var.set(str(self.dilate_kernel_size))
        self.dilate_kernel_scale.set(self.dilate_kernel_size)
        self.dilate_iter_var.set(str(self.dilate_iterations))
        self.dilate_iter_scale.set(self.dilate_iterations)
        self.use_mask_var.set(self.use_color_mask)
        self.hue_low_var.set(str(self.hue_low))
        self.hue_high_var.set(str(self.hue_high))
        self.saturation_low_var.set(str(self.saturation_low))
        self.saturation_high_var.set(str(self.saturation_high))
        self.value_low_var.set(str(self.value_low))
        self.value_high_var.set(str(self.value_high))
        
        self.status_var.set("已恢复默认设置")
        
        # 如果有图像并且开启了预览，更新预览
        if self.original_image is not None and self.show_preview:
            self.update_preview()

    def load_background_preset(self, preset_type):
        """
        加载预设的背景颜色参数
        :param preset_type: 预设类型 ("pink", "white", "dark")
        """
        if preset_type == "pink":
            # 粉红色背景预设
            self.hue_low = 140
            self.hue_high = 180
            self.saturation_low = 50
            self.saturation_high = 255
            self.value_low = 50
            self.value_high = 255
            self.status_var.set("已加载粉红色背景预设")
        elif preset_type == "white":
            # 白色背景预设
            self.hue_low = 0
            self.hue_high = 180
            self.saturation_low = 0
            self.saturation_high = 50
            self.value_low = 200
            self.value_high = 255
            self.status_var.set("已加载白色背景预设")
        elif preset_type == "dark":
            # 暗色背景预设
            self.hue_low = 0
            self.hue_high = 180
            self.saturation_low = 0
            self.saturation_high = 100
            self.value_low = 0
            self.value_high = 50
            self.status_var.set("已加载暗色背景预设")
        
        # 更新UI显示
        self.hue_low_var.set(str(self.hue_low))
        self.hue_high_var.set(str(self.hue_high))
        self.saturation_low_var.set(str(self.saturation_low))
        self.saturation_high_var.set(str(self.saturation_high))
        self.value_low_var.set(str(self.value_low))
        self.value_high_var.set(str(self.value_high))
        
        # 自动启用颜色掩膜
        self.use_mask_var.set(True)
        self.use_color_mask = True
        
        # 如果有图像并且开启了预览，更新预览
        if self.original_image is not None and self.show_preview:
            self.update_preview()

    def _on_use_mask_change(self):
        """
        启用/禁用颜色掩膜时的回调
        """
        self.use_color_mask = self.use_mask_var.get()
        self.status_var.set(f"颜色掩膜: {'已启用' if self.use_color_mask else '已禁用'}")
        
        # 如果开启了预览，自动更新预览
        if self.show_preview and self.original_image is not None:
            self.update_preview()
            
    def _on_color_param_change(self, *args):
        """
        颜色参数变化时的回调
        """
        try:
            # 更新所有颜色参数
            self.hue_low = int(self.hue_low_var.get())
            self.hue_high = int(self.hue_high_var.get())
            self.saturation_low = int(self.saturation_low_var.get())
            self.saturation_high = int(self.saturation_high_var.get())
            self.value_low = int(self.value_low_var.get())
            self.value_high = int(self.value_high_var.get())
            
            # 验证参数范围
            self.hue_low = max(0, min(180, self.hue_low))
            self.hue_high = max(0, min(180, self.hue_high))
            self.saturation_low = max(0, min(255, self.saturation_low))
            self.saturation_high = max(0, min(255, self.saturation_high))
            self.value_low = max(0, min(255, self.value_low))
            self.value_high = max(0, min(255, self.value_high))
            
            # 更新输入框
            self.hue_low_var.set(str(self.hue_low))
            self.hue_high_var.set(str(self.hue_high))
            self.saturation_low_var.set(str(self.saturation_low))
            self.saturation_high_var.set(str(self.saturation_high))
            self.value_low_var.set(str(self.value_low))
            self.value_high_var.set(str(self.value_high))
            
            # 如果开启了预览和颜色掩膜，自动更新预览
            if self.show_preview and self.original_image is not None and self.use_color_mask:
                self.update_preview()
        except ValueError:
            pass
            
    def _validate_and_update_preview(self):
        """
        验证参数并更新预览
        """
        # 验证Canny阈值关系
        if self.canny_low_threshold >= self.canny_high_threshold:
            # 自动调整高阈值大于低阈值
            self.canny_high_threshold = self.canny_low_threshold + 1
            self.canny_high_var.set(str(self.canny_high_threshold))
            self.canny_high_scale.set(self.canny_high_threshold)
        
        # 调用预览更新
        self.update_preview()

    def display_original_image(self, image):
        """
        在原图画布上显示图像
        :param image: 输入图像
        """
        # 调整图像大小以适应画布
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()

        if canvas_width == 1 or canvas_height == 1:  # 确保画布已初始化
            canvas_width = 500
            canvas_height = 400

        img_height, img_width = image.shape[:2]

        # 计算调整比例
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # 调整图像大小
        resized_img = cv2.resize(image, (new_width, new_height))

        # 如果显示预览，在原图上绘制轮廓
        if self.show_preview and self.current_contour is not None:
            # 计算轮廓在调整后图像上的位置
            contour_ratio = ratio
            # 创建副本以避免修改原图
            display_img = resized_img.copy()
            
            # 缩放轮廓点
            scaled_contour = np.array(self.current_contour * contour_ratio, dtype=np.int32)
            
            # 绘制轮廓
            cv2.drawContours(display_img, [scaled_contour], -1, (0, 255, 0), 2)
            
            # 计算带边距的边界框
            margin_pixels = int(self.margin_mm * self.mm_to_pixel_ratio)
            x, y, w, h = cv2.boundingRect(self.current_contour)
            
            # 缩放边界框
            x_scaled = int(x * ratio)
            y_scaled = int(y * ratio)
            w_scaled = int(w * ratio)
            h_scaled = int(h * ratio)
            margin_scaled = int(margin_pixels * ratio)
            
            # 绘制带边距的矩形
            x_margin = max(0, x_scaled - margin_scaled)
            y_margin = max(0, y_scaled - margin_scaled)
            w_margin = min(display_img.shape[1] - x_margin, w_scaled + 2 * margin_scaled)
            h_margin = min(display_img.shape[0] - y_margin, h_scaled + 2 * margin_scaled)
            
            # 绘制虚线矩形表示带边距的区域
            cv2.rectangle(display_img, (x_margin, y_margin), (x_margin + w_margin, y_margin + h_margin), 
                          (0, 0, 255), 2, cv2.LINE_AA)
        else:
            display_img = resized_img

        # 转换为PhotoImage
        img_pil = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # 清除画布并显示新图像
        self.original_canvas.delete("all")
        self.original_canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
        self.original_canvas.image = img_tk  # 保持引用，防止被垃圾回收
    
    def display_preview_image(self, image):
        """
        在预览画布上显示图像
        :param image: 输入图像
        """
        # 调整图像大小以适应画布
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width == 1 or canvas_height == 1:  # 确保画布已初始化
            canvas_width = 500
            canvas_height = 400

        img_height, img_width = image.shape[:2]

        # 计算调整比例
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # 调整图像大小
        resized_img = cv2.resize(image, (new_width, new_height))

        # 转换为PhotoImage
        img_pil = Image.fromarray(resized_img)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # 清除画布并显示新图像
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
        self.preview_canvas.image = img_tk  # 保持引用，防止被垃圾回收

    def toggle_preview(self):
        """
        切换预览显示状态
        """
        self.show_preview = self.preview_var.get()
        if self.show_preview:
            self.update_preview()
        else:
            # 清空预览窗口
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(self.preview_canvas.winfo_width()//2, 
                                           self.preview_canvas.winfo_height()//2, 
                                           text="请点击'显示预览'按钮查看实时预览", 
                                           fill="white", font=('SimHei', 12))
            self.current_contour = None
            self.processed_image = None
        
        # 更新原图显示
        if self.original_image is not None:
            self.display_original_image(self.original_image)
    
    def update_preview(self):
        """
        更新预览图像
        """
        if self.original_image is None:
            return
        
        try:
            # 先更新参数，确保使用最新设置
            self.update_all_params()
            
            self.status_var.set("正在更新预览...")
            self.root.update()

            # 检测边缘
            edge_mask = self.detect_edges(self.original_image)

            # 查找产品轮廓
            contour = self.find_product_contour(edge_mask)
            if contour is None:
                self.status_var.set("预览: 未找到产品轮廓")
                # 清空预览窗口并显示提示
                self.preview_canvas.delete("all")
                self.preview_canvas.create_text(self.preview_canvas.winfo_width()//2, 
                                               self.preview_canvas.winfo_height()//2, 
                                               text="未找到产品轮廓，请调整参数", 
                                               fill="white", font=('SimHei', 12))
                self.current_contour = None
                self.processed_image = None
                return
            
            # 保存当前轮廓
            self.current_contour = contour

            # 计算边距像素数
            margin_pixels = int(self.margin_mm * self.mm_to_pixel_ratio)

            # 裁切图像生成预览
            self.processed_image = self.crop_with_margin(self.original_image, contour, margin_pixels)

            # 显示预览图像
            self.display_preview_image(self.processed_image)
            self.status_var.set(f"预览已更新，边缘保留 {self.margin_mm}mm")
            
            # 更新原图显示，显示轮廓
            self.display_original_image(self.original_image)
        except Exception as e:
            self.status_var.set(f"预览更新失败: {str(e)}")
    
    def update_all_params(self):
        """
        更新所有轮廓识别参数（保持向后兼容）"""
        # 参数现在会自动应用，此方法仅保持向后兼容
        pass

    def create_color_mask(self, image):
        """
        创建颜色掩膜，用于过滤特定颜色（如粉红色背景）
        :param image: 输入图像
        :return: 颜色掩膜
        """
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 创建颜色范围的下界和上界
        lower_bound = np.array([self.hue_low, self.saturation_low, self.value_low])
        upper_bound = np.array([self.hue_high, self.saturation_high, self.value_high])
        
        # 创建颜色掩膜
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 反转掩膜，因为我们要保留的是产品而不是背景
        mask = cv2.bitwise_not(mask)
        
        # 形态学操作，去除小噪点
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def detect_edges(self, image):
        """
        检测图像边缘
        :param image: 输入图像
        :return: 边缘掩码
        """
        if self.use_color_mask:
            # 使用颜色掩膜
            color_mask = self.create_color_mask(image)
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 将颜色掩膜应用到灰度图
            masked_gray = cv2.bitwise_and(gray, gray, mask=color_mask)
            
            # 高斯模糊降噪
            blurred = cv2.GaussianBlur(masked_gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        else:
            # 传统边缘检测流程
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 高斯模糊降噪
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
        
        # 膨胀操作，连接断开的边缘
        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=self.dilate_iterations)
        
        return dilated

    def find_product_contour(self, edge_mask):
        """
        查找产品轮廓
        :param edge_mask: 边缘掩码
        :return: 最大轮廓
        """
        # 查找轮廓
        contours, _ = cv2.findContours(edge_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到面积最大的轮廓
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        return max_contour

    def crop_with_margin(self, image, contour, margin_pixels):
        """
        根据轮廓裁切图像并保留边距
        :param image: 输入图像
        :param contour: 产品轮廓
        :param margin_pixels: 边距像素数
        :return: 裁切后的图像
        """
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 计算带边距的边界框
        x_margin = max(0, x - margin_pixels)
        y_margin = max(0, y - margin_pixels)
        w_margin = min(image.shape[1] - x_margin, w + 2 * margin_pixels)
        h_margin = min(image.shape[0] - y_margin, h + 2 * margin_pixels)

        # 裁切图像
        cropped = image[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]

        return cropped

    def crop_edges(self):
        """
        裁切图像边缘
        """
        if self.original_image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        try:
            # 先更新参数，确保使用最新设置
            self.update_all_params()
            
            self.status_var.set("正在检测边缘...")
            self.root.update()

            # 检测边缘
            edge_mask = self.detect_edges(self.original_image)

            # 查找产品轮廓
            contour = self.find_product_contour(edge_mask)
            if contour is None:
                messagebox.showwarning("警告", "未找到产品轮廓")
                self.status_var.set("未找到产品轮廓")
                return

            # 保存当前轮廓
            self.current_contour = contour
            
            # 计算边距像素数
            margin_pixels = int(self.margin_mm * self.mm_to_pixel_ratio)

            # 裁切图像
            self.processed_image = self.crop_with_margin(self.original_image, contour, margin_pixels)

            # 显示处理后的图像
            self.display_preview_image(self.processed_image)
            # 更新原图显示，显示轮廓
            self.display_original_image(self.original_image)
            
            self.status_var.set(f"已完成裁切，边缘保留 {self.margin_mm}mm")

        except Exception as e:
            messagebox.showerror("错误", f"处理图像失败: {str(e)}")
            self.status_var.set("处理图像失败")

    def save_result(self):
        """
        保存处理后的图像
        """
        if self.processed_image is None:
            messagebox.showwarning("警告", "没有可保存的处理结果")
            return

        # 默认保存路径
        if self.image_path:
            file_name = os.path.basename(self.image_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            # 如果设置了输出目录，使用该目录
            if self.output_dir:
                default_save_path = os.path.join(self.output_dir, f"{file_name_without_ext}_cropped.jpg")
            else:
                default_save_path = os.path.join(os.path.dirname(self.image_path), f"{file_name_without_ext}_cropped.jpg")
        else:
            # 如果设置了输出目录，使用该目录
            if self.output_dir:
                default_save_path = os.path.join(self.output_dir, "cropped_result.jpg")
            else:
                default_save_path = "cropped_result.jpg"

        # 获取保存路径
        # 从default_save_path中提取文件名和目录
        save_dir = os.path.dirname(default_save_path)
        save_filename = os.path.basename(default_save_path)
        
        save_path = filedialog.asksaveasfilename(title="保存图像", defaultextension=".jpg",
                                                filetypes=[("JPEG文件", "*.jpg"), ("PNG文件", "*.png"), ("所有文件", "*.*")],
                                                initialdir=save_dir,
                                                initialfile=save_filename)

        if save_path:
            try:
                # 转换为BGR格式并保存
                bgr_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_image)
                self.status_var.set(f"已保存结果到: {save_path}")
                messagebox.showinfo("成功", f"图像已保存到: {save_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存图像失败: {str(e)}")
                self.status_var.set("保存图像失败")

def main():
    """
    主函数
    """
    # 创建并运行GUI
    root = tk.Tk()
    app = EdgeCropTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()