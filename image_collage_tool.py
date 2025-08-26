import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import math
from functools import partial

class ImageCollageTool:
    def __init__(self, root):
        self.root = root
        self.root.title("图片拼接工具")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # 配置变量
        self.image_folder = tk.StringVar()
        self.image_folder.set(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
        self.output_folder = tk.StringVar()
        self.output_folder.set(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output_Images"))
        
        # 图像相关变量
        self.images = []  # 所有加载的图像列表 [(image_path, cv2_image, pil_image, thumbnail), ...]
        self.main_image_index = -1  # 主图索引
        self.selected_images = []  # 选中的图像索引列表
        self.canvas_images = []  # 画布上的图像对象
        self.dragging_image = None  # 当前拖拽的图像索引
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.canvas_scale = 1.0  # 画布缩放比例
        self.main_image_scale_threshold = tk.DoubleVar(value=0.7)  # 主图缩放阈值，默认为0.7
        
        # 创建输出文件夹
        os.makedirs(self.output_folder.get(), exist_ok=True)
        
        # 创建UI
        self.create_ui()
        
        # 绑定事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 确保画布加载完成后立即更新预览
        self.root.after(100, self.initialize_preview)
        
    def create_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部控制区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件夹选择
        ttk.Label(control_frame, text="图片文件夹:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.image_folder, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(control_frame, text="浏览", command=self.browse_folder).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="加载图片", command=self.load_images).grid(row=0, column=3, padx=5, pady=5)
        
        # 输出文件夹选择
        ttk.Label(control_frame, text="输出文件夹:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.output_folder, width=50).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(control_frame, text="浏览", command=self.browse_output_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="拼接图片", command=self.create_collage).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除选择", command=self.clear_selection).pack(side=tk.LEFT, padx=5)
        
        # 图片碰撞时的缩放比例设置
        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(threshold_frame, text="图片碰撞时的缩放比例:").pack(side=tk.LEFT, padx=5)
        
        # 创建用于显示当前阈值的变量
        self.threshold_display = tk.StringVar()
        self.threshold_display.set(f"{self.main_image_scale_threshold.get():.1f}")
        
        # 创建滑块并绑定值变化事件
        threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                   variable=self.main_image_scale_threshold, length=200,
                                   command=lambda e: [self.update_threshold_display(), self.update_preview()])
        threshold_scale.pack(side=tk.LEFT, padx=5)
        
        # 显示当前阈值
        ttk.Label(threshold_frame, textvariable=self.threshold_display).pack(side=tk.LEFT, padx=5)
        ttk.Button(threshold_frame, text="应用", command=self.update_preview).pack(side=tk.LEFT, padx=5)
        
        # 创建左侧缩略图区域和右侧预览区域的分割窗口
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左侧缩略图区域
        thumbnail_frame = ttk.Frame(paned_window)
        paned_window.add(thumbnail_frame, weight=1)
        
        # 缩略图标题
        ttk.Label(thumbnail_frame, text="图片列表 (点击选择主图，Ctrl+点击选择附图)").pack(anchor=tk.W, padx=5, pady=5)
        
        # 缩略图滚动区域
        thumbnail_scroll = ttk.Scrollbar(thumbnail_frame, orient=tk.VERTICAL)
        thumbnail_canvas = tk.Canvas(thumbnail_frame, yscrollcommand=thumbnail_scroll.set)
        thumbnail_scroll.config(command=thumbnail_canvas.yview)
        
        thumbnail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        thumbnail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 缩略图内容框架
        self.thumbnail_content = ttk.Frame(thumbnail_canvas)
        thumbnail_canvas.create_window((0, 0), window=self.thumbnail_content, anchor=tk.NW)
        
        self.thumbnail_content.bind("<Configure>", lambda e: thumbnail_canvas.configure(scrollregion=thumbnail_canvas.bbox("all")))
        
        # 右侧预览区域
        preview_frame = ttk.Frame(paned_window)
        paned_window.add(preview_frame, weight=3)
        
        # 预览标题
        ttk.Label(preview_frame, text="预览 (拖动图片调整位置)").pack(anchor=tk.W, padx=5, pady=5)
        
        # 预览画布
        self.preview_canvas = tk.Canvas(preview_frame, bg="lightgray")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 绑定画布事件
        self.preview_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.preview_canvas.bind("<MouseWheel>", self.on_canvas_scroll)  # Windows滚轮事件
        self.preview_canvas.bind("<Button-4>", self.on_canvas_scroll)  # Linux上滚事件
        self.preview_canvas.bind("<Button-5>", self.on_canvas_scroll)  # Linux下滚事件
        
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder.set(folder)
            
    def browse_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)
    
    def load_images(self):
        # 清除现有图像
        self.clear_all()
        
        folder = self.image_folder.get()
        if not os.path.isdir(folder):
            messagebox.showerror("错误", f"文件夹不存在: {folder}")
            return
        
        # 获取文件夹中的所有图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            messagebox.showinfo("提示", "文件夹中没有找到图像文件")
            return
        
        # 加载图像
        for file in image_files:
            file_path = os.path.join(folder, file)
            try:
                # 使用OpenCV加载图像
                cv_img = cv2.imread(file_path)
                if cv_img is None:
                    continue
                
                # 转换为RGB格式
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                # 创建PIL图像
                pil_img = Image.fromarray(cv_img_rgb)
                
                # 创建缩略图
                thumbnail_size = (100, 100)
                thumbnail = pil_img.copy()
                thumbnail.thumbnail(thumbnail_size)
                thumbnail = ImageTk.PhotoImage(thumbnail)
                
                # 添加到图像列表
                self.images.append((file_path, cv_img, pil_img, thumbnail))
            except Exception as e:
                print(f"加载图像失败: {file_path}, 错误: {e}")
        
        # 显示缩略图
        self.display_thumbnails()
        
    def display_thumbnails(self):
        # 清除现有缩略图
        for widget in self.thumbnail_content.winfo_children():
            widget.destroy()
        
        # 每行显示的缩略图数量
        thumbnails_per_row = 3
        
        # 显示缩略图
        for i, (file_path, _, _, thumbnail) in enumerate(self.images):
            # 计算行列位置
            row = i // thumbnails_per_row
            col = i % thumbnails_per_row
            
            # 创建缩略图框架
            frame = ttk.Frame(self.thumbnail_content, padding=5)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky=tk.NSEW)
            
            # 显示缩略图
            label = ttk.Label(frame, image=thumbnail)
            label.pack()
            
            # 显示文件名
            file_name = os.path.basename(file_path)
            if len(file_name) > 20:
                file_name = file_name[:17] + "..."
            ttk.Label(frame, text=file_name).pack()
            
            # 绑定点击事件
            label.bind("<Button-1>", partial(self.on_thumbnail_click, index=i))
            
    def on_thumbnail_click(self, event, index):
        # 检查是否按下Ctrl键
        ctrl_pressed = event.state & 0x4  # 0x4 是Ctrl键的掩码
        
        if not ctrl_pressed:
            # 单击选择主图
            self.main_image_index = index
            self.selected_images = [index]  # 清除其他选择
        else:
            # Ctrl+点击选择附图
            if index == self.main_image_index:
                # 如果点击的是主图，不做任何操作
                return
            
            if index in self.selected_images:
                # 如果已经选中，则取消选择
                self.selected_images.remove(index)
            else:
                # 否则添加到选择列表
                self.selected_images.append(index)
        
        # 更新预览
        self.update_preview()
        
    def update_preview(self):
        # 清除画布
        self.preview_canvas.delete("all")
        self.canvas_images = []
        
        if not self.selected_images:
            return
        
        # 获取画布尺寸
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # 画布尚未完全初始化，延迟更新
            self.root.after(100, self.update_preview)
            return
            
        # 重置缩放比例变量，确保每次更新都重新计算
        if hasattr(self, 'main_canvas_scale'):
            delattr(self, 'main_canvas_scale')
        if hasattr(self, 'attachment_canvas_scale'):
            delattr(self, 'attachment_canvas_scale')
        
        # 计算所有图像的尺寸
        image_sizes = []
        total_area = 0
        max_width = 0
        max_height = 0
        
        for i in self.selected_images:
            _, cv_img, _, _ = self.images[i]
            h, w = cv_img.shape[:2]
            image_sizes.append((w, h))
            total_area += w * h
            max_width = max(max_width, w)
            max_height = max(max_height, h)
        
        # 计算拼接画布的尺寸，确保有足够空间防止重叠
        # 使用更合理的布局计算方式，确保图片都在视野内
        num_images = len(self.selected_images)
        grid_size = math.ceil(math.sqrt(num_images))
        
        # 计算合适的缩放比例，确保所有图片都能在画布内显示
        # 留出20%的边距空间
        available_width = canvas_width * 0.8
        available_height = canvas_height * 0.8
        
        # 计算网格布局所需的总宽度和高度
        grid_width = max_width * 1.2  # 增加20%的间距
        grid_height = max_height * 1.2
        total_grid_width = grid_width * grid_size
        total_grid_height = grid_height * grid_size
        
        # 初始缩放比例设为1.0（原始大小）
        self.canvas_scale = 1.0
        
        # 初始位置
        positions = []
        
        # 主图放在左上角，附图放在右侧或下方
        if self.main_image_index in self.selected_images:
            main_idx = self.selected_images.index(self.main_image_index)
            _, main_cv_img, _, _ = self.images[self.main_image_index]
            main_h, main_w = main_cv_img.shape[:2]
            
            # 主图位置 - 左上角，但留出边距
            main_x = canvas_width * 0.1
            main_y = canvas_height * 0.1
            positions.append((main_x, main_y))
            
            # 计算其他图像的位置，避免重叠
            remaining_indices = [i for i in range(len(self.selected_images)) if i != main_idx]
            
            # 计算主图初始大小 - 占预览窗口的2/5
            target_main_width = canvas_width * 0.4  # 2/5 的窗口宽度
            target_main_height = canvas_height * 0.4  # 2/5 的窗口高度
            
            # 计算主图的缩放比例，保持宽高比
            main_scale_x = target_main_width / main_w
            main_scale_y = target_main_height / main_h
            initial_main_scale = min(main_scale_x, main_scale_y)
            
            # 直接设置主图初始缩放比例，确保主图大小正确
            main_scale = initial_main_scale
            self.main_canvas_scale = main_scale  # 立即设置主图缩放比例
            
            # 计算主图的边界，使用实际的缩放比例
            main_right = main_x + main_w * main_scale
            main_bottom = main_y + main_h * main_scale
            
            # 为附图创建网格布局
            remaining_grid_size = math.ceil(math.sqrt(len(remaining_indices)))
            
            # 计算网格单元大小
            max_attachment_width = max([image_sizes[i][0] for i in range(len(image_sizes)) if i != main_idx], default=0)
            max_attachment_height = max([image_sizes[i][1] for i in range(len(image_sizes)) if i != main_idx], default=0)
            
            # 确保网格单元足够大，防止图片重叠
            attachment_grid_width = max_attachment_width * 1.2
            attachment_grid_height = max_attachment_height * 1.2
            
            # 决定附图布局方向：附图始终放在右下角
            # 计算主图的右下角坐标
            main_right = main_x + main_w * main_scale
            main_bottom = main_y + main_h * main_scale
            
            # 确保附图与主图有足够的间距
            start_x = max(canvas_width * 0.6, main_right + 20)  # 从画布60%宽度开始或主图右侧加20像素
            start_y = max(canvas_height * 0.6, main_bottom + 20)  # 从画布60%高度开始或主图底部加20像素
            
            # 计算附图初始大小 - 占预览窗口的1/4
            target_attachment_width = canvas_width * 0.25  # 1/4 的窗口宽度
            target_attachment_height = canvas_height * 0.25  # 1/4 的窗口高度
            
            # 计算附图的缩放比例
            if max_attachment_width > 0 and max_attachment_height > 0:
                attachment_scale_x = target_attachment_width / max_attachment_width
                attachment_scale_y = target_attachment_height / max_attachment_height
                attachment_scale = min(attachment_scale_x, attachment_scale_y)
                
                # 确保附图不会太小，至少为主图的1/3大小
                min_attachment_scale = initial_main_scale / 3
                attachment_scale = max(attachment_scale, min_attachment_scale)
            else:
                attachment_scale = initial_main_scale / 3  # 默认为主图的1/3大小
            
            # 主图始终使用初始缩放比例（2/5窗口大小）
            # 只有在超出画布边界时才缩小
            if main_x + main_w * initial_main_scale > canvas_width * 0.9 or main_y + main_h * initial_main_scale > canvas_height * 0.9:
                # 主图太大，需要缩放以适应画布
                main_scale_x = (canvas_width * 0.6) / main_w  # 留出40%空间给附图
                main_scale_y = (canvas_height * 0.6) / main_h
                main_scale = min(main_scale_x, main_scale_y)  # 不放大，只缩小
            else:
                # 否则使用初始缩放比例
                main_scale = initial_main_scale
                
            # 确保主图不会太小
            min_main_scale = 0.1  # 最小缩放比例
            if main_scale < min_main_scale:
                main_scale = min_main_scale
            
            # 检查附图是否超出画布边界
            # 使用初始计算的附图缩放比例
            max_x = start_x + (attachment_grid_width * remaining_grid_size * attachment_scale)
            max_y = start_y + (attachment_grid_height * remaining_grid_size * attachment_scale)
            
            # 只有在超出画布边界时才缩小附图
            if max_x > canvas_width * 0.95 or max_y > canvas_height * 0.95:
                # 附图太大，需要缩放
                scale_x = (canvas_width * 0.95 - start_x) / (attachment_grid_width * remaining_grid_size)
                scale_y = (canvas_height * 0.95 - start_y) / (attachment_grid_height * remaining_grid_size)
                attachment_scale = min(scale_x, scale_y, attachment_scale)  # 不放大，只缩小
                
            # 确保附图不会太小
            min_attachment_scale = 0.05  # 最小缩放比例
            if attachment_scale < min_attachment_scale:
                attachment_scale = min_attachment_scale
            
            # 检测主图和附图是否发生碰撞
            # 计算主图的边界
            main_right = main_x + main_w * main_scale
            main_bottom = main_y + main_h * main_scale
            
            # 计算附图的边界
            attachment_left = start_x
            attachment_top = start_y
            attachment_right = start_x + (attachment_grid_width * remaining_grid_size * attachment_scale)
            attachment_bottom = start_y + (attachment_grid_height * remaining_grid_size * attachment_scale)
            
            # 检查是否发生碰撞
            collision_detected = False
            
            # 如果附图区域与主图区域有重叠，则认为发生碰撞
            if (attachment_left < main_right and attachment_right > main_x and
                attachment_top < main_bottom and attachment_bottom > main_y):
                collision_detected = True
            
            # 只有在发生碰撞时才同步缩小主图和附图
            if collision_detected:
                # 计算需要的缩放比例
                # 使用碰撞时的缩放阈值
                collision_scale_factor = self.main_image_scale_threshold.get()
                
                # 应用缩放比例
                main_scale = initial_main_scale * collision_scale_factor
                attachment_scale = attachment_scale * collision_scale_factor
            
            # 存储主图和附图的缩放比例，确保在整个方法中一致使用
            self.main_canvas_scale = main_scale
            self.attachment_canvas_scale = attachment_scale
            
            for i, idx in enumerate(remaining_indices):
                grid_x = i % remaining_grid_size
                grid_y = i // remaining_grid_size
                
                # 计算图片中心位置，使用附图的缩放比例
                # 直接使用当前计算的attachment_scale而不是self.attachment_canvas_scale
                x = start_x + attachment_grid_width * grid_x * attachment_scale + (attachment_grid_width * attachment_scale) / 2
                y = start_y + attachment_grid_height * grid_y * attachment_scale + (attachment_grid_height * attachment_scale) / 2
                
                positions.append((x, y))
        else:
            # 如果没有选择主图，则使用均匀网格布局
            # 计算网格布局的起始位置，使其居中
            start_x = (canvas_width - total_grid_width * self.canvas_scale) / 2 + (grid_width * self.canvas_scale) / 2
            start_y = (canvas_height - total_grid_height * self.canvas_scale) / 2 + (grid_height * self.canvas_scale) / 2
            
            for i in range(len(self.selected_images)):
                grid_x = i % grid_size
                grid_y = i // grid_size
                
                # 计算图片中心位置
                x = start_x + grid_width * self.canvas_scale * grid_x
                y = start_y + grid_height * self.canvas_scale * grid_y
                
                positions.append((x, y))
        
        # 在画布上显示图像
        for i, idx in enumerate(self.selected_images):
            file_path, cv_img, pil_img, _ = self.images[idx]
            
            # 根据图像类型选择缩放比例
            if idx == self.main_image_index and hasattr(self, 'main_canvas_scale'):
                # 主图使用主图缩放比例
                current_scale = self.main_canvas_scale
            elif hasattr(self, 'attachment_canvas_scale'):
                # 附图使用附图缩放比例
                current_scale = self.attachment_canvas_scale
            else:
                # 默认使用全局缩放比例
                current_scale = self.canvas_scale
            
            # 调整图像大小
            h, w = cv_img.shape[:2]
            new_w = int(w * current_scale)
            new_h = int(h * current_scale)
            
            # 创建缩放后的图像
            resized_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            photo_img = ImageTk.PhotoImage(resized_img)
            
            # 获取位置
            x, y = positions[i]
            
            # 在画布上创建图像
            image_id = self.preview_canvas.create_image(x, y, image=photo_img, anchor=tk.CENTER)
            
            # 保存引用以防止垃圾回收
            self.canvas_images.append((image_id, photo_img, x, y, idx))
            
            # 如果是主图，添加标记
            if idx == self.main_image_index:
                self.preview_canvas.create_text(x, y - new_h/2 - 10, text="主图", fill="red", font=("Arial", 12, "bold"))
        
        # 在所有图像显示后，检查并解决碰撞问题
        if len(self.canvas_images) > 1:
            self.update_image_positions_after_collision()
    
    def on_canvas_press(self, event):
        # 检查是否点击了图像
        x, y = event.x, event.y
        for i, (image_id, _, img_x, img_y, idx) in enumerate(self.canvas_images):
            # 获取图像的边界框
            bbox = self.preview_canvas.bbox(image_id)
            if bbox and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                self.dragging_image = i
                self.drag_start_x = x
                self.drag_start_y = y
                break
    
    def on_canvas_drag(self, event):
        if self.dragging_image is not None:
            # 计算移动距离
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            
            # 更新图像位置
            image_id, photo_img, img_x, img_y, idx = self.canvas_images[self.dragging_image]
            new_x = img_x + dx
            new_y = img_y + dy
            
            # 获取画布尺寸
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # 获取图像边界框
            bbox = self.preview_canvas.bbox(image_id)
            if bbox:
                img_width = bbox[2] - bbox[0]
                img_height = bbox[3] - bbox[1]
                
                # 限制图像不能超出画布边界
                # 左边界
                if new_x - img_width/2 < 0:
                    new_x = img_width/2
                # 右边界
                elif new_x + img_width/2 > canvas_width:
                    new_x = canvas_width - img_width/2
                # 上边界
                if new_y - img_height/2 < 0:
                    new_y = img_height/2
                # 下边界
                elif new_y + img_height/2 > canvas_height:
                    new_y = canvas_height - img_height/2
            
            # 移动图像，只改变位置，不改变大小
            self.preview_canvas.coords(image_id, new_x, new_y)
            
            # 更新拖拽起始位置和图像位置
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.canvas_images[self.dragging_image] = (image_id, photo_img, new_x, new_y, idx)
            
            # 如果拖动的是主图，确保其他图像不会与之重叠
            if idx == self.main_image_index and len(self.canvas_images) > 1:
                # 不立即调用碰撞检测，等待拖动结束后再处理，避免拖动过程中图像大小变化
                pass
    
    def on_canvas_release(self, event):
        # 拖动结束后，检查是否需要更新图像位置
        dragged_image = self.dragging_image
        self.dragging_image = None
        
        # 如果有多个图像，检查并解决碰撞问题
        if len(self.canvas_images) > 1:
            self.update_image_positions_after_collision()
    
    def on_canvas_scroll(self, event):
        # 处理缩放
        if event.num == 4 or event.delta > 0:
            # 放大
            self.canvas_scale *= 1.1
        elif event.num == 5 or event.delta < 0:
            # 缩小
            self.canvas_scale *= 0.9
        
        # 更新预览
        self.update_preview()
    
    def create_collage(self):
        if not self.selected_images:
            messagebox.showinfo("提示", "请先选择图片")
            return
        
        # 获取所有选中图像的位置和尺寸
        image_data = []
        
        # 获取画布尺寸
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        for image_id, _, x, y, idx in self.canvas_images:
            # 获取图像在画布上的边界框
            bbox = self.preview_canvas.bbox(image_id)
            if not bbox:
                continue
            
            # 计算图像在画布上的宽度和高度
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # 计算左上角坐标
            left_x = x - width / 2
            top_y = y - height / 2
            
            # 确保图像不超出画布边界
            if left_x < 0:
                left_x = 0
            if top_y < 0:
                top_y = 0
            if left_x + width > canvas_width:
                left_x = canvas_width - width
            if top_y + height > canvas_height:
                top_y = canvas_height - height
            
            # 获取原始图像
            _, cv_img, _, _ = self.images[idx]
            
            # 添加到图像数据列表
            image_data.append({
                'image': cv_img,
                'x': left_x,  # 使用调整后的左上角坐标
                'y': top_y,
                'width': width,
                'height': height,
                'is_main': idx == self.main_image_index,
                'idx': idx  # 保存原始索引用于排序
            })
        
        # 检测并解决重叠问题
        self.resolve_overlaps(image_data)
        
        # 计算拼接图像的尺寸
        min_x = min(data['x'] for data in image_data)
        min_y = min(data['y'] for data in image_data)
        max_x = max(data['x'] + data['width'] for data in image_data)
        max_y = max(data['y'] + data['height'] for data in image_data)
        
        collage_width = int(max_x - min_x)
        collage_height = int(max_y - min_y)
        
        # 创建空白拼接图像
        collage = np.ones((collage_height, collage_width, 3), dtype=np.uint8) * 255
        
        # 按照图像的重要性排序（主图优先，然后是其他图像）
        image_data.sort(key=lambda x: (0 if x['is_main'] else 1, x['idx']))
        
        # 创建一个掩码来跟踪已经填充的区域
        mask = np.zeros((collage_height, collage_width), dtype=np.uint8)
        
        # 将图像放置到拼接图像上
        for data in image_data:
            # 计算图像在拼接图像上的位置
            x = int(data['x'] - min_x)
            y = int(data['y'] - min_y)
            
            # 调整图像大小
            resized = cv2.resize(data['image'], (int(data['width']), int(data['height'])))
            
            # 创建当前图像的掩码
            current_mask = np.zeros((collage_height, collage_width), dtype=np.uint8)
            
            # 确保坐标在有效范围内
            if x < 0: x = 0
            if y < 0: y = 0
            
            # 确保不超出边界
            h, w = resized.shape[:2]
            if y + h > collage_height: h = collage_height - y
            if x + w > collage_width: w = collage_width - x
            
            # 更新当前图像的掩码
            try:
                current_mask[y:y+h, x:x+w] = 1
            except ValueError as e:
                print(f"掩码错误: {e}")
                continue
            
            # 检查是否与已有图像重叠
            overlap = np.logical_and(mask, current_mask)
            if np.any(overlap):
                # 如果是主图，覆盖其他图像
                if data['is_main']:
                    # 放置图像
                    try:
                        collage[y:y+h, x:x+w] = resized[:h, :w]
                        # 更新掩码
                        mask[y:y+h, x:x+w] = 1
                    except ValueError as e:
                        print(f"放置主图错误: {e}")
                        continue
                else:
                    # 如果不是主图，只在未被占用的区域放置
                    try:
                        # 创建未重叠区域的掩码
                        non_overlap = np.logical_and(current_mask, np.logical_not(mask))
                        
                        # 只在未重叠区域放置图像
                        for i in range(h):
                            for j in range(w):
                                if y+i < collage_height and x+j < collage_width and non_overlap[y+i, x+j]:
                                    collage[y+i, x+j] = resized[i, j]
                        
                        # 更新掩码
                        mask = np.logical_or(mask, non_overlap)
                    except ValueError as e:
                        print(f"放置非主图错误: {e}")
                        continue
            else:
                # 如果没有重叠，直接放置图像
                try:
                    collage[y:y+h, x:x+w] = resized[:h, :w]
                    # 更新掩码
                    mask[y:y+h, x:x+w] = 1
                except ValueError as e:
                    print(f"放置图像错误: {e}")
                    print(f"位置: ({x}, {y}), 尺寸: {resized.shape}, 拼接图像尺寸: {collage.shape}")
                    continue
        
        # 保存拼接图像
        output_path = os.path.join(self.output_folder.get(), f"collage_{len(self.selected_images)}images.jpg")
        cv2.imwrite(output_path, collage)
        
        messagebox.showinfo("成功", f"拼接图像已保存到: {output_path}")
    
    def resolve_overlaps(self, image_data):
        """解决图像重叠问题"""
        # 按照图像的重要性排序（主图优先，然后是其他图像）
        image_data.sort(key=lambda x: (0 if x['is_main'] else 1, x['idx']))
        
        # 检测并解决重叠
        for i in range(len(image_data)):
            for j in range(i+1, len(image_data)):
                # 检查图像i和图像j是否重叠
                if self.is_overlapping(image_data[i], image_data[j]):
                    # 如果重叠，移动图像j
                    self.move_image_away(image_data[i], image_data[j])
    
    def is_overlapping(self, img1, img2):
        """检查两个图像是否重叠"""
        # 计算图像1的边界
        left1 = img1['x']
        right1 = img1['x'] + img1['width']
        top1 = img1['y']
        bottom1 = img1['y'] + img1['height']
        
        # 计算图像2的边界
        left2 = img2['x']
        right2 = img2['x'] + img2['width']
        top2 = img2['y']
        bottom2 = img2['y'] + img2['height']
        
        # 检查是否重叠
        return not (right1 <= left2 or left1 >= right2 or bottom1 <= top2 or top1 >= bottom2)
    
    def move_image_away(self, fixed_img, moving_img):
        """将moving_img移动到fixed_img旁边，避免重叠"""
        # 计算fixed_img的边界
        fixed_left = fixed_img['x']
        fixed_right = fixed_img['x'] + fixed_img['width']
        fixed_top = fixed_img['y']
        fixed_bottom = fixed_img['y'] + fixed_img['height']
        
        # 计算moving_img的边界
        moving_left = moving_img['x']
        moving_right = moving_img['x'] + moving_img['width']
        moving_top = moving_img['y']
        moving_bottom = moving_img['y'] + moving_img['height']
        
        # 计算四个方向的移动距离
        move_left = moving_left - fixed_right - 10  # 额外10像素的间距
        move_right = fixed_left - moving_right - 10
        move_up = moving_top - fixed_bottom - 10
        move_down = fixed_top - moving_bottom - 10
        
        # 找出最小的移动距离
        min_move = min(abs(move_left), abs(move_right), abs(move_up), abs(move_down))
        
        # 根据最小移动距离调整图像位置
        if min_move == abs(move_left) and move_left < 0:
            moving_img['x'] = fixed_right + 10
        elif min_move == abs(move_right) and move_right < 0:
            moving_img['x'] = fixed_left - moving_img['width'] - 10
        elif min_move == abs(move_up) and move_up < 0:
            moving_img['y'] = fixed_bottom + 10
        elif min_move == abs(move_down) and move_down < 0:
            moving_img['y'] = fixed_top - moving_img['height'] - 10
    
    def clear_selection(self):
        self.selected_images = []
        self.main_image_index = -1
        self.update_preview()
    
    def clear_all(self):
        self.images = []
        self.selected_images = []
        self.main_image_index = -1
        self.canvas_images = []
        self.preview_canvas.delete("all")
        
        # 清除缩略图
        for widget in self.thumbnail_content.winfo_children():
            widget.destroy()
    
    def initialize_preview(self):
        """初始化预览，确保画布加载完成后正确显示图像"""
        # 检查画布是否已经初始化
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # 画布尚未完全初始化，延迟更新
            self.root.after(100, self.initialize_preview)
            return
            
        # 如果已经有选中的图像，更新预览
        if self.selected_images:
            self.update_preview()
    
    def update_threshold_display(self, value=None):
        """更新碰撞时的缩放比例显示"""
        self.threshold_display.set(f"{self.main_image_scale_threshold.get():.1f}")
        
    def update_image_positions_after_collision(self):
        """碰撞检测后更新图像位置，确保它们不会重叠"""
        if not self.canvas_images or len(self.canvas_images) < 2:
            return
            
        # 获取主图和附图的位置和尺寸
        main_image = None
        attachment_images = []
        
        for image_id, photo_img, x, y, idx in self.canvas_images:
            bbox = self.preview_canvas.bbox(image_id)
            if not bbox:
                continue
                
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if idx == self.main_image_index:
                main_image = {'id': image_id, 'x': x, 'y': y, 'width': width, 'height': height}
            else:
                attachment_images.append({'id': image_id, 'x': x, 'y': y, 'width': width, 'height': height})
                
        if not main_image or not attachment_images:
            return
            
        # 检查主图和每个附图是否重叠，如果重叠则移动附图
        for attachment in attachment_images:
            # 计算主图的边界
            main_left = main_image['x'] - main_image['width'] / 2
            main_right = main_image['x'] + main_image['width'] / 2
            main_top = main_image['y'] - main_image['height'] / 2
            main_bottom = main_image['y'] + main_image['height'] / 2
            
            # 计算附图的边界
            att_left = attachment['x'] - attachment['width'] / 2
            att_right = attachment['x'] + attachment['width'] / 2
            att_top = attachment['y'] - attachment['height'] / 2
            att_bottom = attachment['y'] + attachment['height'] / 2
            
            # 检查是否重叠
            if (att_left < main_right and att_right > main_left and
                att_top < main_bottom and att_bottom > main_top):
                # 计算四个方向的移动距离
                move_left = att_left - main_right - 10  # 额外10像素的间距
                move_right = main_left - att_right - 10
                move_up = att_top - main_bottom - 10
                move_down = main_top - att_bottom - 10
                
                # 找出最小的移动距离
                min_move = min(abs(move_left), abs(move_right), abs(move_up), abs(move_down))
                
                # 根据最小移动距离调整图像位置
                new_x = attachment['x']
                new_y = attachment['y']
                
                if min_move == abs(move_left) and move_left < 0:
                    new_x = main_right + attachment['width'] / 2 + 10
                elif min_move == abs(move_right) and move_right < 0:
                    new_x = main_left - attachment['width'] / 2 - 10
                elif min_move == abs(move_up) and move_up < 0:
                    new_y = main_bottom + attachment['height'] / 2 + 10
                elif min_move == abs(move_down) and move_down < 0:
                    new_y = main_top - attachment['height'] / 2 - 10
                
                # 移动附图
                self.preview_canvas.coords(attachment['id'], new_x, new_y)
                
                # 更新canvas_images中的位置
                for i, (image_id, photo_img, x, y, idx) in enumerate(self.canvas_images):
                    if image_id == attachment['id']:
                        self.canvas_images[i] = (image_id, photo_img, new_x, new_y, idx)
                        break
        
    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCollageTool(root)
    root.mainloop()