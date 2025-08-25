import cv2
import pytesseract
import os
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import threading
import numpy as np
import sys

sys.path.append(r"D:\pip_hub")

from paddleocr import PaddleOCR
import shutil
import qrcode  # 用于二维码操作，这里主要辅助识别
# from pyzbar.pyzbar import decode  # 可以通过 pip 安装

# 图片文件夹路径
image_folder = r'C:\Users\LHB\Pictures\OCR_Captures'
result_folder = r'C:\Users\LHB\Pictures\OCR_Results'
processed_folder = r'C:\Users\LHB\Pictures\Processed_Images'
temp_folder = r'C:\Users\LHB\Pictures\Temp_Results'
output_folder = r'C:\Users\LHB\Pictures\Output_Images'

# 创建结果文件夹、已处理文件夹和临时文件夹
os.makedirs(result_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 全局配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\run_a\AppData\Local\Programs\Tesseract - OCR\tesseract.exe'
save_directory = r'C:\Users\LHB\Pictures\OCR_Captures'
os.makedirs(save_directory, exist_ok=True)

# 全局状态管理
counter = 0  # 序号计数器
mouse_pos = (0, 0)  # 鼠标坐标
width, height, fps = 2560, 1920, 30  # 初始参数
fixed_width = 640  # 固定显示宽度
fixed_height = 360  # 固定显示高度
is_resolution_changing = False  # 标记是否正在切换分辨率
last_frame = None  # 用于缓存上一帧图像

# 初始化 PaddleOCR
try:
    # 更新 PaddleOCR 初始化参数，移除可能不兼容的参数
    ocr = PaddleOCR(lang='ch')
    print("PaddleOCR 初始化成功")
except Exception as e:
    print(f"PaddleOCR 初始化失败: {e}")
    messagebox.showerror("错误", f"PaddleOCR 初始化失败: {e}\n程序将继续运行，但OCR功能将不可用")
    ocr = None  # 设置为 None 而不是退出程序


def save_image(frame):
    """ 带序号的自动保存 """
    global counter
    while True:
        filename = f"captured_{counter:04d}.png"
        save_path = os.path.join(save_directory, filename)
        if not os.path.exists(save_path):
            break
        counter += 1
    try:
        cv2.imwrite(save_path, frame)
        print(f"Saved: {save_path}")
        counter += 1
        process_image(save_path)
        return save_path
    except Exception as e:
        print(f"保存图片失败: {e}")
        return None


def detect_label_auto(img):
    """自动检测标签区域，返回(x, y, w, h)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(denoised, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 选最大矩形轮廓
    best = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            best = c
    if best is not None:
        x, y, w, h = cv2.boundingRect(best)
        return (x, y, w, h)
    return None


def process_qr_code(img, points, output_path, qr_index):
    """
    拼图功能：原图+自动裁切标签，箭头指示
    """
    try:
        # 1. 计算标签位置
        label_rect = detect_label_auto(img)
        if not label_rect:
            print("未检测到标签，跳过拼图")
            return
        x_lbl, y_lbl, w_lbl, h_lbl = label_rect

        # 2. 自动裁切标签
        label_img = img[y_lbl:y_lbl + h_lbl, x_lbl:x_lbl + w_lbl]

        # 3. 准备新的白色背景
        margin = 50
        orig_h, orig_w = img.shape[:2]
        tag_h, tag_w = label_img.shape[:2]
        new_h = orig_h + tag_h + margin * 2
        new_w = max(orig_w, tag_w) + margin * 2
        new_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255

        # 4. 拼合图片
        # 原图放在新背景最上面（左上），铺满
        new_img[margin:margin + orig_h, margin:margin + orig_w] = img
        # 标签放在新背景下方，靠右对齐
        tag_x = new_w - tag_w - margin
        tag_y = margin + orig_h
        new_img[tag_y:tag_y + tag_h, tag_x:tag_x + tag_w] = label_img

        # 5. 生成指示性箭头
        # 原图标签中心
        orig_center = (margin + x_lbl + w_lbl // 2, margin + y_lbl + h_lbl // 2)
        # 新标签中心
        tag_center = (tag_x + tag_w // 2, tag_y + tag_h // 2)
        cv2.arrowedLine(new_img, orig_center, tag_center, (0, 0, 255), 4, tipLength=0.15)

        # 6. 保存结果
        cv2.imwrite(output_path, new_img)
        print(f"二维码+标签拼图已保存至: {output_path}")
    except Exception as e:
        print(f"二维码拼图处理失败: {e}")


def preprocess_image(img):
    """ 图像预处理函数，提高识别率 """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def process_image(image_path):
    print(f"正在处理图片文件: {image_path}")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        return
    
    # 检查文件是否为图片
    try:
        img_test = cv2.imread(image_path)
        if img_test is None:
            print(f"错误: 无法读取图片 - {image_path}")
            return
    except Exception as e:
        print(f"错误: 读取图片失败 - {image_path}, 错误: {e}")
        return

    # 进行文字识别
    ocr_result = []
    if ocr is not None:
        try:
            # 添加图片尺寸检查和预处理
            img = cv2.imread(image_path)
            max_side = max(img.shape[0], img.shape[1])
            if max_side > 4000:
                # 计算缩放比例
                scale = 4000 / max_side
                new_width = int(img.shape[1] * scale)
                new_height = int(img.shape[0] * scale)
                # 保存原始图片的副本
                original_img = img.copy()
                # 缩放图片
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                # 保存缩放后的图片到临时路径
                temp_path = os.path.join(temp_folder, f"resized_{os.path.basename(image_path)}")
                cv2.imwrite(temp_path, img)
                image_path = temp_path  # 使用缩放后的图片路径进行OCR处理
            
            # 使用新的 predict 方法替代过时的 ocr 方法
            ocr_result = ocr.predict(image_path)
            print(f"文字识别成功: {image_path}")
            print("识别结果:", ocr_result)
        except Exception as e:
            print(f"文字识别失败: {e}")
            messagebox.showwarning("警告", f"文字识别失败: {e}")
            # 继续执行，不要直接返回，以便处理二维码
    else:
        print("OCR 引擎未初始化，跳过文字识别")
        messagebox.showinfo("信息", "OCR 引擎未初始化，跳过文字识别")

    # 进行二维码和条形码识别
    qr_results = []
    try:
        img = cv2.imread(image_path)

        # 使用 cv2.QRCodeDetector 进行识别
        detector = cv2.QRCodeDetector()
        # 改进二维码识别方法，先尝试直接识别原图，再尝试预处理后的图像
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img)
        
        # 如果直接识别失败，尝试使用预处理后的图像
        if not retval or not any(decoded_info):
            preprocessed_img = preprocess_image(img)
            retval, decoded_info, points, _ = detector.detectAndDecodeMulti(preprocessed_img)
            
        if retval:
            for qr_index, (barcode_data, point) in enumerate(zip(decoded_info, points)):
                if barcode_data:
                    print(f"识别到的二维码: {barcode_data}")
                    qr_results.append(barcode_data)
                    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_output_{qr_index}.png")
                    process_qr_code(img, point, output_path, qr_index)
        else:
            # 尝试使用 pyzbar 库进行识别（如果可用）
            try:
                from pyzbar.pyzbar import decode
                decoded_objects = decode(img)
                if decoded_objects:
                    for i, obj in enumerate(decoded_objects):
                        barcode_data = obj.data.decode("utf-8")
                        print(f"通过pyzbar识别到的二维码/条形码: {barcode_data}")
                        qr_results.append(barcode_data)
                        points = np.array([obj.polygon], np.int32)
                        output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_output_pyzbar_{i}.png")
                        process_qr_code(img, points, output_path, i)
                else:
                    print("未识别到二维码/条形码")
            except ImportError:
                print("pyzbar库未安装，无法使用此方法识别二维码/条形码")
            except Exception as e:
                print(f"使用pyzbar识别二维码/条形码失败: {e}")

    except Exception as e:
        print(f"二维码/条形码识别失败: {e}")

    # 保存识别结果到临时文件夹
    temp_result_file_path = os.path.join(temp_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_result.txt")
    try:
        with open(temp_result_file_path, 'w', encoding='utf-8') as result_file:
            # 保存文字识别结果
            if ocr_result and len(ocr_result) > 0:
                try:
                    for res in ocr_result:
                        if isinstance(res, list):
                            for line in res:
                                if isinstance(line, list) and len(line) > 1 and isinstance(line[1], list) and len(line[1]) > 0:
                                    result_file.write(str(line[1][0]) + '\n')
                except Exception as e:
                    print(f"处理OCR结果时出错: {e}")
                    result_file.write(f"OCR结果格式错误: {str(ocr_result)}\n")
            else:
                result_file.write("未识别到文字内容\n")
                
            # 保存二维码/条形码识别结果
            if qr_results:
                result_file.write("\n--- 二维码/条形码内容 ---\n")
                for qr_result in qr_results:
                    result_file.write(qr_result + '\n')
            else:
                result_file.write("\n未识别到二维码/条形码\n")
                
        print(f"识别结果已保存到临时文件: {temp_result_file_path}")
    except Exception as e:
        print(f"保存识别结果失败: {e}")
        return

    # 检查结果文件是否已存在
    result_file_path = os.path.join(result_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_result.txt")
    if os.path.exists(result_file_path):
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
    else:
        try:
            shutil.move(temp_result_file_path, result_file_path)
            print(f"结果文件已移动到: {result_file_path}")
        except Exception as e:
            print(f"移动临时结果文件失败: {e}")

    # 移动已处理的图片到新的文件夹
    try:
        shutil.copy(image_path, os.path.join(processed_folder, os.path.basename(image_path)))
        print(f"图片已复制到: {os.path.join(processed_folder, os.path.basename(image_path))}")
    except Exception as e:
        print(f"复制图片失败: {e}")


def button_callback(action):
    """ 按钮功能映射 """
    global width, height, fps, last_frame
    if action == 'save':
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                save_image(frame)
            else:
                messagebox.showwarning("警告", "无法从摄像头获取图像")
        else:
            messagebox.showwarning("警告", "摄像头未打开")
    elif action == 'open':
        # 打开文件对话框选择图片
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            try:
                # 读取并处理图片
                process_image(file_path)
                # 显示图片
                img = cv2.imread(file_path)
                if img is not None:
                    # 更新last_frame以在界面上显示
                    last_frame = cv2.resize(img, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
                    messagebox.showinfo("成功", f"已处理图片: {file_path}")
                else:
                    messagebox.showerror("错误", f"无法读取图片: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"处理图片时出错: {str(e)}")
    elif action == 'process_folder':
        # 打开文件夹对话框
        from tkinter import filedialog
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder_path:
            try:
                # 获取文件夹中的所有图片
                image_files = []
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    image_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(ext)])
                
                if not image_files:
                    messagebox.showinfo("信息", "所选文件夹中没有找到图片文件")
                    return
                
                # 询问用户是否处理所有图片
                if messagebox.askyesno("确认", f"找到 {len(image_files)} 个图片文件，是否全部处理？"):
                    # 创建进度条窗口
                    progress_window = tk.Toplevel(root)
                    progress_window.title("处理进度")
                    progress_window.geometry("300x100")
                    
                    # 添加进度条
                    progress_label = tk.Label(progress_window, text="正在处理图片...")
                    progress_label.pack(pady=10)
                    progress_bar = ttk.Progressbar(progress_window, length=250, mode="determinate")
                    progress_bar.pack(pady=10)
                    
                    # 处理所有图片
                    processed_count = 0
                    for img_path in image_files:
                        try:
                            process_image(img_path)
                            processed_count += 1
                            # 更新进度条
                            progress_bar["value"] = (processed_count / len(image_files)) * 100
                            progress_label.config(text=f"正在处理: {processed_count}/{len(image_files)}")
                            progress_window.update()
                        except Exception as e:
                            print(f"处理图片 {img_path} 时出错: {str(e)}")
                    
                    # 完成后关闭进度条窗口
                    progress_window.destroy()
                    messagebox.showinfo("完成", f"已成功处理 {processed_count}/{len(image_files)} 个图片")
            except Exception as e:
                messagebox.showerror("错误", f"处理文件夹时出错: {str(e)}")
    elif action == 'exit':
        if cap.isOpened():
            cap.release()
        root.quit()


def update_resolution_async(resolution):
    """ 异步更新分辨率 """
    global width, height, is_resolution_changing
    is_resolution_changing = True
    width, height = map(int, resolution.split('x'))
    # 尝试设置分辨率，若失败则恢复之前的分辨率
    if not set_camera_resolution(cap, width, height):
        width, height = get_camera_resolution(cap)
    import time
    time.sleep(1)
    is_resolution_changing = False


def update_resolution(resolution):
    """ 启动异步线程更新分辨率 """
    threading.Thread(target=update_resolution_async, args=(resolution,)).start()


def on_key(event):
    """ 键盘事件处理 """
    if event.char.lower() == 'q':
        cap.release()
        root.quit()
    elif event.char.lower() =='s':
        ret, frame = cap.read()
        if ret:
            save_image(frame)


def on_mouse_move(event):
    """ 处理鼠标移动事件，更新鼠标位置 """
    global mouse_pos
    mouse_pos = (event.x, event.y)


def update_frame():
    global is_resolution_changing, last_frame
    
    # 检查摄像头是否打开
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if last_frame is not None:
                frame = last_frame.copy()
            else:
                frame = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)
                cv2.putText(frame, "无法读取摄像头", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            last_frame = frame.copy()
    else:
        # 摄像头未打开，使用上一帧或创建空白帧
        if last_frame is not None:
            frame = last_frame.copy()
        else:
            frame = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)
            cv2.putText(frame, "摄像头未打开", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if is_resolution_changing:
        # 正在切换分辨率，显示加载提示
        loading_text = "正在切换分辨率，请稍候..."
        frame = cv2.putText(frame, loading_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # 缩放图像到固定大小
        dim = (fixed_width, fixed_height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # 显示鼠标坐标在右下角，使用默认字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"X:{mouse_pos[0]} Y:{mouse_pos[1]}"
        text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = frame.shape[0] - 10
        cv2.putText(frame, text, (text_x, text_y), font, 0.5, (0, 0, 0), 1)

    # 将 OpenCV 图像转换为 PIL 图像
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    # 更新 Tkinter 标签中的图像
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)


def set_camera_resolution(cap, new_width, new_height):
    """ 设置摄像头分辨率，成功返回 True，失败返回 False """
    return cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)


def get_camera_resolution(cap):
    """ 获取当前摄像头分辨率 """
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# 初始化 Tkinter 窗口
root = tk.Tk()
root.title("OCR Capture")

# 创建一个新的框架用于放置按钮
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, fill=tk.X)

# 创建按钮并放置在新的框架中
btn_save = tk.Button(button_frame, text="Save", command=lambda: button_callback('save'))
btn_save.pack(side=tk.LEFT)

btn_open = tk.Button(button_frame, text="Open Image", command=lambda: button_callback('open'))
btn_open.pack(side=tk.LEFT)

btn_process_folder = tk.Button(button_frame, text="Process Folder", command=lambda: button_callback('process_folder'))
btn_process_folder.pack(side=tk.LEFT)

btn_exit = tk.Button(button_frame, text="Exit", command=lambda: button_callback('exit'))
btn_exit.pack(side=tk.LEFT)

# 创建下拉菜单用于选择分辨率
resolution_var = tk.StringVar(value="2560x1920")  # 设置默认分辨率为高分辨率
resolutions = ["1280x720", "1920x1080", "2560x1920"]
resolution_menu = ttk.Combobox(button_frame, textvariable=resolution_var, values=resolutions)
resolution_menu.pack(side=tk.LEFT)
resolution_menu.bind("<<ComboboxSelected>>", lambda event: update_resolution(resolution_var.get()))

# 创建标签用于显示图像
label = tk.Label(root)
label.pack()
# 绑定鼠标移动事件
label.bind("<Motion>", on_mouse_move)

# 绑定键盘事件
root.bind('<Key>', on_key)

# OpenCV 窗口初始化
try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        raise Exception("摄像头未能成功打开")
        
    # 初始化 last_frame
    ret, last_frame = cap.read()
    if ret:
        last_frame = cv2.resize(last_frame, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
    else:
        # 如果无法读取帧，创建一个空白帧
        last_frame = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)
        print("警告: 无法从摄像头读取图像，将使用空白图像")
except Exception as e:
    print(f"摄像头初始化失败: {e}")
    # 创建一个空的VideoCapture对象
    cap = cv2.VideoCapture()
    # 创建一个空白帧
    last_frame = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)
    # 在空白帧上显示提示文字
    cv2.putText(last_frame, "摄像头不可用", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 更新图像帧
update_frame()

# 启动 Tkinter 主循环
root.mainloop()