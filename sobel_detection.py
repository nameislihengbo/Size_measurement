import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def preprocess_image_for_text(image_path):
    """
    预处理图像以增强白色背景黑色文字的对比度
    
    参数:
        image_path: 图像文件路径
    
    返回:
        gray: 灰度图像
        binary: 二值化图像
        original_img: 原始彩色图像
    """
    try:
        # 以二进制模式读取图像文件
        with open(image_path, 'rb') as f:
            img_data = f.read()
        # 将二进制数据转换为图像
        original_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        # 检查图像是否成功读取
        if original_img is None:
            raise ValueError(f"无法读取图像文件: {image_path}")
        # 转换为灰度图
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        # 应用高斯模糊降噪 - 使用较小的核大小以保留更多细节
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 减小核大小，保留更多细节
        
        # 二值化处理（自适应阈值）- 调整参数提高精度
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2  # 减小块大小和常数值
        )
        
        # 形态学操作 - 使用更小的核和更少的迭代次数
        kernel = np.ones((2, 2), np.uint8)  # 减小核大小
        # 先进行开操作（先腐蚀后膨胀），去除小噪点
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # 再进行闭操作（先膨胀后腐蚀），填充内部空洞，但使用较小的迭代次数
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    except Exception as e:
        raise RuntimeError(f"图像预处理失败: {str(e)}")
    
    return gray, binary, original_img


def extract_largest_rotated_rectangle(binary_image, original_image, min_area=100, max_area=500000, contour_mode=cv2.RETR_EXTERNAL):
    """
    从二值化图像中提取最大的外接矩形（支持任意角度标签），提高参数容差
    
    参数:
        binary_image: 二值化图像
        original_image: 原始彩色图像
        min_area: 最小区域面积（减小值以提高容差）
        max_area: 最大区域面积（增大值以提高容差）
        contour_mode: 轮廓检索模式（默认外部轮廓）
    
    返回:
        result_image: 绘制了最大外接矩形的图像
        largest_rect: 最大外接矩形信息，如果没有找到则为None
        largest_contour: 最大轮廓，如果没有找到则为None
    """
    # 复制原始图像以避免修改
    result_image = original_image.copy()
    
    # 查找轮廓 - 使用不同的轮廓检索模式，可选择RETR_EXTERNAL或RETR_LIST
    contours, _ = cv2.findContours(binary_image, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化最大轮廓和面积
    largest_contour = None
    max_contour_area = 0
    
    # 遍历所有轮廓，找到面积最大的轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 过滤面积过小或过大的轮廓 - 提高参数容差
        if area < min_area or area > max_area:
            continue
        
        # 更新最大轮廓
        if area > max_contour_area:
            max_contour_area = area
            largest_contour = contour
    
    largest_rect = None
    # 如果找到有效轮廓，计算其最小外接矩形
    if largest_contour is not None:
        # 轮廓收缩处理，使边界更贴合内容
        # 1. 先进行轮廓近似，减少噪点影响，使用更小的epsilon值提高精度
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)  # 减小epsilon值，提高精度
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 2. 轮廓收缩处理，向内收缩轮廓
        # 创建一个稍小的核用于腐蚀操作
        shrink_kernel = np.ones((2, 2), np.uint8)
        mask = np.zeros(binary_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [approx_contour], 0, 255, -1)  # 填充轮廓
        # 对轮廓进行腐蚀操作，使其向内收缩
        eroded_mask = cv2.erode(mask, shrink_kernel, iterations=1)
        # 重新查找收缩后的轮廓
        shrunk_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果收缩后的轮廓存在，使用收缩后的轮廓；否则使用原始近似轮廓
        if shrunk_contours and len(shrunk_contours[0]) > 4:  # 确保有足够的点来计算矩形
            # 计算最小外接矩形（支持任意角度）
            largest_rect = cv2.minAreaRect(shrunk_contours[0])
        else:
            # 计算最小外接矩形（支持任意角度）
            largest_rect = cv2.minAreaRect(approx_contour)
        
        # 获取矩形的四个顶点
        box = cv2.boxPoints(largest_rect)
        # 将浮点坐标转换为整数坐标
        box = np.int32(box)
        
        # 在图像上绘制矩形
        cv2.drawContours(result_image, [box], 0, (0, 255, 0), 3)  # 增加线条宽度，更清晰可见
        
        # 在矩形中心显示面积信息
        center_x, center_y = int(largest_rect[0][0]), int(largest_rect[0][1])
        cv2.putText(
            result_image, 
            f"最大区域 (面积: {int(max_contour_area)})", 
            (center_x - 100, center_y - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 255), 
            2
        )
    
    return result_image, largest_rect, largest_contour


def sobel_edge_detection(image_path, threshold=100):
    # 这个函数保留用于向后兼容，实际文字提取使用上面的函数
    gray, binary, img = preprocess_image_for_text(image_path)
    # 使用Canny边缘检测替代Sobel，更适合文字边缘
    edge = cv2.Canny(gray, 50, 150)
    return gray, None, None, edge, img


# 保留原函数用于向后兼容
# def extract_rotated_rectangles(edge_image, original_image):
#     ...

# 主函数
if __name__ == "__main__":
    # 替换为你的图像路径
    # 使用原始字符串并确保路径分隔符正确
    image_path = r"E:\工作区\Hops图片维护设备\1-客户资料\2025-7-31-Hellen发\N13161-122(一对一)\N13161-122(一对一)\DSC01468.JPG"
    # 获取绝对路径
    image_path = os.path.abspath(image_path)
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)
    
    try:
        # 预处理图像
        gray, binary, original_img = preprocess_image_for_text(image_path)
        
        # 提取最大的外接矩形 - 调整参数提高容差
        # 降低min_area阈值，提高max_area阈值，使用cv2.RETR_LIST获取更多轮廓
        result_image, largest_rect, largest_contour = extract_largest_rotated_rectangle(
            binary, 
            original_img, 
            min_area=30,     # 进一步降低最小面积阈值，提高对小区域的敏感度
            max_area=500000, # 适当降低最大面积阈值，避免过大区域干扰
            contour_mode=cv2.RETR_EXTERNAL  # 改为RETR_EXTERNAL以获取更精确的外部轮廓
        )
        
        # 输出结果信息
        if largest_rect is not None:
            area = cv2.contourArea(largest_contour)
            print(f"成功提取到最大的外接矩形，面积: {int(area)}")
            print(f"矩形中心点: ({int(largest_rect[0][0])}, {int(largest_rect[0][1])})")
            print(f"矩形尺寸: ({int(largest_rect[1][0])}x{int(largest_rect[1][1])})")
            print(f"矩形旋转角度: {largest_rect[2]}度")
        else:
            print("未找到符合条件的文字区域")
            
            # 如果未找到，尝试使用不同的预处理参数
            print("尝试使用替代参数重新预处理图像...")
            # 尝试不同的二值化方法
            _, alternative_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 再次尝试提取
            result_image, largest_rect, largest_contour = extract_largest_rotated_rectangle(
                alternative_binary, 
                original_img, 
                min_area=50, 
                max_area=1000000
            )
            
            if largest_rect is not None:
                area = cv2.contourArea(largest_contour)
                print(f"使用替代参数成功提取到最大的外接矩形，面积: {int(area)}")
        
        # 显示结果
        plt.figure(figsize=(12, 8))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(binary, cmap='gray')
        plt.title('二值化图像')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('提取的最大外接矩形')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 同时显示OpenCV窗口
        cv2.imshow('原始图像', original_img)
        cv2.imshow('二值化图像', binary)
        cv2.imshow('提取的最大外接矩形', result_image)
        
        # 等待按键，然后关闭所有窗口
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        sys.exit(1)
