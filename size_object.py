"""
Filename: init.py
Usage: This script will measure different objects in the frame using a reference object of known dimension. 
The object with known dimension must be the leftmost object.
"""
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import os
from datetime import datetime

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def save_image(image, save_path):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        is_success, buffer = cv2.imencode(".jpg", image)
        if is_success:
            with open(save_path, "wb") as f:
                f.write(buffer.tobytes())
        return is_success
    except Exception as e:
        print(f"保存图像时出错: {save_path}")
        print(f"错误信息: {str(e)}")
        return False

# 图片读取和保存相关路径
image_directory = r"C:\Users\LHB\Pictures\OCR_Captures"
processed_dir = r"C:\Users\LHB\Pictures\Processed_Images"

# 生成结果文件名（使用时间戳）
results_directory = r"C:\Users\LHB\Pictures\OCR_Results"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(results_directory, f"detection_results_{timestamp}.txt")

# 遍历图片目录，处理每张图片
with open(result_file, 'w', encoding='utf-8') as f:
    for filename in os.listdir(image_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_directory, filename)
            # 用imdecode方式读取图片，支持中文路径
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"读取图像时出错: {image_path}")
                print(f"错误信息: {str(e)}")
                continue
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 下面为原有处理流程
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)  # 高斯模糊核大小参数 (9, 9)
            edged = cv2.Canny(blur, 50, 100)          # Canny边缘检测阈值参数 50, 100
            edged = cv2.dilate(edged, None, iterations=1)  # 膨胀迭代次数参数 1
            edged = cv2.erode(edged, None, iterations=1)   # 腐蚀迭代次数参数 1
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            (cnts, _) = contours.sort_contours(cnts)
            cnts = [x for x in cnts if cv2.contourArea(x) > 100]  # 轮廓面积阈值参数 100
            if len(cnts) == 0:
                print(f"未找到有效轮廓: {image_path}")
                continue

            # 参考物体
            ref_object = cnts[0]
            box = cv2.minAreaRect(ref_object)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            dist_in_pixel = euclidean(tl, tr)
            dist_in_cm = 2
            pixel_per_cm = dist_in_pixel/dist_in_cm

            # 置信度计算函数
            def calculate_confidence(contour, image_size):
                contour_area = cv2.contourArea(contour)
                image_area = image_size[0] * image_size[1]
                area_ratio = contour_area / image_area
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                complexity = len(approx)
                confidence = 0.0
                if 4 <= complexity <= 8:
                    confidence = 0.6
                    if 0.1 <= area_ratio <= 0.9:
                        confidence += 0.2
                    if len(approx) == 4:
                        confidence += 0.2
                return min(confidence, 1.0)

            # 绘制并标注
            for idx, cnt in enumerate(cnts):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box
                cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
                mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
                mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
                wid = euclidean(tl, tr)/pixel_per_cm
                ht = euclidean(tr, br)/pixel_per_cm

                # 放大20.6倍，单位改为mm
                wid_mm = wid * 20.6
                ht_mm = ht * 20.6

                # 计算偏转角度（以minAreaRect的angle为准，修正OpenCV角度定义）
                angle = rect[2]
                # OpenCV的angle定义：宽<高时，angle为与X轴夹角，宽>高时，angle+90
                if wid_mm < ht_mm:
                    rotation_angle = angle
                else:
                    rotation_angle = angle + 90
                # 以Y轴为基准，偏转角度为90-rotation_angle
                deviation_angle = 90 - rotation_angle

                # 计算置信度
                confidence = calculate_confidence(cnt, image.shape[:2])

                cv2.putText(image, "{:.1f}mm".format(wid_mm), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(image, "{:.1f}mm".format(ht_mm), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                # 标注偏转角度
                cv2.putText(image, "Angle:{:.1f}".format(deviation_angle), (int(tl[0]), int(tl[1]) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # 只在第一个轮廓左上角输出置信度
                if idx == 0:
                    cv2.putText(image, "Conf: {:.1%}".format(confidence), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    main_deviation_angle = deviation_angle  # 记录主对象偏转角度

            # 保存处理后图片
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processed_path = os.path.join(processed_dir, f"{base_name}_processed.jpg")
            save_image(image, processed_path)

            # 写入结果到txt文件（这里只写入文件名和处理图片路径，可根据需要扩展）
            f.write(f"图像: {filename}\n")
            f.write(f"处理后图片: {processed_path}\n")
            # 可选：写入主对象的尺寸（以第一个cnt为例）
            f.write(f"主对象宽度: {wid_mm:.1f} mm\n")
            f.write(f"主对象高度: {ht_mm:.1f} mm\n")
            f.write(f"主对象置信度: {confidence:.1%}\n")
            f.write(f"主对象偏转角度: {main_deviation_angle:.1f} 度\n")
            f.write("-" * 30 + "\n")
            print(f"图像: {filename} 已处理并保存到 {processed_path}")

print(f"结果已保存到: {result_file}")