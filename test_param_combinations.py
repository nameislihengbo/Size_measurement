import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective, contours
import imutils
from datetime import datetime
from tqdm import tqdm  # 新增

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

# 步进方式生成参数组合
def range_step(start, stop, step):
    return list(range(start, stop + 1, step))

# 高斯模糊核步进（保证为奇数）
blur_ksizes = [(k, k) for k in range_step(3, 15, 2)]  # 3,5,7,9,11,13,15 共7种

# Canny阈值步进
canny1_list = range_step(10, 100, 30)   # 10,40,70,100 共4种
canny2_list = range_step(80, 200, 40)   # 80,120,160,200 共4种
canny_params = [(c1, c2) for c1 in canny1_list for c2 in canny2_list if c2 > c1]  # 10种

# 膨胀/腐蚀迭代次数步进
dilate_iters = range_step(1, 3, 1)      # 1,2,3 共3种
erode_iters = range_step(1, 3, 1)       # 1,2,3 共3种

# 面积阈值步进
area_thresholds = range_step(50, 300, 50)  # 50,100,150,200,250,300 共6种

# 组合总数 = 7 * 10 * 3 * 3 * 6 = 3780 种

scale = 20.6  # 固定放大倍数

image_directory = r"C:\Users\LHB\Pictures\OCR_Captures"

# 统计所有参数组合总数和图片总数
param_combos = [
    (blur_ksize, canny1, canny2, dilate_iter, erode_iter, min_area)
    for blur_ksize in blur_ksizes
    for canny1, canny2 in canny_params
    for dilate_iter in dilate_iters
    for erode_iter in erode_iters
    for min_area in area_thresholds
]
all_filenames = [fn for fn in os.listdir(image_directory) if fn.endswith(('.png', '.jpg', '.jpeg'))]

total_combos = len(param_combos)
for combo in tqdm(param_combos, desc="参数组合进度"):
    blur_ksize, canny1, canny2, dilate_iter, erode_iter, min_area = combo
    param_tag = f"b{blur_ksize[0]}x{blur_ksize[1]}_c{canny1}-{canny2}_d{dilate_iter}_e{erode_iter}_a{min_area}"
    processed_dir = rf"C:\Users\LHB\Pictures\Processed_Images\process_{param_tag}"
    results_directory = rf"C:\Users\LHB\Pictures\OCR_Results\result_{param_tag}"
    processed_dir_false = rf"C:\Users\LHB\Pictures\Processed_Images_False\process_{param_tag}"
    results_directory_false = rf"C:\Users\LHB\Pictures\OCR_Results_False\result_{param_tag}"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)
    os.makedirs(processed_dir_false, exist_ok=True)
    os.makedirs(results_directory_false, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_directory, f"detection_results_{timestamp}.txt")
    result_file_false = os.path.join(results_directory_false, f"detection_results_{timestamp}.txt")

    valid_count = 0
    fail_count = 0

    with open(result_file, 'w', encoding='utf-8') as f, \
         open(result_file_false, 'w', encoding='utf-8') as f_false:
        for filename in tqdm(all_filenames, desc=f"{param_tag}", leave=False):
            image_path = os.path.join(image_directory, filename)
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"读取图像时出错: {image_path}")
                print(f"错误信息: {str(e)}")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                processed_path = os.path.join(processed_dir_false, f"{base_name}_processed.jpg")
                f_false.write(f"图像: {filename}\n")
                f_false.write(f"处理后图片: {processed_path}\n")
                f_false.write(f"异常: {str(e)}\n")
                f_false.write("-" * 30 + "\n")
                fail_count += 1
                continue
            if image is None:
                print(f"无法读取图像: {image_path}")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                processed_path = os.path.join(processed_dir_false, f"{base_name}_processed.jpg")
                f_false.write(f"图像: {filename}\n")
                f_false.write(f"处理后图片: {processed_path}\n")
                f_false.write(f"异常: 无法读取图像\n")
                f_false.write("-" * 30 + "\n")
                fail_count += 1
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, blur_ksize, 0)
            edged = cv2.Canny(blur, canny1, canny2)
            edged = cv2.dilate(edged, None, iterations=dilate_iter)
            edged = cv2.erode(edged, None, iterations=erode_iter)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) == 0:
                print(f"未找到有效轮廓: {image_path}")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                processed_path = os.path.join(processed_dir_false, f"{base_name}_processed.jpg")
                save_image(image, processed_path)
                f_false.write(f"图像: {filename}\n")
                f_false.write(f"处理后图片: {processed_path}\n")
                f_false.write(f"异常: 未找到有效轮廓\n")
                f_false.write("-" * 30 + "\n")
                fail_count += 1
                continue
            (cnts, _) = contours.sort_contours(cnts)
            cnts = [x for x in cnts if cv2.contourArea(x) > min_area]
            if len(cnts) == 0:
                print(f"未找到有效轮廓: {image_path}")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                processed_path = os.path.join(processed_dir_false, f"{base_name}_processed.jpg")
                save_image(image, processed_path)
                f_false.write(f"图像: {filename}\n")
                f_false.write(f"处理后图片: {processed_path}\n")
                f_false.write(f"异常: 轮廓面积过滤后无有效轮廓\n")
                f_false.write("-" * 30 + "\n")
                fail_count += 1
                continue

            ref_object = cnts[0]
            box = cv2.minAreaRect(ref_object)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            dist_in_pixel = euclidean(tl, tr)
            dist_in_cm = 2
            pixel_per_cm = dist_in_pixel/dist_in_cm

            for cnt in cnts:
                box = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box
                cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
                mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
                mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
                wid = euclidean(tl, tr)/pixel_per_cm
                ht = euclidean(tr, br)/pixel_per_cm
                wid_mm = wid * scale
                ht_mm = ht * scale
                cv2.putText(image, "{:.1f}mm".format(wid_mm), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(image, "{:.1f}mm".format(ht_mm), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processed_path = os.path.join(processed_dir, f"{base_name}_processed.jpg")
            save_image(image, processed_path)

            f.write(f"图像: {filename}\n")
            f.write(f"处理后图片: {processed_path}\n")
            f.write(f"主对象宽度: {wid_mm:.1f} mm\n")
            f.write(f"主对象高度: {ht_mm:.1f} mm\n")
            f.write("-" * 30 + "\n")
            print(f"[{param_tag}] 图像: {filename} 已处理并保存到 {processed_path}")
            valid_count += 1

    print(f"[{param_tag}] 结果已保存到: {result_file}")
    print(f"[{param_tag}] 有效识别数量: {valid_count}，失败数量: {fail_count}")

print(f"本次测试参数组合总数: {total_combos}")
