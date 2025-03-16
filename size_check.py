import cv2
import numpy as np
import imutils
import os
import time
import math
from PIL import Image, ImageDraw, ImageFont

# 添加绘制中文文本的函数
def cv2_put_chinese_text(img, text, position, font_path, font_size, color):
    """
    在OpenCV图像上绘制中文
    :param img: OpenCV图像
    :param text: 要绘制的文本
    :param position: 文本位置 (x, y)
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param color: 字体颜色 (B, G, R)
    :return: 添加了文本的图像
    """
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img_pil)
    
    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # 如果找不到指定字体，使用默认字体
        font = ImageFont.load_default()
        print(f"警告: 找不到字体 {font_path}，使用默认字体")
    
    # 绘制文本
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    
    # 将PIL图像转换回OpenCV图像
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_cv

# 获取文本大小的函数
def get_text_size(text, font_path, font_size):
    """
    获取文本大小
    :param text: 要测量的文本
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :return: (width, height) 文本宽度和高度
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # 如果找不到指定字体，使用默认字体
        font = ImageFont.load_default()
        print(f"警告: 找不到字体 {font_path}，使用默认字体")
    
    # 创建一个临时图像来测量文本大小
    img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(img)
    
    # 获取文本大小
    text_size = draw.textbbox((0, 0), text, font=font)
    
    # 返回宽度和高度
    return (text_size[2] - text_size[0], text_size[3] - text_size[1])

def measure_finger_size(image):
    try:
        # 移除读取图像和路径检查的代码
        # if not os.path.exists(image_path):
        #     print(f"错误：找不到图片文件 {image_path}")
        #     return None
        # image = cv2.imread(image_path)
        # if image is None:
        #     print(f"错误：无法读取图片 {image_path}")
        #     return None

        # 创建输出目录（如果不存在）
        output_dir = os.path.abspath(os.path.dirname(__file__))
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # 设置字体路径
        # 尝试几种常见的中文字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
            "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
            "C:/Windows/Fonts/msyh.ttc",    # Windows 微软雅黑
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"  # Linux
        ]
        
        # 查找可用的字体
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                print(f"使用字体: {font_path}")
                break
        
        if font_path is None:
            print("警告: 未找到中文字体，将使用默认字体")
            # 尝试使用默认字体
            font_path = ""
        
        # 保存原始图像用于调试
        orig = image.copy()
        result_image = orig.copy()

        # 调整图像大小，保持纵横比
        image = imutils.resize(image, width=600)
        ratio = orig.shape[1] / float(image.shape[1])
        
        # ===== 改进的卡片识别部分 =====
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(os.path.join(debug_dir, "01_gray.jpg"), gray)
        
        # 使用更强的模糊处理来消除卡片上的细节
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        # cv2.imwrite(os.path.join(debug_dir, "02_blurred.jpg"), blurred)
        
        # 使用Otsu阈值处理，更好地分离前景和背景
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # cv2.imwrite(os.path.join(debug_dir, "03_thresh.jpg"), thresh)
        
        # 使用更大的核进行形态学操作，填充卡片内部的所有细节
        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        # cv2.imwrite(os.path.join(debug_dir, "04_closed.jpg"), closed)
        
        # 寻找轮廓 - 只查找外部轮廓
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建轮廓调试图像
        contour_debug = image.copy()
        cv2.drawContours(contour_debug, contours, -1, (0, 255, 0), 1)
        # cv2.imwrite(os.path.join(debug_dir, "05_all_contours.jpg"), contour_debug)
        
        # 按轮廓面积排序（从大到小）
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 打印前几个轮廓的面积
        print("轮廓面积:")
        for i, contour in enumerate(contours[:3]):
            area = cv2.contourArea(contour)
            print(f"  轮廓 {i+1}: {area}")
        
        # 初始化卡片轮廓变量
        card_contour = None
        card_contour_small = None
        
        # 只考虑最大的轮廓作为卡片
        if len(contours) > 0:
            largest_contour = contours[0]
            area = cv2.contourArea(largest_contour)
            
            # 确保面积足够大
            if area > 10000:
                # 计算凸包以平滑轮廓
                hull = cv2.convexHull(largest_contour)
                
                # 近似轮廓，尝试得到四边形
                perimeter = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
                
                # 如果近似后的点数是四个，直接使用
                if len(approx) == 4:
                    card_contour_small = approx.copy()
                    scaled_contour = approx * ratio
                    scaled_contour = scaled_contour.astype(np.int32)
                    card_contour = scaled_contour
                    
                    cv2.drawContours(result_image, [card_contour], -1, (0, 255, 0), 2)
                    print("已识别到卡片轮廓（四边形）")
                else:
                    # 使用最小外接矩形
                    rect = cv2.minAreaRect(hull)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    card_contour_small = box.copy()
                    scaled_contour = box * ratio
                    scaled_contour = scaled_contour.astype(np.int32)
                    card_contour = scaled_contour
                    
                    cv2.drawContours(result_image, [card_contour], -1, (0, 255, 255), 2)
                    print("使用最小外接矩形作为卡片轮廓")
        
        # ===== 手指识别部分 =====
        # 使用肤色检测方法
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义HSV空间中的肤色范围
        lower_skin = np.array([0, 50, 100], dtype=np.uint8)
        upper_skin = np.array([20, 170, 255], dtype=np.uint8)
        
        # 创建肤色掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        # cv2.imwrite(os.path.join(debug_dir, "06_skin_mask.jpg"), skin_mask)
        
        # 执行形态学操作以改善掩码
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite(os.path.join(debug_dir, "07_skin_mask_processed.jpg"), skin_mask)
        
        # 在掩码上寻找轮廓
        finger_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 按面积排序
        finger_contours = sorted(finger_contours, key=cv2.contourArea, reverse=True)
        
        # 初始化手指轮廓变量
        finger_contour = None
        
        # 选择最大的肤色轮廓作为手指
        if len(finger_contours) > 0:
            largest_skin = finger_contours[0]
            area = cv2.contourArea(largest_skin)
            
            if area > 500:
                # 使用凸包平滑轮廓
                hull = cv2.convexHull(largest_skin)
                
                # 调整轮廓点坐标以匹配原始图像尺寸
                scaled_contour = hull * ratio
                scaled_contour = scaled_contour.astype(np.int32)
                finger_contour = scaled_contour
                
                # 在结果图像上绘制手指轮廓（红色）
                cv2.drawContours(result_image, [finger_contour], -1, (0, 0, 255), 2)
                print(f"已识别到手指轮廓，面积: {area}")
                
                # ===== 改进的手指宽度测量方法 =====
                
                # 1. 计算手指轮廓的最小外接矩形
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # 获取矩形的中心、宽度、高度和旋转角度
                center, (width, height), angle = rect
                
                # 确保角度在0-90度之间
                if angle < -45:
                    angle += 90
                    width, height = height, width
                
                # 2. 创建手指掩码
                finger_mask = np.zeros_like(gray)
                cv2.drawContours(finger_mask, [hull], -1, 255, -1)
                # cv2.imwrite(os.path.join(debug_dir, "08_finger_mask.jpg"), finger_mask)
                
                # 3. 计算骨架（中轴线）
                # 使用距离变换和阈值处理找到近似骨架
                dist_transform = cv2.distanceTransform(finger_mask, cv2.DIST_L2, 5)
                _, skeleton = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
                skeleton = skeleton.astype(np.uint8)
                # cv2.imwrite(os.path.join(debug_dir, "09_skeleton.jpg"), skeleton)
                
                # 4. 找到骨架的端点
                # 使用霍夫线变换找到主方向
                lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
                
                # 创建骨架可视化图像
                skeleton_vis = cv2.cvtColor(finger_mask.copy(), cv2.COLOR_GRAY2BGR)
                
                # 如果找到线段
                main_line = None
                max_length = 0
                
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        # 找到最长的线段作为主方向
                        if length > max_length:
                            max_length = length
                            main_line = line[0]
                    
                    # 如果找到主线
                    if main_line is not None:
                        x1, y1, x2, y2 = main_line
                        # 绘制主线
                        cv2.line(skeleton_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 计算主线的角度
                        main_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        
                        # 5. 计算垂直于主线的宽度
                        # 创建旋转矩阵
                        M = cv2.getRotationMatrix2D((center[0], center[1]), main_angle, 1)
                        
                        # 旋转手指掩码
                        rotated_mask = cv2.warpAffine(finger_mask, M, (finger_mask.shape[1], finger_mask.shape[0]))
                        # cv2.imwrite(os.path.join(debug_dir, "10_rotated_mask.jpg"), rotated_mask)
                        
                        # 计算每一列的非零像素数量（即每个位置的宽度）
                        col_sums = np.sum(rotated_mask > 0, axis=0)
                        row_sums = np.sum(rotated_mask > 0, axis=1)
                        
                        # 找到非零区域
                        col_indices = np.where(col_sums > 0)[0]
                        row_indices = np.where(row_sums > 0)[0]
                        
                        if len(col_indices) > 0 and len(row_indices) > 0:
                            # 计算手指的宽度和高度
                            rotated_width = len(col_indices)
                            rotated_height = len(row_indices)
                            
                            # 取较小的值作为手指宽度
                            finger_width_pixels = min(rotated_width, rotated_height)

                            # 按比例进行一定补偿
                            finger_width_pixels = int(finger_width_pixels * 0.85)
                            
                            # 创建宽度可视化
                            width_vis = rotated_mask.copy()
                            mid_row = (row_indices[0] + row_indices[-1]) // 2
                            cv2.line(width_vis, (col_indices[0], mid_row), (col_indices[-1], mid_row), 128, 2)
                            # cv2.imwrite(os.path.join(debug_dir, "11_width_visualization.jpg"), width_vis)
                        else:
                            # 如果旋转后无法计算宽度，回退到最小外接矩形的方法
                            finger_width_pixels = min(width, height)
                else:
                    # 如果没有找到线段，回退到最小外接矩形的方法
                    finger_width_pixels = min(width, height)
                    
                    # 在调试图像上绘制最小外接矩形
                    cv2.drawContours(skeleton_vis, [box], 0, (0, 0, 255), 2)
                
                # cv2.imwrite(os.path.join(debug_dir, "12_skeleton_vis.jpg"), skeleton_vis)
                
                # 如果卡片被识别，计算实际宽度
                if card_contour is not None and card_contour_small is not None:
                    try:
                        # 信用卡标准尺寸为85.60 × 53.98毫米
                        card_rect = cv2.minAreaRect(card_contour_small)
                        card_width = max(card_rect[1][0], card_rect[1][1])
                        card_height = min(card_rect[1][0], card_rect[1][1])
                        
                        # 假设卡片的长边是85.60毫米，2毫米补偿
                        mm_per_pixel = (85.60-2) / card_width
                        finger_width_mm = finger_width_pixels * mm_per_pixel
                        
                        print(f"手指宽度: {finger_width_mm:.2f} 毫米")
                        
                        # ===== 计算手指周长 =====
                        # 假设手指横截面近似为椭圆
                        # 估计手指厚度（假设宽高比为1.2:1）
                        # finger_height_mm = finger_width_mm / 1.2
                        finger_height_mm = finger_width_mm / 1
                        
                        # 使用椭圆周长公式计算手指周长
                        # 椭圆周长近似公式：π * (a + b) * (1 + (3λ²)/(10 + √(4 - 3λ²)))
                        # 其中 λ = (a-b)/(a+b), a和b是半长轴和半短轴
                        a = finger_width_mm / 2
                        b = finger_height_mm / 2
                        h = ((a - b) / (a + b)) ** 2
                        finger_circumference_mm = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
                        
                        print(f"手指周长: {finger_circumference_mm:.2f} 毫米")
                        
                        # 计算戒指尺寸（中国标准）
                        # 中国戒指尺寸 = 周长(mm) / π
                        ring_size_cn = finger_circumference_mm / np.pi
                        print(f"戒指尺寸(中国标准): {ring_size_cn:.1f}")
                        
                        # 计算戒指尺寸（美国标准）
                        # 美国戒指尺寸 = 周长(mm) / 2.55 - 36.5
                        ring_size_us = finger_circumference_mm / 2.55 - 36.5
                        print(f"戒指尺寸(美国标准): {ring_size_us:.1f}")
                        
                        # 计算戒指尺寸（欧洲标准）
                        # 欧洲戒指尺寸 = 周长(mm) / 3.14 * 2
                        ring_size_eu = finger_circumference_mm / 3.14 * 2
                        print(f"戒指尺寸(欧洲标准): {ring_size_eu:.1f}")
                        
                    except Exception as e:
                        print(f"计算实际宽度时出错: {str(e)}")
                        finger_width_mm = None
                        finger_circumference_mm = None
                        ring_size_cn = None
                        ring_size_us = None
                        ring_size_eu = None
                else:
                    finger_width_mm = None
                    finger_circumference_mm = None
                    ring_size_cn = None
                    ring_size_us = None
                    ring_size_eu = None
                
                # ===== 在图像上标注手指宽度和周长 =====
                # 在原始图像上绘制测量线
                if main_line is not None:
                    # 计算垂直于主线的方向
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx*dx + dy*dy)
                    
                    # 单位向量
                    udx = dx / length
                    udy = dy / length
                    
                    # 垂直向量（顺时针旋转90度）
                    vdx = -udy
                    vdy = udx
                    
                    # 计算中点
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    # 计算垂直线的方向向量
                    vdx_norm = vdx / np.sqrt(vdx*vdx + vdy*vdy)  # 归一化
                    vdy_norm = vdy / np.sqrt(vdx*vdx + vdy*vdy)
                    
                    # 找到与手指轮廓的交点
                    # 从中心点向两个方向射线，找到与轮廓的交点
                    step_size = 0.5  # 步长
                    max_steps = int(finger_width_pixels * 1.5)  # 最大步数
                    
                    # 向一个方向寻找交点
                    p1x, p1y = mid_x, mid_y
                    for i in range(1, max_steps):
                        test_x = int(mid_x + vdx_norm * i * step_size)
                        test_y = int(mid_y + vdy_norm * i * step_size)
                        
                        # 确保点在图像范围内
                        if (test_x < 0 or test_x >= finger_mask.shape[1] or 
                            test_y < 0 or test_y >= finger_mask.shape[0]):
                            break
                        
                        # 如果点在手指轮廓外，停止
                        if finger_mask[test_y, test_x] == 0:
                            break
                        
                        p1x, p1y = test_x, test_y
                    
                    # 向另一个方向寻找交点
                    p2x, p2y = mid_x, mid_y
                    for i in range(1, max_steps):
                        test_x = int(mid_x - vdx_norm * i * step_size)
                        test_y = int(mid_y - vdy_norm * i * step_size)
                        
                        # 确保点在图像范围内
                        if (test_x < 0 or test_x >= finger_mask.shape[1] or 
                            test_y < 0 or test_y >= finger_mask.shape[0]):
                            break
                        
                        # 如果点在手指轮廓外，停止
                        if finger_mask[test_y, test_x] == 0:
                            break
                        
                        p2x, p2y = test_x, test_y
                    
                    # 计算实际宽度（像素）
                    actual_width_pixels = np.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)
                    
                    # 更新手指宽度
                    finger_width_pixels = actual_width_pixels
                    
                    # 调整坐标以匹配原始图像尺寸
                    p1 = (int(p1x * ratio), int(p1y * ratio))
                    p2 = (int(p2x * ratio), int(p2y * ratio))
                    
                    # 绘制宽度线（紫色）
                    cv2.line(result_image, p1, p2, (255, 0, 255), 2)
                    
                    # 计算线的中点
                    line_center_x = int((p1[0] + p2[0]) / 2)
                    line_center_y = int((p1[1] + p2[1]) / 2)
                else:
                    # 如果没有找到主线，使用最小外接矩形的短边
                    scaled_box = box * ratio
                    scaled_box = scaled_box.astype(np.int32)
                    
                    # 在原始图像上绘制最小外接矩形
                    cv2.drawContours(result_image, [scaled_box], 0, (255, 255, 0), 2)
                    
                    # 找到矩形的短边
                    edges = []
                    for i in range(4):
                        edge_length = np.linalg.norm(scaled_box[i] - scaled_box[(i+1)%4])
                        edges.append((i, edge_length))
                    
                    edges.sort(key=lambda x: x[1])
                    shortest_edge_idx = edges[0][0]
                    
                    # 获取短边的两个端点
                    p1 = scaled_box[shortest_edge_idx]
                    p2 = scaled_box[(shortest_edge_idx+1)%4]
                    
                    # 绘制宽度线
                    cv2.line(result_image, tuple(p1), tuple(p2), (255, 0, 255), 2)
                    
                    # 计算线的中点
                    line_center_x = int((p1[0] + p2[0]) / 2)
                    line_center_y = int((p1[1] + p2[1]) / 2)
                
                # 设置字体大小
                font_size = 24
                
                # 在图像上标注宽度值
                if finger_width_mm is not None:
                    width_label = f"宽度: {finger_width_mm:.2f} mm"
                else:
                    width_label = f"宽度: {finger_width_pixels:.2f} px"
                
                # 计算文本大小
                width_text_size = get_text_size(width_label, font_path, font_size)
                
                # 确定宽度文本位置
                width_text_x = line_center_x - width_text_size[0] // 2
                width_text_y = line_center_y - 10
                
                # 确保文本在图像内
                width_text_x = max(width_text_x, 10)
                width_text_x = min(width_text_x, result_image.shape[1] - width_text_size[0] - 10)
                width_text_y = max(width_text_y, width_text_size[1] + 10)
                width_text_y = min(width_text_y, result_image.shape[0] - 10)
                
                # 绘制宽度文本背景
                cv2.rectangle(result_image, 
                             (width_text_x - 5, width_text_y - width_text_size[1] - 5), 
                             (width_text_x + width_text_size[0] + 5, width_text_y + 5), 
                             (255, 255, 255), -1)
                
                # 在图像上添加宽度标注
                result_image = cv2_put_chinese_text(result_image, width_label, 
                                                  (width_text_x, width_text_y - width_text_size[1]), 
                                                  font_path, font_size, (0, 0, 0))
                
                # 在图像上标注周长值（如果有）
                if finger_circumference_mm is not None:
                    # 创建周长标签
                    circumference_label = f"周长: {finger_circumference_mm:.2f} mm"
                    
                    # 计算文本大小
                    circ_text_size = get_text_size(circumference_label, font_path, font_size)
                    
                    # 确定周长文本位置（在宽度标签下方）
                    circ_text_x = width_text_x
                    circ_text_y = width_text_y + 40
                    
                    # 确保文本在图像内
                    circ_text_y = min(circ_text_y, result_image.shape[0] - 10)
                    
                    # 绘制周长文本背景
                    cv2.rectangle(result_image, 
                                 (circ_text_x - 5, circ_text_y - circ_text_size[1] - 5), 
                                 (circ_text_x + circ_text_size[0] + 5, circ_text_y + 5), 
                                 (255, 255, 255), -1)
                    
                    # 在图像上添加周长标注
                    result_image = cv2_put_chinese_text(result_image, circumference_label, 
                                                      (circ_text_x, circ_text_y - circ_text_size[1]), 
                                                      font_path, font_size, (0, 0, 0))
        
        # 修改返回值，同时返回图像和测量结果
        return result_image, finger_width_mm, finger_circumference_mm, ring_size_cn
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# 调用示例
if __name__ == "__main__":
    image_path = 'demoPicture.jpg'
    measure_finger_size(image_path)