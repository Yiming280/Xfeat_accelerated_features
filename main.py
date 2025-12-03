import cv2
import torch
import numpy as np
from modules.xfeat import XFeat
import os
import matplotlib.pyplot as plt
import time


def draw_matches(img1, img2, mkpts_0, mkpts_1, H=None):
    """
    可视化匹配结果 & warp后图框。
    img1, img2: RGB np.ndarray
    mkpts_0, mkpts_1: (N, 2)
    H: 单应性（可选）
    """
    # 拼接图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # 绘制匹配连线
    for (x1, y1), (x2, y2) in zip(mkpts_0, mkpts_1):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # 点1
        cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
        # 点2（图2在右侧，需要偏移 w1）
        cv2.circle(canvas, (int(x2) + w1, int(y2)), 3, color, -1)

        # 连线
        cv2.line(canvas, (int(x1), int(y1)),
                 (int(x2) + w1, int(y2)), color, 2)

    # 如果有单应性，绘制 warp 后的框
    if H is not None:
        box = np.array([[0, 0],
                        [w1, 0],
                        [w1, h1],
                        [0, h1]], dtype=np.float32).reshape(-1, 1, 2)
        warped_box = cv2.perspectiveTransform(box, H).reshape(-1, 2)

        # 在右图上画框（要加 w1 偏移）
        for i in range(4):
            pt1 = (int(warped_box[i][0] + w1), int(warped_box[i][1]))
            pt2 = (int(warped_box[(i + 1) % 4][0] + w1), int(warped_box[(i + 1) % 4][1]))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

    return canvas

def compute_mse(img1, img2_warp):
    """计算两张图像的MSE"""
    if img1.shape != img2_warp.shape:
        raise ValueError("Images must have the same shape for MSE")
    diff = img1.astype(np.float32) - img2_warp.astype(np.float32)
    mse = np.mean(diff**2)
    return mse

def draw_diff(img1, img2_warp, mkpts_0, mask):
    """绘制两张图像的差异图"""
    diff = np.abs(img1.astype(np.float32) - img2_warp.astype(np.float32))
    diff_norm = diff / diff.max()

    # 创建3通道差异图用于标注
    diff_vis = (diff_norm*255).astype(np.uint8)
    if diff_vis.shape[2] != 3:
        diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2RGB)

    for (x1, y1), m in zip(mkpts_0.astype(int), mask):
            if m == 1:
                cv2.circle(diff_vis, (x1, y1), 8, (0,255,0), -1)  # inlier绿
            else:
                cv2.circle(diff_vis, (x1, y1), 8, (255,0,0), -1)  # outlier红
    
    combined = np.hstack([img1, img2_warp, diff_vis])

    # 显示
    plt.figure(figsize=(20,8))
    plt.imshow(combined)
    plt.axis('off')
    plt.title(f"Left: img1 {img1.shape[1]}x{img1.shape[0]} | Middle: img2 warped | Right: difference (outliers red)")
    plt.show()

def crop_roi_rect(img, roi):
    """
    img: numpy array (H,W,3) or (H,W)
    roi: (x, y, w, h)
    """
    x, y, w, h = roi
    return img[y:y+h, x:x+w]

def main():
    # 1. 初始化 XFeat   
    os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU
    xfeat = XFeat()

    img1 = cv2.imread("assets/26-down1-481.bmp")
    img2 = cv2.imread("assets/26-down2-482.bmp")
    # print("图像尺寸：", img1.shape, img2.shape)
    # mkpts_0, mkpts_1 = xfeat.match_xfeat(img1, img2)
    # print("匹配到的点数量：", mkpts_0.shape[0])
    # # print("图1的匹配点：", mkpts_0[:5])   # 前 5 个点
    # # print("图2的匹配点：", mkpts_1[:5])
    
    roi = (1300, 1800, 1400, 600) # (x, y, w, h)
    img1_roi = crop_roi_rect(img1, roi)
    img2_roi = crop_roi_rect(img2, roi)
    # mkpts_0, mkpts_1 = xfeat.match_xfeat(img1_roi, img2_roi)
    # H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, 4.0)
    # vis = draw_matches(img1_roi, img2_roi, mkpts_0[:30], mkpts_1[:30], H)
    # plt.figure(figsize=(15,8))
    # plt.imshow(vis)
    # plt.axis('off')
    # plt.title(f'Matches & Warp at Last Size: {(img1_roi.shape[1], img1_roi.shape[0])}')
    # plt.show()
    h1, w1 = img1_roi.shape[:2]
    h2, w2 = img2_roi.shape[:2]

    scales = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]
    times = []
    sizes = []
    mse_list = []
    inlier_rates = []

    mse_threshold = 50000.0  # 根据图像亮度和范围可调整
    inlier_threshold = 0.2
    min_correct_size = None
    min_correct_results = None

    for scale in scales:
        # Resize images
        new_w1, new_h1 = int(w1 * scale), int(h1 * scale)
        new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
        if new_w1 < 200:
            print("图像过小，停止测试")
            break  # 图像太小，停止测试

        img1_resized = cv2.resize(img1_roi, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2_roi, (new_w2, new_h2), interpolation=cv2.INTER_AREA)

        # Record size
        sizes.append(new_w1 * new_h1)

        # Measure time
        start_time = time.perf_counter()
        mkpts_0, mkpts_1 = xfeat.match_xfeat(img1_resized, img2_resized)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        times.append(elapsed)

        # 匹配正确率
        if len(mkpts_0) >= 4:  # RANSAC至少需要4点
            H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, 4.0)
            H_time = time.perf_counter() - start_time
            inlier_rate = mask.sum() / len(mask)
            # Warp img2到img1尺寸
            img2_warp = cv2.warpPerspective(img2_resized, np.linalg.inv(H), (new_w1, new_h1))
            mse = compute_mse(img1_resized, img2_warp)
            draw_diff(img1_resized, img2_warp, mkpts_0, mask)
        else:
            H = None
            mask = None
            inlier_rate = 0.0
            mse = np.inf

        mse_list.append(mse)
        inlier_rates.append(inlier_rate)
        print(f"Scale: {scale:.3f}, Size: {img1_resized.shape[1]}x{img1_resized.shape[0]}, Time: {elapsed:.3f}s, Matches: {len(mkpts_0)}, MSE: {mse:.2f}, Inliers: {inlier_rate:.2f}, H_time: {H_time:.3f}s")

        if mse <= mse_threshold and inlier_rate >= inlier_threshold:
            min_correct_size = (img1_resized.shape[1], img1_resized.shape[0])
            min_correct_results = (img1_resized, img2_resized, mkpts_0, mkpts_1, H)
        else:            
            print("匹配未达标，测试更小尺寸")
            # break

    # 3. Plot computation time vs image size
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(sizes, times, marker='o')
    if min_correct_size:
        plt.axvline(min_correct_size[0]*min_correct_size[1], color='r', linestyle='--', label='Min correct size')
    plt.xlabel("Image Size (pixels)")
    plt.ylabel("Computation Time (s)")
    plt.title("Computation Time vs Image Size")
    plt.grid(True)
    plt.legend()

    plt.subplot(1,2,2)
    color_mse = 'tab:red'
    plt.plot(sizes, mse_list, marker='o', color=color_mse, label='MSE')
    plt.xlabel("Image Size (pixels)")
    plt.ylabel("MSE", color=color_mse)
    plt.tick_params(axis='y', labelcolor=color_mse)
    plt.title("MSE & Inlier Rate vs Image Size")
    plt.grid(True)

    # 双y轴显示 inlier rate
    ax2 = plt.gca().twinx()
    color_inlier = 'tab:blue'
    ax2.plot(sizes, inlier_rates, marker='x', color=color_inlier, label='Inlier Rate')
    ax2.set_ylabel("Inlier Rate", color=color_inlier)
    ax2.tick_params(axis='y', labelcolor=color_inlier)
    ax2.axhline(inlier_threshold, color='g', linestyle='-.', label='Inlier Threshold')
    if min_correct_size:
        plt.axvline(min_correct_size[0]*min_correct_size[1], color='k', linestyle='--', label='Min correct size')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # 可视化最小可配准尺寸匹配
    
    vis = draw_matches(img1_resized, img2_resized, mkpts_0[:30], mkpts_1[:30], H)
    plt.figure(figsize=(15,8))
    plt.imshow(vis)
    plt.axis('off')
    plt.title(f'Matches & Warp at Last Size: {(new_w1, new_h1)}')
    plt.show()


if __name__ == "__main__":
    main()
