"""
自动观看广告脚本。
循环：
  截图 -> 模板匹配出"观看广告"按钮 -> 逐个点击
       -> 检测广告是否真的弹出（界面差异 + activity 变化）
          - 弹了：等 10-15s -> 关闭广告 -> 回到游戏 -> 继续下一个
          - 没弹：判定为该按钮已 0/5，跳过
  当前屏处理完后向上滑动；连续 N 屏没有任何成功点击 -> 停止。
Ctrl+C 安全退出。
"""
import time
import random
import signal
import sys
import re
from pathlib import Path

import cv2
import numpy as np
import uiautomator2 as u2

ROOT = Path(__file__).parent
TEMPLATE = ROOT / "template.png"           # 看广告按钮模板
CLOSE_TEMPLATE = ROOT / "close_template.png"  # 关闭按钮模板（从 close.jpg 裁出来的）
CONFIRM_TEMPLATE = ROOT / "confirm_template.png"  # "恭喜获得"弹窗"确认"按钮（从 get.jpg 裁的）
TEMPLATE_SCALE = 1.5            # 看广告按钮模板相对实机截图的缩放
CLOSE_TEMPLATE_SCALES = [1.0, 1.125, 1.25, 0.9]  # 多尺度匹配关闭按钮（close.jpg 1280 vs 设备 1440）
CONFIRM_TEMPLATE_SCALES = [1.125, 1.0, 1.25, 0.9]
MATCH_THRESHOLD = 0.80          # 看广告按钮匹配阈值
ORANGE_RATIO_THRESHOLD = 0.30   # 匹配区域内"亮橙色"像素比例 < 此值 -> 视为 0/5 灰按钮，跳过
CLOSE_MATCH_THRESHOLD = 0.80    # 关闭按钮阈值（提高，防止广告内 UI 误匹配）
CONFIRM_MATCH_THRESHOLD = 0.75  # 确认按钮阈值
NMS_IOU = 0.3

# 关闭按钮的搜索 ROI（按比例，限制在顶部右侧，避免误命中广告内容区）
CLOSE_ROI_NORM = (0.45, 0.0, 1.0, 0.15)   # (x1, y1, x2, y2) 归一化
# 确认按钮在屏幕中下方
CONFIRM_ROI_NORM = (0.10, 0.40, 0.90, 0.85)

AD_MIN_WAIT = 8                 # 广告至少播 8 秒才开始找关闭按钮（避免误点广告里的元素）
AD_MAX_WAIT = 35                # 最长等 35 秒还没出现关闭模板 -> 走兜底
CLOSE_POLL_INTERVAL = 0.8       # 轮询关闭按钮的间隔
CONFIRM_MAX_WAIT = 8            # 关闭广告后等"恭喜获得"弹窗最多多久
CONFIRM_POLL_INTERVAL = 0.6

AD_OPEN_DIFF_THRESHOLD = 0.20   # 截图差异 > 该比例 -> 广告打开（信号 1）
AD_OPEN_NODE_GROWTH = 1.5       # XML 节点数膨胀 >= 该倍数 -> 广告打开（信号 2）
AD_OPEN_MARKER_IDS = ["ifg", "nqn"]  # 广告页特有的 resource-id（信号 3）
AD_OPEN_POLL_TIMEOUT = 5.0      # 点完后最多等多久检测广告
AD_OPEN_POLL_INTERVAL = 0.6
POST_CLICK_WAIT = 1.0           # 点击后稍等一下再开始轮询
POST_CLOSE_WAIT = 3.0           # 关闭广告后等待回到游戏
MAX_EMPTY_SCROLLS = 4           # 连续这么多次滑动都没有可点的按钮 -> 停止（调大一点给广告自己消化的时间）
SCROLL_SETTLE = 1.5

# 危险区域黑名单（绝对坐标，[x1,y1,x2,y2]）——抖音小游戏外壳的"关闭"按钮
BANNED_REGIONS = [
    (1256, 176, 1408, 304),     # 外壳"关闭" content-desc=关闭 of MiniGameHost
    (1102, 176, 1254, 304),     # 外壳"更多"
]

# 广告关闭按钮兜底坐标（按比例，归一化到屏幕尺寸）
CLOSE_FALLBACK_NORM = (0.96, 0.06)


# ----- 模板匹配 -----
def nms(boxes, scores, iou=NMS_IOU):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][ov < iou]
    return keep


def load_template():
    tmpl = cv2.imread(str(TEMPLATE))
    if tmpl is None:
        sys.exit(f"找不到模板 {TEMPLATE}")
    h, w = tmpl.shape[:2]
    tmpl = cv2.resize(tmpl, (int(w * TEMPLATE_SCALE), int(h * TEMPLATE_SCALE)))
    return cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)


def load_close_template():
    tmpl = cv2.imread(str(CLOSE_TEMPLATE))
    if tmpl is None:
        sys.exit(f"找不到关闭模板 {CLOSE_TEMPLATE}")
    return cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)


def load_confirm_template():
    tmpl = cv2.imread(str(CONFIRM_TEMPLATE))
    if tmpl is None:
        sys.exit(f"找不到确认模板 {CONFIRM_TEMPLATE}")
    return cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)


def multi_scale_match(screen_bgr, tmpl_gray, scales, threshold, roi_norm=None):
    """返回 (x, y, w, h, score, scale) 或 None。
    roi_norm: 可选 (x1,y1,x2,y2) 归一化区域，只在该区域内匹配。"""
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    if roi_norm:
        rx1 = int(W * roi_norm[0]); ry1 = int(H * roi_norm[1])
        rx2 = int(W * roi_norm[2]); ry2 = int(H * roi_norm[3])
        roi = gray[ry1:ry2, rx1:rx2]
    else:
        rx1, ry1 = 0, 0
        roi = gray
    best = None
    for s in scales:
        h0, w0 = tmpl_gray.shape
        nw, nh = int(w0 * s), int(h0 * s)
        if nw < 20 or nh < 20 or nw > roi.shape[1] or nh > roi.shape[0]:
            continue
        t = cv2.resize(tmpl_gray, (nw, nh)) if s != 1.0 else tmpl_gray
        res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
        _, mv, _, mloc = cv2.minMaxLoc(res)
        if best is None or mv > best[0]:
            best = (mv, mloc, nw, nh, s)
    if best is None or best[0] < threshold:
        return None
    score, (x, y), w, h, s = best
    # 把 ROI 内坐标加回原图坐标
    return (x + rx1, y + ry1, w, h, score, s)


def find_buttons(screen_bgr, tmpl_gray):
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
    res = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= MATCH_THRESHOLD)
    h, w = tmpl_gray.shape
    boxes = [(int(x), int(y), w, h) for x, y in zip(xs, ys)]
    scores = [float(res[y, x]) for x, y in zip(xs, ys)]
    keep = nms(boxes, scores)
    hits = [(boxes[i], scores[i]) for i in keep]
    # 过滤黑名单
    hits = [hh for hh in hits if not in_banned(hh[0])]
    # 过滤灰按钮（颜色判断 0/5）
    filtered = []
    for (box, sc) in hits:
        x, y, bw, bh = box
        region = hsv[y:y + bh, x:x + bw]
        orange = cv2.inRange(region, (5, 80, 80), (30, 255, 255))
        ratio = orange.mean() / 255
        if ratio < ORANGE_RATIO_THRESHOLD:
            print(f"  [filter] 灰按钮 ({x+bw//2},{y+bh//2}) orange={ratio:.0%} 跳过")
            continue
        filtered.append((box, sc, ratio))
    # 按 y、x 排序，从上到下、从左到右
    filtered.sort(key=lambda h: (h[0][1], h[0][0]))
    return [(b, s) for (b, s, _r) in filtered]


def in_banned(box):
    cx = box[0] + box[2] // 2
    cy = box[1] + box[3] // 2
    for (x1, y1, x2, y2) in BANNED_REGIONS:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def screenshot_bgr(d):
    img = d.screenshot(format="opencv")
    return img


def count_nodes(xml):
    return xml.count("<node ")


def has_marker(xml, markers):
    return any((':id/' + m + '"') in xml for m in markers)


def wait_ad_open(d, before_img, before_xml):
    """轮询多个信号判断广告是否打开。返回 (opened: bool, signal: str)。"""
    before_nodes = count_nodes(before_xml)
    deadline = time.time() + AD_OPEN_POLL_TIMEOUT
    last_diff = 0.0
    last_nodes = before_nodes
    while time.time() < deadline and not STOP:
        time.sleep(AD_OPEN_POLL_INTERVAL)
        cur_img = screenshot_bgr(d)
        cur_xml = d.dump_hierarchy()
        cur_nodes = count_nodes(cur_xml)
        dr = diff_ratio(before_img, cur_img)
        last_diff, last_nodes = dr, cur_nodes
        if has_marker(cur_xml, AD_OPEN_MARKER_IDS):
            return True, f"marker-id {AD_OPEN_MARKER_IDS}"
        if before_nodes > 0 and cur_nodes / before_nodes >= AD_OPEN_NODE_GROWTH:
            return True, f"nodes {before_nodes}->{cur_nodes}"
        if dr >= AD_OPEN_DIFF_THRESHOLD:
            return True, f"diff {dr:.2%}"
    return False, f"timeout (last diff={last_diff:.2%}, nodes={before_nodes}->{last_nodes})"


def diff_ratio(a, b):
    """两截图差异度（0~1）。降采样后阈值二值化算改变像素占比。"""
    if a.shape != b.shape:
        return 1.0
    sa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (256, 512))
    sb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (256, 512))
    d = cv2.absdiff(sa, sb)
    return float((d > 25).mean())


# ----- 广告关闭 -----
def find_close_button(screen_bgr, close_tmpl_gray):
    m = multi_scale_match(screen_bgr, close_tmpl_gray, CLOSE_TEMPLATE_SCALES,
                          CLOSE_MATCH_THRESHOLD, roi_norm=CLOSE_ROI_NORM)
    if not m:
        return None
    x, y, w, h, score, scale = m
    # X 在胶囊右侧约 85%
    return (x + int(w * 0.85), y + h // 2, score, scale)


def find_confirm_button(screen_bgr, confirm_tmpl_gray):
    m = multi_scale_match(screen_bgr, confirm_tmpl_gray, CONFIRM_TEMPLATE_SCALES,
                          CONFIRM_MATCH_THRESHOLD, roi_norm=CONFIRM_ROI_NORM)
    if not m:
        return None
    x, y, w, h, score, scale = m
    return (x + w // 2, y + h // 2, score, scale)


def try_close_ad(d, close_tmpl_gray, screen_size):
    """轮询找关闭按钮模板，找到后点击并校验屏幕是否真的离开广告页。
    超时直接返回 False，不做坐标兜底（避免误点广告内容）。"""
    w, h = screen_size
    deadline = time.time() + AD_MAX_WAIT
    # 暖机
    t_warmup = time.time() + AD_MIN_WAIT
    while time.time() < t_warmup and not STOP:
        time.sleep(0.5)

    while time.time() < deadline and not STOP:
        screen = screenshot_bgr(d)
        m = find_close_button(screen, close_tmpl_gray)
        if m:
            cx, cy, score, scale = m
            print(f"  [close] 模板命中 score={score:.2f} scale={scale} click=({cx},{cy})")
            d.click(cx, cy)
            time.sleep(POST_CLOSE_WAIT)
            # 校验：点完后关闭按钮还在 -> 误命中了，回到轮询
            verify = screenshot_bgr(d)
            still_there = find_close_button(verify, close_tmpl_gray)
            big_change = diff_ratio(screen, verify) > 0.15
            if not still_there and big_change:
                return True
            print(f"  [close] 点击后屏幕变化 {diff_ratio(screen, verify):.2%}, "
                  f"关闭模板仍在={bool(still_there)} -> 视为误点，继续轮询")
        time.sleep(CLOSE_POLL_INTERVAL)

    print(f"  [close] 模板超时未命中，放弃本次关闭，回到主循环重新扫描")
    return False


def try_click_confirm(d, confirm_tmpl_gray):
    """关闭广告后会弹"恭喜获得"，找到"确认"绿按钮就点。没出现就跳过（不是每次都有）。"""
    deadline = time.time() + CONFIRM_MAX_WAIT
    while time.time() < deadline and not STOP:
        screen = screenshot_bgr(d)
        m = find_confirm_button(screen, confirm_tmpl_gray)
        if m:
            cx, cy, score, scale = m
            print(f"  [confirm] 命中 score={score:.2f} scale={scale} click=({cx},{cy})")
            d.click(cx, cy)
            time.sleep(1.2)
            return True
        time.sleep(CONFIRM_POLL_INTERVAL)
    print("  [confirm] 未出现弹窗，跳过")
    return False


# ----- 主循环 -----
STOP = False


def install_sigint():
    def handler(sig, frame):
        global STOP
        print("\n[!] 收到 Ctrl+C，本轮结束后退出")
        STOP = True
    signal.signal(signal.SIGINT, handler)


def main():
    install_sigint()
    d = u2.connect()
    w, h = d.window_size()
    print(f"[init] 设备 {w}x{h}, app={d.app_current()}")
    tmpl_gray = load_template()
    close_tmpl_gray = load_close_template()
    confirm_tmpl_gray = load_confirm_template()

    empty_scrolls = 0
    total_ads = 0
    total_skips = 0
    depleted = set()    # 记录已判定为 0/5 的按钮位置（用 grid-key 容差比较）

    def grid_key(cx, cy, grid=80):
        return (cx // grid, cy // grid)

    while not STOP:
        screen = screenshot_bgr(d)
        all_hits = find_buttons(screen, tmpl_gray)
        # 过滤掉已判定 0/5 的位置
        hits = []
        for (b, s) in all_hits:
            cx, cy = b[0] + b[2] // 2, b[1] + b[3] // 2
            if grid_key(cx, cy) in depleted:
                continue
            hits.append((b, s))
        print(f"\n[scan] 候选 {len(all_hits)} 个，过滤掉 {len(all_hits)-len(hits)} 个已耗尽，剩 {len(hits)}")

        if not hits:
            # 屏上看得到按钮但全在 depleted 里 -> 真的全用完了，立刻停
            if all_hits:
                print(f"[stop] 屏上 {len(all_hits)} 个按钮均已耗尽，结束。"
                      f"共完成广告 {total_ads}，跳过 {total_skips}")
                break
            # 屏上完全无按钮 -> 滑动找新内容
            empty_scrolls += 1
            if empty_scrolls >= MAX_EMPTY_SCROLLS:
                print(f"[stop] 连续 {MAX_EMPTY_SCROLLS} 次无按钮，结束。共完成广告 {total_ads}，跳过 {total_skips}")
                break
            print(f"[scroll] 第 {empty_scrolls}/{MAX_EMPTY_SCROLLS} 次空滑动")
            d.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.3), duration=0.5)
            time.sleep(SCROLL_SETTLE)
            continue
        empty_scrolls = 0

        # 每轮只点一个按钮（防止重复点击 + 广告异步打开导致误操作）
        box, score = hits[0]
        x, y, bw, bh = box
        cx, cy = x + bw // 2, y + bh // 2
        print(f"  [click] ({cx},{cy}) score={score:.2f}")

        before_img = screen.copy()
        before_xml = d.dump_hierarchy()
        d.click(cx, cy)
        time.sleep(POST_CLICK_WAIT)

        opened, signal = wait_ad_open(d, before_img, before_xml)
        print(f"  [detect] {'广告打开' if opened else '未弹出，标记 0/5'}  signal={signal}")

        if not opened:
            total_skips += 1
            depleted.add(grid_key(cx, cy))
            continue

        total_ads += 1
        print(f"  [ad] 等待广告完成（最少 {AD_MIN_WAIT}s，最多 {AD_MAX_WAIT}s）")
        closed = try_close_ad(d, close_tmpl_gray, (w, h))
        time.sleep(POST_CLOSE_WAIT)

        if closed:
            # 关闭广告后通常会弹"恭喜获得"——点掉它
            try_click_confirm(d, confirm_tmpl_gray)
        else:
            # 没成功关闭：直接回主循环重新 scan，不做坐标兜底
            print("  [recover] 本次未成功关闭，回到主循环重新扫描")


if __name__ == "__main__":
    main()
