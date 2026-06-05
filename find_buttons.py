"""调试模板匹配：在 screen.png 上找出所有"观看广告"按钮位置，输出 matched.png。"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
SCREEN = ROOT / "screen.png"
TEMPLATE = ROOT / "template.png"


def nms(boxes, scores, iou_thresh=0.3):
    """简单 NMS，去掉重叠匹配。boxes: [(x,y,w,h)]"""
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
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


def match(screen_gray, tmpl_gray, threshold):
    res = cv2.matchTemplate(screen_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    h, w = tmpl_gray.shape
    boxes = [(int(x), int(y), w, h) for x, y in zip(xs, ys)]
    scores = [float(res[y, x]) for x, y in zip(xs, ys)]
    keep = nms(boxes, scores)
    return [(boxes[i], scores[i]) for i in keep]


SCALE = 1.5  # 用户提供的 template 相对实机截图缩小了 1.5x


def main():
    screen = cv2.imread(str(SCREEN))
    tmpl = cv2.imread(str(TEMPLATE))
    if screen is None or tmpl is None:
        raise SystemExit(f"读图失败: screen={SCREEN.exists()} tmpl={TEMPLATE.exists()}")

    # 把模板放大到实机尺寸
    h0, w0 = tmpl.shape[:2]
    tmpl = cv2.resize(tmpl, (int(w0 * SCALE), int(h0 * SCALE)))

    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape
    print(f"screen: {screen.shape}, template(scaled x{SCALE}): {tmpl.shape}")

    # 1) 全模板匹配（精确，可能只匹配同计数的按钮）
    full_hits = match(screen_gray, tmpl_gray, threshold=0.75)
    print(f"\n[full template] hits @0.75: {len(full_hits)}")
    for (b, s) in full_hits:
        print(f"  box={b} score={s:.3f}")

    # 2) 左半模板（去掉数字区域，约前 55%），匹配所有计数的按钮
    left_tmpl = tmpl_gray[:, : int(tw * 0.55)]
    left_hits = match(screen_gray, left_tmpl, threshold=0.7)
    print(f"\n[left half template] hits @0.70: {len(left_hits)}")
    for (b, s) in left_hits:
        print(f"  box={b} score={s:.3f}")

    # 画在图上
    out = screen.copy()
    for (b, s) in full_hits:
        x, y, w, h = b
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(out, f"FULL {s:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for (b, s) in left_hits:
        x, y, w, h = b
        # 用整按钮宽度回推（按模板原宽 tw）
        cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 255, 0), 2)
        cv2.putText(out, f"L {s:.2f}", (x, y + th + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    cv2.imwrite(str(ROOT / "matched.png"), out)
    print(f"\n输出: {ROOT / 'matched.png'}")


if __name__ == "__main__":
    main()
