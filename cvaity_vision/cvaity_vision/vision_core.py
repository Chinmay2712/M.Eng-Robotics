"""vision_core.py - picture processing for the camera homework (no ROS in here).

Two detectors, both work on a BGR picture (numpy array) and both are written so they behave the same
with OpenCV 4.5 (Ubuntu 22.04 on the Pi) and newer versions:

  detect_lane(frame)  two white stripes on a dark street -> middle between the stripes
  detect_rect(frame)  one white rectangle on a dark background -> centre and direction of its long axis

draw_lane / draw_rect paint the result into a copy of the picture (like the screenshots in the homework).
"""
import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------
# Lane detection
# ---------------------------------------------------------------------------------------------
LANE_PARAMS = {
    'thresh': 0,            # grey level above which a pixel counts as "white"; 0 = automatic (Otsu)
    'thresh_min': 120,      # automatic threshold is kept between these two values
    'thresh_max': 225,
    'peak_frac': 0.93,      # see _binary_mask
    'car_y_frac': 0.82,     # height of the car marker in the picture (fraction of the image height)
    'band_frac': 0.04,      # stripe position is measured in this band around the car row (fraction of H)
    'min_height_frac': 0.30,  # a stripe must be at least this tall (fraction of H)
    'max_width_frac': 0.14,   # ... and at most this wide (fraction of W); wider white areas are background
    'min_fill': 0.25,       # white pixels / bounding box area of a stripe
    'min_sep_frac': 0.12,   # the two stripes must be at least this far apart (fraction of W)
    'lookahead_frac': 0.35  # length of the steering arrow (fraction of H)
}


def _binary_mask(gray, thresh, tmin, tmax, peak_frac=0.93):
    """White = 255.

    Fixed threshold if thresh > 0. Otherwise the automatic threshold is the larger of
      - Otsu's threshold (kept inside [tmin, tmax]) and
      - peak_frac x the brightness of the brightest 1 % of the picture.
    The second part separates white paper / tape from bright glare and reflections, which are
    usually a bit darker than the tape itself.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if thresh and thresh > 0:
        t = float(thresh)
    else:
        t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = min(max(t, tmin), tmax)
        peak = float(np.percentile(blur, 99.0))
        t = min(250.0, max(t, peak_frac * peak))
    _, mask = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
    return mask, t


def detect_lane(frame, params=None):
    """Find the two white stripes and the middle between them.

    Returns a dict:
      ok            True if two stripes were found
      car           (car_center_x, car_y) fixed position of the car in the picture
      lane_center_x middle between the stripes [px] (None if not ok)
      left_x, right_x  stripe positions at the car row [px]
      offset_px     lane_center_x - car_center_x  (positive: the lane middle is to the right -> steer right)
      offset_norm   offset_px / (W/2), limited to -1 ... 1
      stripes       list of all stripe candidates (x, height, score)
      mask, thresh  binary picture used and the threshold (for debugging)
    """
    p = dict(LANE_PARAMS)
    if params:
        p.update(params)
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, t = _binary_mask(gray, p['thresh'], p['thresh_min'], p['thresh_max'], p['peak_frac'])

    # keep only tall, thin, bright shapes: opening with a tall kernel removes specks and glare texture
    k_h = max(5, int(0.06 * h_img))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    car_x = w_img / 2.0
    car_y = p['car_y_frac'] * h_img
    band = max(2, int(p['band_frac'] * h_img))
    y0, y1 = int(car_y - band), int(car_y + band)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stripes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < p['min_height_frac'] * h_img:
            continue
        mean_w = float(area) / float(h)            # average stripe width (the bounding box is wider if the stripe is slanted)
        if mean_w > p['max_width_frac'] * w_img or mean_w < 3:
            continue
        if area < p['min_fill'] * w * h * 0.3:     # very ragged shapes (glare) are not stripes
            continue
        ys, xs = np.nonzero(labels[max(0, y0):min(h_img, y1 + 1), :] == i)
        if len(xs) == 0:                          # stripe does not reach the car row: use its lowest rows
            ys, xs = np.nonzero(labels[y + max(0, h - 6):y + h, :] == i)
            if len(xs) == 0:
                continue
        sx = float(xs.mean())
        stripes.append({'x': sx, 'height': int(h), 'score': float(h) * float(area) / float(w * h),
                        'bbox': (int(x), int(y), int(w), int(h))})
    stripes.sort(key=lambda s: s['x'])

    result = {'ok': False, 'car': (car_x, car_y), 'lane_center_x': None, 'left_x': None, 'right_x': None,
              'offset_px': None, 'offset_norm': None, 'stripes': stripes, 'mask': mask, 'thresh': t,
              'size': (w_img, h_img), 'lookahead': p['lookahead_frac'] * h_img}

    # choose the pair of neighbouring stripes whose middle is closest to the car
    best = None
    for a, b in zip(stripes[:-1], stripes[1:]):
        if b['x'] - a['x'] < p['min_sep_frac'] * w_img:
            continue
        mid = 0.5 * (a['x'] + b['x'])
        d = abs(mid - car_x)
        if best is None or d < best[0]:
            best = (d, a, b)
    if best is not None:
        _, a, b = best
        mid = 0.5 * (a['x'] + b['x'])
        off = mid - car_x
        result.update({'ok': True, 'lane_center_x': mid, 'left_x': a['x'], 'right_x': b['x'],
                       'offset_px': off, 'offset_norm': float(max(-1.0, min(1.0, off / (w_img / 2.0))))})
    return result


def draw_lane(frame, res):
    """Copy of the picture with: blue car marker, green steering arrow (or red marker + WARNING)."""
    out = frame.copy()
    h_img, w_img = out.shape[:2]
    cx, cy = int(round(res['car'][0])), int(round(res['car'][1]))
    bw, bh = max(6, w_img // 70), max(4, h_img // 90)
    for s in res['stripes']:                      # all stripe candidates: small yellow dots (debug)
        cv2.circle(out, (int(round(s['x'])), cy), 4, (0, 255, 255), -1)
    if res['ok']:
        tip = (int(round(res['lane_center_x'])), int(round(cy - res['lookahead'])))
        cv2.arrowedLine(out, (cx, cy), tip, (0, 255, 0), 3, tipLength=0.15)
        cv2.rectangle(out, (cx - bw, cy - bh), (cx + bw, cy + bh), (255, 0, 0), 2)
    else:
        cv2.rectangle(out, (cx - bw, cy - bh), (cx + bw, cy + bh), (0, 0, 255), 2)
        cv2.putText(out, 'WARNING', (cx - 70, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out


# ---------------------------------------------------------------------------------------------
# Rectangle direction
# ---------------------------------------------------------------------------------------------
RECT_PARAMS = {
    'thresh': 0,
    'thresh_min': 150,
    'thresh_max': 235,
    'peak_frac': 0.93,
    'min_area_frac': 0.01,   # the rectangle must cover at least this part of the picture
    'min_elongation': 1.5    # long side / short side; below this the direction is not clear
}


def detect_rect(frame, params=None):
    """Find the biggest white shape and the direction of its long axis.

    The direction comes from the image moments of the shape, not from cv2.minAreaRect, because the
    angle convention of minAreaRect changed between OpenCV versions.

    Returns a dict:
      ok          True if a shape with a clear long axis was found
      center      (x, y) in pixels
      angle_deg   direction of the long axis, 0 = to the right, 90 = up, 0 <= angle < 180
      p1, p2      end points of the long axis (pixels)
      length, width   extent along and across the long axis [px]
      mask, thresh    binary picture and threshold (for debugging)
    """
    p = dict(RECT_PARAMS)
    if params:
        p.update(params)
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, t = _binary_mask(gray, p['thresh'], p['thresh_min'], p['thresh_max'], p['peak_frac'])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    res = {'ok': False, 'center': None, 'angle_deg': None, 'p1': None, 'p2': None, 'length': 0.0,
           'width': 0.0, 'mask': mask, 'thresh': t, 'size': (w_img, h_img)}
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[-2]                          # works with the 2-value and the 3-value return
    if not contours:
        return res
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < p['min_area_frac'] * w_img * h_img:
        return res

    m = cv2.moments(cnt)
    if m['m00'] <= 0:
        return res
    cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    theta = 0.5 * math.atan2(2.0 * m['mu11'], m['mu20'] - m['mu02'])   # long axis, image coordinates (y down)
    ux, uy = math.cos(theta), math.sin(theta)

    pts = cnt.reshape(-1, 2).astype(np.float64)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    t_min, t_max = float(along.min()), float(along.max())
    length = t_max - t_min
    width = float(across.max() - across.min())

    res['center'] = (cx, cy)
    res['p1'] = (cx + t_min * ux, cy + t_min * uy)
    res['p2'] = (cx + t_max * ux, cy + t_max * uy)
    res['length'], res['width'] = length, width
    ang = (-math.degrees(theta)) % 180.0                   # y up, counter-clockwise
    res['angle_deg'] = 0.0 if ang > 179.999 else ang
    res['ok'] = width > 0 and (length / max(width, 1.0)) >= p['min_elongation']
    return res


def draw_rect(frame, res, arrow=False):
    """Copy of the picture with a red line along the long axis, a blue centre dot and the angle."""
    out = frame.copy()
    if res['center'] is None:
        cv2.putText(out, 'WARNING: no rectangle', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return out
    c = (int(round(res['center'][0])), int(round(res['center'][1])))
    p1 = (int(round(res['p1'][0])), int(round(res['p1'][1])))
    p2 = (int(round(res['p2'][0])), int(round(res['p2'][1])))
    if arrow:   # head at the end that points more to the right (if vertical: the upper end)
        right = p2 if (p2[0], -p2[1]) > (p1[0], -p1[1]) else p1
        left = p1 if right is p2 else p2
        cv2.arrowedLine(out, left, right, (0, 255, 0), 3, tipLength=0.12)
    cv2.line(out, p1, p2, (0, 0, 255), 2)
    cv2.circle(out, c, 6, (255, 0, 0), -1)
    if res['ok']:
        cv2.putText(out, 'angle %.0f deg' % res['angle_deg'], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
    else:
        cv2.putText(out, 'WARNING: direction not clear', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out
