"""vision_core_straight.py - picture processing for the camera homework (no ROS in here).

Two detectors, both work on a BGR picture (numpy array) and both are written so they behave the same
with OpenCV 4.5 (Ubuntu 22.04 on the Pi) and newer versions:

  detect_lane(frame)  two white stripes on a dark street -> middle between the stripes
  detect_rect(frame)  one white rectangle on a dark background -> centre and direction of its long axis

draw_lane / draw_rect paint the result into a copy of the picture (like the screenshots in the homework).
"""
import math
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------
# Lane detection
# ---------------------------------------------------------------------------------------------
LANE_PARAMS = {
    'method': 'tophat',     # 'tophat' (default): tape = narrow AND brighter than the floor right next to it,
                            #   works whatever the exposure, lamp or white table around the sheet;
                            # 'global': one grey level for the whole picture (old method, uses thresh/peak_frac)
    'tophat_width_frac': 0.18,  # anything brighter but WIDER than this (fraction of W, measured across a row)
                                # is background (table, wall) and drops out; must be wider than the tape looks
    'min_contrast': 30,     # tape must be at least this many grey levels brighter than the floor next to it
    'contrast_frac': 0.45,  # ... and at least this fraction of the strongest contrast in the picture
    'thresh': 0,            # 'global' method only: grey level for "white"; 0 = automatic
    'thresh_min': 120,      # automatic threshold is kept between these two values
    'thresh_max': 225,
    'peak_frac': 0.93,      # see _binary_mask
    'car_y_frac': 0.82,     # height of the car marker in the picture (fraction of the image height)
    'band_frac': 0.04,      # stripe position is measured in this band around the car row (fraction of H)
    'min_height_frac': 0.30,  # a stripe must be at least this tall (fraction of H)
    'max_width_frac': 0.14,   # ... and on average at most this wide (fraction of W)
    'min_fill': 0.25,       # white pixels / bounding box area (x 0.3) - very ragged shapes are not stripes
    'max_border_frac': 0.6, # a white area that touches the left/right picture edge in more than this fraction
                            # of its rows is background (table, paper edge), not tape
    'min_sep_frac': 0.12,   # the two stripes must be at least this far apart (fraction of W)
    'lane_width_frac': 0.70,  # lane width at the car row (fraction of W) used when only one line is seen,
                              # until both lines have been seen once (then the measured width is used)
    'car_half_width_frac': 0.065,  # half width of the car marker (fraction of W): WARNING as soon as the
                                   # car body touches a lane line, not only its centre (as in the demo)
    'lookahead_frac': 0.35  # length of the steering arrow (fraction of H)
}

LOST_FRAMES = 15            # after this many pictures without any line, the remembered lane is forgotten


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


def _tophat_mask(gray, p):
    """White = narrow bright lines. A horizontal 'opening' estimates the floor brightness under each pixel
    (it removes everything bright that is narrower than the kernel); tape = picture - floor > contrast.
    Large white areas (table) are wider than the kernel, so they belong to the 'floor' and drop out.
    Returns the mask and the contrast threshold used."""
    h_img, w_img = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    k_w = max(9, int(p['tophat_width_frac'] * w_img)) | 1
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1)))
    strongest = float(np.percentile(tophat, 99.5))
    t = max(float(p['min_contrast']), p['contrast_frac'] * strongest)
    _, mask = cv2.threshold(tophat, t, 255, cv2.THRESH_BINARY)
    return mask, t


def _find_stripes(mask, p, car_y):
    """All white shapes that look like tape: tall, thin, enclosed by dark (not the picture edge)."""
    h_img, w_img = mask.shape[:2]
    band = max(2, int(p['band_frac'] * h_img))
    y0, y1 = int(car_y - band), int(car_y + band)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stripes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < p['min_height_frac'] * h_img:
            continue
        mean_w = float(area) / float(h)            # average width (the bounding box is wider if slanted)
        if mean_w > p['max_width_frac'] * w_img or mean_w < 3:
            continue
        if area < p['min_fill'] * w * h * 0.3:
            continue
        if x == 0 or x + w >= w_img:               # touches a side edge: in how many of its rows?
            sub = labels[y:y + h, :]
            edge_rows = np.count_nonzero((sub[:, 0] == i) | (sub[:, w_img - 1] == i))
            if edge_rows > p['max_border_frac'] * h:
                continue
        ys, xs = np.nonzero(labels[max(0, y0):min(h_img, y1 + 1), :] == i)
        if len(xs) == 0:                          # does not reach the car row: use its lowest rows
            ys, xs = np.nonzero(labels[y + max(0, h - 6):y + h, :] == i)
            if len(xs) == 0:
                continue
        stripes.append({'x': float(xs.mean()), 'height': int(h), 'bbox': (int(x), int(y), int(w), int(h))})
    stripes.sort(key=lambda s: s['x'])
    return stripes


def new_lane_state():
    """Memory of the lane between pictures (one per node)."""
    return {'left': None, 'right': None, 'lane_w': None, 'lost': 0}


def detect_lane(frame, params=None, state=None):
    """Find the lane the car is in and decide whether the car centre is inside it.

    A line that is not visible (out of the picture) is estimated from the other line and the lane width.
    WARNING only when the car centre is outside the lane, or when no line is visible at all.
    Which side a single visible line belongs to (left or right) is taken from the previous pictures, so a
    car that drives over a line is detected as "outside". Pass the same state dict for every picture.

    Returns a dict:
      status        'inside' | 'outside' | 'no_lane'
      ok            True if status == 'inside'
      car           (car_center_x, car_y)
      left_x, right_x   lane lines at the car row [px]; seen_left / seen_right: True if seen, False if estimated
      lane_center_x middle of the lane [px] (None if no_lane)
      offset_px     lane_center_x - car_center_x (positive: lane middle to the right -> steer right)
      offset_norm   offset_px / (W/2), limited to -1 ... 1 (None if no_lane)
      stripes       all tape candidates at the car row; mask, thresh for debugging
    """
    p = dict(LANE_PARAMS)
    if params:
        p.update(params)
    if state is None:
        state = new_lane_state()
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if p.get('method', 'tophat') == 'global':
        mask, t = _binary_mask(gray, p['thresh'], p['thresh_min'], p['thresh_max'], p['peak_frac'])
    else:
        mask, t = _tophat_mask(gray, p)
    k_h = max(5, int(0.06 * h_img))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    car_x = w_img / 2.0
    car_y = p['car_y_frac'] * h_img
    stripes = _find_stripes(mask, p, car_y)
    xs = [s['x'] for s in stripes]
    lane_w = state['lane_w'] or p['lane_width_frac'] * w_img
    min_sep = p['min_sep_frac'] * w_img
    have_prev = state['left'] is not None and state['right'] is not None

    left = right = None
    seen_l = seen_r = False
    pairs = [(a, b) for a, b in zip(xs[:-1], xs[1:]) if b - a >= min_sep]
    if pairs:
        if have_prev:       # the pair closest to the lane of the last picture
            a, b = min(pairs, key=lambda q: abs(q[0] - state['left']) + abs(q[1] - state['right']))
        else:               # the pair around the car, otherwise the one whose middle is closest to it
            around = [q for q in pairs if q[0] <= car_x <= q[1]]
            a, b = min(around or pairs, key=lambda q: abs(0.5 * (q[0] + q[1]) - car_x))
        left, right, seen_l, seen_r = a, b, True, True
        state['lane_w'] = (b - a) if state['lane_w'] is None else 0.8 * state['lane_w'] + 0.2 * (b - a)
    elif xs:
        if have_prev:       # nearest remembered line decides the side
            x = min(xs, key=lambda v: min(abs(v - state['left']), abs(v - state['right'])))
            is_left = abs(x - state['left']) <= abs(x - state['right'])
        else:               # first picture: assume the car starts inside the lane
            x = min(xs, key=lambda v: abs(v - car_x))
            is_left = x <= car_x
        if is_left:
            left, right, seen_l = x, x + lane_w, True
        else:
            left, right, seen_r = x - lane_w, x, True

    res = {'status': 'no_lane', 'ok': False, 'car': (car_x, car_y), 'left_x': left, 'right_x': right,
           'seen_left': seen_l, 'seen_right': seen_r, 'lane_center_x': None, 'offset_px': None,
           'offset_norm': None, 'stripes': stripes, 'mask': mask, 'thresh': t, 'size': (w_img, h_img),
           'lookahead': p['lookahead_frac'] * h_img, 'car_half_w': p['car_half_width_frac'] * w_img}
    if left is None:
        state['lost'] += 1
        if state['lost'] >= LOST_FRAMES:
            state.update({'left': None, 'right': None})
        return res
    state.update({'left': left, 'right': right, 'lost': 0})
    mid = 0.5 * (left + right)
    off = mid - car_x
    res.update({'lane_center_x': mid, 'offset_px': off,
                'offset_norm': float(max(-1.0, min(1.0, off / (w_img / 2.0))))})
    m = p['car_half_width_frac'] * w_img
    res['car_half_w'] = m
    res['status'] = 'inside' if left + m <= car_x <= right - m else 'outside'
    res['ok'] = res['status'] == 'inside'
    return res


def draw_lane(frame, res):
    """Copy of the picture: lane lines (seen = yellow, estimated = orange), blue car marker and green
    steering arrow when inside the lane; red marker and WARNING when outside or no lane."""
    out = frame.copy()
    h_img, w_img = out.shape[:2]
    cx, cy = int(round(res['car'][0])), int(round(res['car'][1]))
    bw, bh = max(6, int(round(res['car_half_w']))), max(6, h_img // 60)   # box = car body width
    for s in res['stripes']:                      # every tape candidate: small yellow dot
        cv2.circle(out, (int(round(s['x'])), cy), 4, (0, 255, 255), -1)
    for x, seen in ((res['left_x'], res['seen_left']), (res['right_x'], res['seen_right'])):
        if x is not None and not seen:            # estimated line (out of view): orange ring, may be off-picture
            xi = int(round(min(max(x, 0), w_img - 1)))
            cv2.circle(out, (xi, cy), 7, (0, 140, 255), 2)
    if res['status'] == 'inside':
        tip = (int(round(res['lane_center_x'])), int(round(cy - res['lookahead'])))
        cv2.arrowedLine(out, (cx, cy), tip, (0, 255, 0), 3, tipLength=0.15)
        cv2.rectangle(out, (cx - bw, cy - bh), (cx + bw, cy + bh), (255, 0, 0), 2)
    else:
        cv2.rectangle(out, (cx - bw, cy - bh), (cx + bw, cy + bh), (0, 0, 255), 2)
        cv2.putText(out, 'WARNING', (cx - 70, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        text = 'car outside the lane' if res['status'] == 'outside' else 'no lane visible'
        cv2.putText(out, text, (cx - 90, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


# ---------------------------------------------------------------------------------------------
# Rectangle direction
# ---------------------------------------------------------------------------------------------
RECT_PARAMS = {
    'method': 'tophat',     # 'tophat' (default): white = brighter than its surroundings and thinner than
                            #   tophat_size_frac in every direction; adapts to exposure, lamp and white table.
                            # 'global': one grey level for the whole picture (old method)
    'tophat_size_frac': 0.20,  # must be larger than the rectangle's SHORT side (fraction of the picture width)
    'min_contrast': 30,     # at least this many grey levels brighter than the surroundings
    'contrast_frac': 0.45,  # ... and at least this fraction of the strongest contrast in the picture
    'reject_border': True,  # ignore white shapes that touch the picture edge (table slivers)
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
    if p.get('method', 'tophat') == 'global':
        mask, t = _binary_mask(gray, p['thresh'], p['thresh_min'], p['thresh_max'], p['peak_frac'])
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        k = max(9, int(p['tophat_size_frac'] * w_img)) | 1
        tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        t = max(float(p['min_contrast']), p['contrast_frac'] * float(np.percentile(tophat, 99.5)))
        _, mask = cv2.threshold(tophat, t, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    res = {'ok': False, 'center': None, 'angle_deg': None, 'p1': None, 'p2': None, 'length': 0.0,
           'width': 0.0, 'mask': mask, 'thresh': t, 'size': (w_img, h_img)}
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[-2]                          # works with the 2-value and the 3-value return
    if p.get('reject_border', True):
        def touches(c):
            x, y, w, h = cv2.boundingRect(c)
            return x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img
        contours = [c for c in contours if not touches(c)]
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


# ---------------------------------------------------------------------------------------------
# Frame rate and processing time (used by the nodes for a one-line report per second)
# ---------------------------------------------------------------------------------------------
class RateMeter:
    """tick() after every processed frame; it returns a report text once per second, otherwise None."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.n = 0
        self.busy = 0.0

    def tick(self, busy_seconds):
        self.n += 1
        self.busy += busy_seconds
        now = time.perf_counter()
        if now - self.t0 < 1.0:
            return None
        text = '%.1f frames/s processed, %.1f ms per frame' % (self.n / (now - self.t0), 1000.0 * self.busy / self.n)
        self.t0, self.n, self.busy = now, 0, 0.0
        return text
