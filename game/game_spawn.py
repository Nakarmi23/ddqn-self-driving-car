"""Spawn validation shared by `Game._check_spawn()` and `scripts/check_spawns.py`.

A spawn is valid when the car starts in the last lap segment, between the
last checkpoint and checkpoint 0 (the start/finish line), facing checkpoint 0,
with the hitbox at least `SPAWN_MARGIN` px from every checkpoint and touching
no wall. A hitbox on a checkpoint counts the pass on step 1 without the car
moving (issue #8), and a small margin lets any small change to the hitbox
size or spawn bring that back.
"""
import math

from .game_point import Point

SPAWN_MARGIN = 10  # px between the spawn hitbox and the nearest checkpoint


def _cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _on_segment(px, py, x1, y1, x2, y2):
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)


def _segments_intersect(a, b, c, d):
    """True if segment a-b touches segment c-d, collinear overlap included."""
    d1 = _cross(c[0], c[1], d[0], d[1], a[0], a[1])
    d2 = _cross(c[0], c[1], d[0], d[1], b[0], b[1])
    d3 = _cross(a[0], a[1], b[0], b[1], c[0], c[1])
    d4 = _cross(a[0], a[1], b[0], b[1], d[0], d[1])
    if ((d1 > 0) != (d2 > 0) and d1 != 0 and d2 != 0
            and (d3 > 0) != (d4 > 0) and d3 != 0 and d4 != 0):
        return True
    return ((d1 == 0 and _on_segment(*a, *c, *d))
            or (d2 == 0 and _on_segment(*b, *c, *d))
            or (d3 == 0 and _on_segment(*c, *a, *b))
            or (d4 == 0 and _on_segment(*d, *a, *b)))


def _point_segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (
        (p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _segment_distance(a, b, c, d):
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def hitbox_distance(car, x1, y1, x2, y2):
    """Shortest distance from the car's hitbox outline to segment (x1,y1)-(x2,y2);
    0 if they touch. Unlike Car.check_checkpoint_collision this also catches a
    segment lying along a hitbox edge."""
    corners = [(p.x, p.y) for p in (car.front_right, car.back_right,
                                    car.back_left, car.front_left)]
    edges = zip(corners, corners[1:] + corners[:1])
    return min(_segment_distance(a, b, (x1, y1), (x2, y2)) for a, b in edges)


def _first_checkpoint_on_ray(car, checkpoints, direction):
    """Index of the first checkpoint a ray from the car's centre hits, going
    forward (direction=1) or backward (direction=-1) along its heading."""
    heading = car._rotate_point(Point(0, 1), car.angle)  # unit vector of vel > 0
    dx, dy = direction * heading.x, direction * heading.y
    best = None
    for i, cp in enumerate(checkpoints):
        ex, ey = cp.x2 - cp.x1, cp.y2 - cp.y1
        den = dx * ey - dy * ex
        if den == 0:
            continue
        t = ((cp.x1 - car.x) * ey - (cp.y1 - car.y) * ex) / den
        u = ((cp.x1 - car.x) * dy - (cp.y1 - car.y) * dx) / den
        if t >= 0 and 0 <= u <= 1 and (best is None or t < best[1]):
            best = (i, t)
    return None if best is None else best[0]


def spawn_problems(car, track):
    """Every reason the car (as spawned) is not a valid start for `track`;
    an empty list means the spawn is fine."""
    problems = []
    checkpoints = track.checkpoints

    ahead = _first_checkpoint_on_ray(car, checkpoints, 1)
    if ahead != 0:
        problems.append(
            f"first checkpoint ahead is {ahead}, not 0: spawn must face "
            f"checkpoint 0 from behind it")
    behind = _first_checkpoint_on_ray(car, checkpoints, -1)
    if behind != len(checkpoints) - 1:
        problems.append(
            f"first checkpoint behind is {behind}, not {len(checkpoints) - 1}: "
            f"spawn must sit between the last checkpoint and checkpoint 0")

    for i, cp in enumerate(checkpoints):
        dist = hitbox_distance(car, cp.x1, cp.y1, cp.x2, cp.y2)
        if dist < SPAWN_MARGIN:
            problems.append(f"hitbox is {dist:.1f} px from checkpoint {i} "
                            f"(need >= {SPAWN_MARGIN})")

    if any(hitbox_distance(car, w.x1, w.y1, w.x2, w.y2) == 0
           for w in track.walls):
        problems.append("hitbox touches a wall")
    return problems
