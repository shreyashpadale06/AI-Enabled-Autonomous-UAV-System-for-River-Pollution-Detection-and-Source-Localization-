"""Hotspot rule from Section III: same-class detections within r metres are
grouped; a group is a hotspot if >=k detections, from >=m distinct frames,
with mean confidence > s. Reported at the centroid."""
import math
from dataclasses import dataclass, field

@dataclass
class Detection:
    cls: str; conf: float; lat: float; lon: float
    frame_id: int; t: float = 0.0; image: str = ""

@dataclass
class Hotspot:
    cls: str; lat: float; lon: float; n: int; mean_conf: float
    frames: int; images: list = field(default_factory=list)

def haversine_m(a_lat, a_lon, b_lat, b_lon):
    R = 6371000.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lon - a_lon)
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def find_hotspots(dets, r=25.0, k=3, m=2, s=0.5):
    hotspots = []
    for cls in {d.cls for d in dets}:
        pts, seen = [d for d in dets if d.cls == cls], set()
        # single-linkage grouping: connected components under distance <= r
        for i in range(len(pts)):
            if i in seen: continue
            comp, stack = [], [i]; seen.add(i)
            while stack:
                a = stack.pop(); comp.append(pts[a])
                for j in range(len(pts)):
                    if j not in seen and haversine_m(pts[a].lat, pts[a].lon, pts[j].lat, pts[j].lon) <= r:
                        seen.add(j); stack.append(j)
            frames = {d.frame_id for d in comp}
            mean_c = sum(d.conf for d in comp) / len(comp)
            if len(comp) >= k and len(frames) >= m and mean_c > s:
                top = sorted(comp, key=lambda d: -d.conf)[:3]
                hotspots.append(Hotspot(cls, sum(d.lat for d in comp)/len(comp),
                    sum(d.lon for d in comp)/len(comp), len(comp), mean_c,
                    len(frames), [d.image for d in top if d.image]))
    return hotspots

if __name__ == "__main__":   # self-test with synthetic data
    d = [Detection("wastewater", .8, 18.5200+i*1e-5, 73.8500, f) for i, f in enumerate([1,2,3,4])]
    d += [Detection("floating_waste", .9, 18.53, 73.86, 9)]           # lone detection -> rejected
    for h in find_hotspots(d): print(h)
