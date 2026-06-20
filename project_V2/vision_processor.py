"""
================================================================================
  vision_processor.py  —  Camera + YOLO Obstacle Detection
================================================================================
  Version 9.2 — SignDetector Integration

  Changes from v9.1:
      ✓ Added SignDetector integration (sign_detector.py)
      ✓ _sign_detector initialized in __init__
      ✓ sign_detector.load() + start() called inside start()
      ✓ process_frame() called every frame inside _capture_loop()
      ✓ get_latest_sign() added to public API
      ✓ stop() calls sign_detector.stop()
      ✓ Graceful: if sign_detector unavailable, robot continues normally

  Changes from v9.0:
      ✓ Added EnvironmentClassifier using ResNet18-Places365
      ✓ Environment classification runs every ENV_EVERY_N frames
      ✓ Environment label shown in HUD
      ✓ latest_env_label / latest_env_conf accessible via get_environment()

  Author  : Robot Team
  Version : 9.2
================================================================================
"""

import cv2
import threading
import time
import logging
import numpy as np

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("VisionProcessor")


# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

YOLO_EVERY_N = 3
YOLO_CONF    = 0.45
YOLO_MODEL   = "yolov8n.pt"

# ── Environment Classifier ─────────────────────
ENV_EVERY_N       = 30
ENV_MODEL_WEIGHTS = "resnet18_places365.pth"
ENV_TOP1_THRESH   = 0.20

# ── Trapezoid ROI ──────────────────────────────
TRAP_LANE_WIDTH_PCT  = 85
TRAP_TOP_NARROW_PCT  = 25
TRAP_Y_TOP_PCT       = 30
TRAP_Y_BOTTOM_PCT    = 97

# ── Proximity gate ─────────────────────────────
STOP_LINE_PCT = 50

# ── Box filters ────────────────────────────────
MIN_BOX_AREA_RATIO = 0.008
MIN_BOX_WIDTH      = 20
MIN_BOX_HEIGHT     = 20
MIN_BOX_ASPECT     = 0.2

# ── Sliding window persistence ─────────────────
PERSIST_WINDOW = 5
PERSIST_HITS   = 3

# ── Stale detection ────────────────────────────
MAX_DETECTION_AGE_S = 0.5

# ── Danger score weights ───────────────────────
DANGER_CENTER_WEIGHT = 1.8
DANGER_BOTTOM_WEIGHT = 1.4
DANGER_AREA_WEIGHT   = 1.2

# ── Class filters ──────────────────────────────
IGNORE_CLASSES         = []
VALID_OBSTACLE_CLASSES = [
    "bottle", "cup", "bowl", "vase", "book", "cell phone",
    "laptop", "keyboard", "mouse", "remote", "clock",
    "chair", "couch", "bed", "dining table", "desk",
    "suitcase", "backpack", "handbag",
    "refrigerator", "microwave", "oven", "sink",
    "tv", "monitor",
    "potted plant", "umbrella", "box",
    "cat", "dog",
    "person",
]

HUMAN_CLASS    = "person"
HUMAN_MIN_CONF = 0.75

# ── HUD colors ─────────────────────────────────
COLOR_CRITICAL = (0,   0, 255)
COLOR_HIGH     = (0,  80, 255)
COLOR_MEDIUM   = (0, 165, 255)
COLOR_LOW      = (0, 255, 255)
COLOR_CLEAR    = (0, 200,   0)
COLOR_HUMAN    = (0,   0, 255)
COLOR_INFO     = (200, 200,   0)
COLOR_BLACK    = (0,   0,   0)
COLOR_ENV      = (180, 255, 180)

# ── RTSP push target ───────────────────────────
MEDIAMTX_RTSP = "rtsp://127.0.0.1:8554/cam"


# ──────────────────────────────────────────────
#  DATA MODELS
# ──────────────────────────────────────────────

@dataclass
class DetectionResult:
    timestamp:      float = field(default_factory=time.time)
    position:       str   = "NONE"
    confidence:     float = 0.0
    label:          str   = ""
    box:            tuple = ()
    center_x:       int   = 0
    frame_width:    int   = FRAME_WIDTH
    danger_score:   float = 0.0
    detection_age:  float = 0.0
    persistent:     bool  = False
    threat_level:   str   = "NONE"
    approx_dist:    str   = "FAR"
    is_approaching: bool  = False
    is_human:       bool  = False

    def is_valid(self, max_age: float = 0.5) -> bool:
        return (time.time() - self.timestamp) < max_age

    def obstacle_found(self) -> bool:
        return self.position != "NONE"


@dataclass
class ObstacleThreatLevel:
    position:       str   = "NONE"
    threat:         str   = "NONE"
    confidence:     float = 0.0
    label:          str   = ""
    approx_dist:    str   = "FAR"
    is_approaching: bool  = False
    box_area_pct:   float = 0.0
    cy_pct:         float = 0.0


@dataclass
class EnvironmentResult:
    label:      str   = "Unknown"
    confidence: float = 0.0
    timestamp:  float = field(default_factory=time.time)

    def is_valid(self, max_age: float = 10.0) -> bool:
        return (time.time() - self.timestamp) < max_age


# ──────────────────────────────────────────────
#  TRAPEZOID ROI HELPERS
# ──────────────────────────────────────────────

def build_trapezoid_roi(frame_w: int, frame_h: int) -> np.ndarray:
    lane_w  = int(frame_w * (TRAP_LANE_WIDTH_PCT / 100.0))
    lane_w  = max(60, min(frame_w - 10, lane_w))
    x_left  = (frame_w - lane_w) // 2
    x_right = x_left + lane_w

    y_top    = int(frame_h * (TRAP_Y_TOP_PCT    / 100.0))
    y_bottom = int(frame_h * (TRAP_Y_BOTTOM_PCT / 100.0))
    y_top    = max(10, min(frame_h - 10, y_top))
    y_bottom = max(y_top + 10, min(frame_h - 1, y_bottom))

    shrink = int(lane_w * (TRAP_TOP_NARROW_PCT / 100.0))
    shrink = max(0, min(lane_w // 2 - 5, shrink))

    return np.array([[
        (x_left,           y_bottom),
        (x_left  + shrink, y_top),
        (x_right - shrink, y_top),
        (x_right,          y_bottom),
    ]], dtype=np.int32)


def point_in_trapezoid(roi_contour: np.ndarray, px: float, py: float) -> bool:
    return cv2.pointPolygonTest(roi_contour, (px, py), False) >= 0


def box_inside_trapezoid(roi_contour: np.ndarray, box: dict) -> bool:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    test_pts = [
        (float(cx),       float(y2)),
        (float(x1 + 2),   float(y2)),
        (float(x2 - 2),   float(y2)),
        (float(cx),       float(cy)),
        (float(x1 + 2),   float(cy)),
        (float(x2 - 2),   float(cy)),
    ]
    return any(point_in_trapezoid(roi_contour, px, py) for px, py in test_pts)


# ──────────────────────────────────────────────
#  SLIDING WINDOW PERSISTENCE
# ──────────────────────────────────────────────

class SlidingWindowPersistence:

    def __init__(self, window: int = PERSIST_WINDOW, hits: int = PERSIST_HITS):
        self._window  = window
        self._hits    = hits
        self._history = deque(maxlen=window)

    def update(self, detected: bool) -> bool:
        self._history.append(detected)
        return sum(self._history) >= self._hits

    def reset(self):
        self._history.clear()

    @property
    def score(self) -> int:
        return sum(self._history)


# ──────────────────────────────────────────────
#  SMART OBSTACLE ANALYZER
# ──────────────────────────────────────────────

class SmartObstacleAnalyzer:

    DIST_THRESHOLDS = {
        "VERY_CLOSE": 0.12,
        "CLOSE":      0.05,
        "MEDIUM":     0.02,
        "FAR":        0.0,
    }

    def __init__(self):
        self._history: list = []

    def analyze(self, boxes: list, frame_w: int, frame_h: int) -> ObstacleThreatLevel:
        if not boxes:
            if self._history:
                self._history.pop(0)
            return ObstacleThreatLevel()

        best       = boxes[0]
        frame_area = frame_w * frame_h
        box_area   = (best["x2"] - best["x1"]) * (best["y2"] - best["y1"])
        area_ratio = box_area / frame_area
        cy_ratio   = best["cy"] / frame_h

        approx_dist = "FAR"
        for dist_name, threshold in self.DIST_THRESHOLDS.items():
            if area_ratio >= threshold:
                approx_dist = dist_name
                break

        self._history.append({"area": area_ratio, "cx": best["cx"]})
        if len(self._history) > 6:
            self._history.pop(0)

        is_approaching = False
        if len(self._history) >= 3:
            areas = [h["area"] for h in self._history[-3:]]
            is_approaching = areas[-1] > areas[0] * 1.1

        score  = min(area_ratio / 0.12, 1.0) * 40
        score += min(cy_ratio   / 0.95, 1.0) * 25
        score += (20.0 if is_approaching else 0.0)
        score += min(best["conf"],        1.0) * 15

        if   score >= 70: threat = "CRITICAL"
        elif score >= 50: threat = "HIGH"
        elif score >= 30: threat = "MEDIUM"
        elif score >= 10: threat = "LOW"
        else:             threat = "LOW"

        return ObstacleThreatLevel(
            position       = "FORWARD",
            threat         = threat,
            confidence     = best["conf"],
            label          = best["label"],
            approx_dist    = approx_dist,
            is_approaching = is_approaching,
            box_area_pct   = area_ratio * 100,
            cy_pct         = cy_ratio   * 100,
        )


# ──────────────────────────────────────────────
#  ENVIRONMENT CLASSIFIER
# ──────────────────────────────────────────────

class EnvironmentClassifier:

    _LABEL_REMAP = {
        # ── INDOOR ────────────────────────────────────────────────────
        "airplane_cabin":             "Indoor",
        "airport_terminal":           "Indoor",
        "alcove":                     "Indoor",
        "amusement_arcade":           "Indoor",
        "aquarium":                   "Indoor",
        "archive":                    "Indoor",
        "arena/hockey":               "Indoor",
        "arena/performance":          "Indoor",
        "art_gallery":                "Indoor",
        "art_school":                 "Indoor",
        "art_studio":                 "Indoor",
        "artists_loft":               "Indoor",
        "assembly_line":              "Indoor",
        "atrium/public":              "Indoor",
        "attic":                      "Indoor",
        "auditorium":                 "Indoor",
        "auto_factory":               "Indoor",
        "auto_showroom":              "Indoor",
        "bakery/shop":                "Indoor",
        "balcony/interior":           "Indoor",
        "ball_pit":                   "Indoor",
        "ballroom":                   "Indoor",
        "bank_vault":                 "Indoor",
        "banquet_hall":               "Indoor",
        "bar":                        "Indoor",
        "barn":                       "Indoor",
        "basement":                   "Indoor",
        "basketball_court/indoor":    "Indoor",
        "bathroom":                   "Indoor",
        "bazaar/indoor":              "Indoor",
        "bedchamber":                 "Indoor",
        "bedroom":                    "Indoor",
        "biology_laboratory":         "Indoor",
        "bookstore":                  "Indoor",
        "booth/indoor":               "Indoor",
        "bow_window/indoor":          "Indoor",
        "bowling_alley":              "Indoor",
        "boxing_ring":                "Indoor",
        "bus_interior":               "Indoor",
        "bus_station/indoor":         "Indoor",
        "cafeteria":                  "Indoor",
        "car_interior":               "Indoor",
        "chemistry_lab":              "Indoor",
        "childs_room":                "Indoor",
        "church/indoor":              "Indoor",
        "classroom":                  "Indoor",
        "clean_room":                 "Indoor",
        "closet":                     "Indoor",
        "clothing_store":             "Indoor",
        "cockpit":                    "Indoor",
        "coffee_shop":                "Indoor",
        "computer_room":              "Indoor",
        "conference_center":          "Indoor",
        "conference_room":            "Indoor",
        "corridor":                   "Indoor",
        "diner/indoor":               "Indoor",
        "dining_hall":                "Indoor",
        "dining_room":                "Indoor",
        "discotheque":                "Indoor",
        "dorm_room":                  "Indoor",
        "dressing_room":              "Indoor",
        "elevator/door":              "Indoor",
        "elevator_lobby":             "Indoor",
        "elevator_shaft":             "Indoor",
        "embassy":                    "Indoor",
        "engine_room":                "Indoor",
        "entrance_hall":              "Indoor",
        "escalator/indoor":           "Indoor",
        "fabric_store":               "Indoor",
        "fastfood_restaurant":        "Indoor",
        "flea_market/indoor":         "Indoor",
        "florist_shop/indoor":        "Indoor",
        "food_court":                 "Indoor",
        "galley":                     "Indoor",
        "garage/indoor":              "Indoor",
        "general_store/indoor":       "Indoor",
        "greenhouse/indoor":          "Indoor",
        "gymnasium/indoor":           "Indoor",
        "hangar/indoor":              "Indoor",
        "hardware_store":             "Indoor",
        "home_office":                "Indoor",
        "home_theater":               "Indoor",
        "hospital":                   "Indoor",
        "hospital_room":              "Indoor",
        "hotel_room":                 "Indoor",
        "ice_cream_parlor":           "Indoor",
        "ice_skating_rink/indoor":    "Indoor",
        "jacuzzi/indoor":             "Indoor",
        "jail_cell":                  "Indoor",
        "jewelry_shop":               "Indoor",
        "kindergarden_classroom":     "Indoor",
        "kitchen":                    "Indoor",
        "laundromat":                 "Indoor",
        "lecture_room":               "Indoor",
        "legislative_chamber":        "Indoor",
        "library/indoor":             "Indoor",
        "living_room":                "Indoor",
        "loading_dock":               "Indoor",
        "lobby":                      "Indoor",
        "lock_chamber":               "Indoor",
        "locker_room":                "Indoor",
        "market/indoor":              "Indoor",
        "martial_arts_gym":           "Indoor",
        "movie_theater/indoor":       "Indoor",
        "museum/indoor":              "Indoor",
        "music_studio":               "Indoor",
        "natural_history_museum":     "Indoor",
        "nursery":                    "Indoor",
        "nursing_home":               "Indoor",
        "office":                     "Indoor",
        "office_cubicles":            "Indoor",
        "operating_room":             "Indoor",
        "orchestra_pit":              "Indoor",
        "pantry":                     "Indoor",
        "parking_garage/indoor":      "Indoor",
        "pet_shop":                   "Indoor",
        "pharmacy":                   "Indoor",
        "physics_laboratory":         "Indoor",
        "playroom":                   "Indoor",
        "pub/indoor":                 "Indoor",
        "reception":                  "Indoor",
        "recreation_room":            "Indoor",
        "repair_shop":                "Indoor",
        "restaurant":                 "Indoor",
        "restaurant_kitchen":         "Indoor",
        "sauna":                      "Indoor",
        "server_room":                "Indoor",
        "shoe_shop":                  "Indoor",
        "shopping_mall/indoor":       "Indoor",
        "shower":                     "Indoor",
        "stable":                     "Indoor",
        "stage/indoor":               "Indoor",
        "staircase":                  "Indoor",
        "storage_room":               "Indoor",
        "subway_station/platform":    "Indoor",
        "supermarket":                "Indoor",
        "sushi_bar":                  "Indoor",
        "swimming_pool/indoor":       "Indoor",
        "television_room":            "Indoor",
        "television_studio":          "Indoor",
        "throne_room":                "Indoor",
        "ticket_booth":               "Indoor",
        "toyshop":                    "Indoor",
        "train_interior":             "Indoor",
        "utility_room":               "Indoor",
        "waiting_room":               "Indoor",
        "wet_bar":                    "Indoor",
        "beauty_salon":               "Indoor",
        "beer_hall":                  "Indoor",
        "berth":                      "Indoor",
        "butchers_shop":              "Indoor",
        "candy_store":                "Indoor",
        "catacomb":                   "Indoor",
        "delicatessen":               "Indoor",
        "department_store":           "Indoor",
        "drugstore":                  "Indoor",
        "gift_shop":                  "Indoor",
        "ice_cream_parlor":           "Indoor",
        "mezzanine":                  "Indoor",
        "pizzeria":                   "Indoor",
        "science_museum":             "Indoor",
        "veterinarians_office":       "Indoor",

        # ── DESERT ────────────────────────────────────────────────────
        "desert/sand":                "Desert",
        "desert/vegetation":          "Desert",
        "desert_road":                "Desert",
        "badlands":                   "Desert",
        "butte":                      "Desert",
        "tundra":                     "Desert",
        "sandbar":                    "Desert",

        # ── FIELD / GRASS ─────────────────────────────────────────────
        "field/cultivated":           "Field / Grass",
        "field/wild":                 "Field / Grass",
        "corn_field":                 "Field / Grass",
        "wheat_field":                "Field / Grass",
        "hayfield":                   "Field / Grass",
        "lawn":                       "Field / Grass",
        "pasture":                    "Field / Grass",
        "rice_paddy":                 "Field / Grass",
        "farm":                       "Field / Grass",
        "orchard":                    "Field / Grass",
        "vegetable_garden":           "Field / Grass",
        "tree_farm":                  "Field / Grass",
        "soccer_field":               "Field / Grass",
        "football_field":             "Field / Grass",
        "athletic_field/outdoor":     "Field / Grass",
        "golf_course":                "Field / Grass",
        "campsite":                   "Field / Grass",
        "picnic_area":                "Field / Grass",
        "corral":                     "Field / Grass",
        "park":                       "Field / Grass",
        "beer_garden":                "Field / Grass",

        # ── FOREST ────────────────────────────────────────────────────
        "forest/broadleaf":           "Forest",
        "forest_path":                "Forest",
        "forest_road":                "Forest",
        "bamboo_forest":              "Forest",
        "rainforest":                 "Forest",
        "tree_house":                 "Forest",
        "topiary_garden":             "Forest",
        "botanical_garden":           "Forest",
        "japanese_garden":            "Forest",
        "formal_garden":              "Forest",
        "zen_garden":                 "Forest",
        "vineyard":                   "Forest",
        "greenhouse/outdoor":         "Forest",
        "roof_garden":                "Forest",

        # ── DIRT ROAD / PATH ──────────────────────────────────────────
        "field_road":                 "Dirt Road",
        "mountain_path":              "Dirt Road",
        "dirt_road":                  "Dirt Road",
        "path":                       "Dirt Road",
        "railroad_track":             "Dirt Road",
        "rope_bridge":                "Dirt Road",
        "bridge":                     "Dirt Road",
        "viaduct":                    "Dirt Road",
        "aqueduct":                   "Dirt Road",

        # ── URBAN / STREET ────────────────────────────────────────────
        "street":                     "Urban Street",
        "alley":                      "Urban Street",
        "downtown":                   "Urban Street",
        "crosswalk":                  "Urban Street",
        "driveway":                   "Urban Street",
        "highway":                    "Urban Street",
        "parking_lot":                "Urban Street",
        "parking_garage/outdoor":     "Urban Street",
        "gas_station":                "Urban Street",
        "residential_neighborhood":   "Urban Street",
        "slum":                       "Urban Street",
        "plaza":                      "Urban Street",
        "promenade":                  "Urban Street",
        "boardwalk":                  "Urban Street",
        "construction_site":          "Urban Street",
        "industrial_area":            "Urban Street",
        "junkyard":                   "Urban Street",
        "landfill":                   "Urban Street",
        "runway":                     "Urban Street",
        "raceway":                    "Urban Street",
        "racecourse":                 "Urban Street",
        "building_facade":            "Urban Street",
        "skyscraper":                 "Urban Street",
        "fire_escape":                "Urban Street",
        "village":                    "Urban Street",
        "medina":                     "Urban Street",
        "kasbah":                     "Urban Street",
        "airfield":                   "Urban Street",
        "amphitheater":               "Urban Street",
        "amusement_park":             "Urban Street",
        "apartment_building/outdoor": "Urban Street",
        "arch":                       "Urban Street",
        "army_base":                  "Urban Street",
        "balcony/exterior":           "Urban Street",
        "bazaar/outdoor":             "Urban Street",
        "bullring":                   "Urban Street",
        "burial_chamber":             "Urban Street",
        "cabin/outdoor":              "Urban Street",
        "campus":                     "Urban Street",
        "carrousel":                  "Urban Street",
        "castle":                     "Urban Street",
        "cemetery":                   "Urban Street",
        "chalet":                     "Urban Street",
        "church/outdoor":             "Urban Street",
        "cottage":                    "Urban Street",
        "courthouse":                 "Urban Street",
        "courtyard":                  "Urban Street",
        "diner/outdoor":              "Urban Street",
        "doorway/outdoor":            "Urban Street",
        "fire_station":               "Urban Street",
        "fountain":                   "Urban Street",
        "garage/outdoor":             "Urban Street",
        "gazebo/exterior":            "Urban Street",
        "general_store/outdoor":      "Urban Street",
        "hangar/outdoor":             "Urban Street",
        "heliport":                   "Urban Street",
        "hotel/outdoor":              "Urban Street",
        "house":                      "Urban Street",
        "hunting_lodge/outdoor":      "Urban Street",
        "igloo":                      "Urban Street",
        "inn/outdoor":                "Urban Street",
        "kennel/outdoor":             "Urban Street",
        "library/outdoor":            "Urban Street",
        "lighthouse":                 "Urban Street",
        "mansion":                    "Urban Street",
        "manufactured_home":          "Urban Street",
        "market/outdoor":             "Urban Street",
        "mausoleum":                  "Urban Street",
        "mosque/outdoor":             "Urban Street",
        "motel":                      "Urban Street",
        "museum/outdoor":             "Urban Street",
        "oast_house":                 "Urban Street",
        "office_building":            "Urban Street",
        "oilrig":                     "Urban Street",
        "pagoda":                     "Urban Street",
        "palace":                     "Urban Street",
        "patio":                      "Urban Street",
        "pavilion":                   "Urban Street",
        "phone_booth":                "Urban Street",
        "playground":                 "Urban Street",
        "porch":                      "Urban Street",
        "restaurant_patio":           "Urban Street",
        "sandbox":                    "Urban Street",
        "schoolhouse":                "Urban Street",
        "shed":                       "Urban Street",
        "shopfront":                  "Urban Street",
        "ski_resort":                 "Urban Street",
        "stage/outdoor":              "Urban Street",
        "synagogue/outdoor":          "Urban Street",
        "temple/asia":                "Urban Street",
        "tower":                      "Urban Street",
        "train_station/platform":     "Urban Street",
        "volleyball_court/outdoor":   "Urban Street",
        "wind_farm":                  "Urban Street",
        "windmill":                   "Urban Street",
        "yard":                       "Urban Street",
        "youth_hostel":               "Urban Street",
        "baseball_field":             "Urban Street",
        "stadium/baseball":           "Urban Street",
        "stadium/football":           "Urban Street",
        "stadium/soccer":             "Urban Street",
        "landing_deck":               "Urban Street",

        # ── ROCKY TERRAIN ─────────────────────────────────────────────
        "mountain":                   "Rocky Terrain",
        "mountain_snowy":             "Rocky Terrain",
        "cliff":                      "Rocky Terrain",
        "canyon":                     "Rocky Terrain",
        "crevasse":                   "Rocky Terrain",
        "grotto":                     "Rocky Terrain",
        "glacier":                    "Rocky Terrain",
        "ice_floe":                   "Rocky Terrain",
        "ice_shelf":                  "Rocky Terrain",
        "iceberg":                    "Rocky Terrain",
        "snowfield":                  "Rocky Terrain",
        "ski_slope":                  "Rocky Terrain",
        "rock_arch":                  "Rocky Terrain",
        "archaelogical_excavation":   "Rocky Terrain",
        "excavation":                 "Rocky Terrain",
        "trench":                     "Rocky Terrain",
        "volcano":                    "Rocky Terrain",
        "valley":                     "Rocky Terrain",
        "ruin":                       "Rocky Terrain",

        # ── RIVERBANK / MUDDY ─────────────────────────────────────────
        "river":                      "Riverbank",
        "canal/natural":              "Riverbank",
        "canal/urban":                "Riverbank",
        "creek":                      "Riverbank",
        "lake/natural":               "Riverbank",
        "lagoon":                     "Riverbank",
        "marsh":                      "Riverbank",
        "swamp":                      "Riverbank",
        "pond":                       "Riverbank",
        "watering_hole":              "Riverbank",
        "swimming_hole":              "Riverbank",
        "hot_spring":                 "Riverbank",
        "coast":                      "Riverbank",
        "beach":                      "Riverbank",
        "beach_house":                "Riverbank",
        "ocean":                      "Riverbank",
        "wave":                       "Riverbank",
        "islet":                      "Riverbank",
        "harbor":                     "Riverbank",
        "pier":                       "Riverbank",
        "dam":                        "Riverbank",
        "fishpond":                   "Riverbank",
        "underwater/ocean_deep":      "Riverbank",
        "moat/water":                 "Riverbank",
        "water_tower":                "Riverbank",
        "waterfall":                  "Riverbank",
        "raft":                       "Riverbank",
        "boat_deck":                  "Riverbank",
        "boathouse":                  "Riverbank",
        "water_park":                 "Riverbank",
        "swimming_pool/outdoor":      "Riverbank",
        "ice_skating_rink/outdoor":   "Riverbank",
    }

    def __init__(self, weights_path: str = ENV_MODEL_WEIGHTS, device: str = "cpu"):
        self._weights_path = weights_path
        self._device       = device
        self._model        = None
        self._transform    = None
        self._classes      = []
        self._loaded       = False

        self._lock         = threading.Lock()
        self._latest_env   = EnvironmentResult()

        self._pending_frame = None
        self._ready_event   = threading.Event()
        self._running       = False
        self._thread        = None

    def load(self) -> bool:
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            log.info(f"Loading EnvironmentClassifier from: {self._weights_path}")

            model = models.resnet18(pretrained=False)
            model.fc = __import__('torch').nn.Linear(model.fc.in_features, 365)

            state = torch.load(self._weights_path, map_location=self._device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
                state = {k.replace("module.", ""): v for k, v in state.items()}

            model.load_state_dict(state, strict=False)
            model.eval()
            model.to(self._device)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225],
                ),
            ])

            self._classes   = self._load_places365_classes()
            self._model     = model
            self._transform = transform
            self._loaded    = True

            log.info("EnvironmentClassifier loaded ✓")
            return True

        except ImportError as e:
            log.warning(f"EnvironmentClassifier: torch/torchvision not installed — {e}")
            return False
        except FileNotFoundError:
            log.warning(
                f"EnvironmentClassifier: weights file not found → {self._weights_path}. "
                "Environment classification disabled."
            )
            return False
        except Exception as e:
            log.error(f"EnvironmentClassifier load failed: {e}")
            return False

    @staticmethod
    def _load_places365_classes() -> list:
        import os
        candidates = [
            "categories_places365.txt",
            os.path.join(os.path.dirname(__file__), "categories_places365.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                classes = []
                for line in lines:
                    parts = line.split()
                    name = parts[0].lstrip("/")
                    classes.append(name)
                log.info(f"Places365 classes loaded: {len(classes)}")
                return classes
        log.warning("categories_places365.txt not found — using class indices")
        return [f"env_{i}" for i in range(365)]

    def start(self):
        if not self._loaded:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._inference_loop, daemon=True, name="EnvClassifier"
        )
        self._thread.start()
        log.info("EnvironmentClassifier thread started")

    def stop(self):
        self._running = False
        self._ready_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit_frame(self, frame: np.ndarray):
        if not self._loaded or not self._running:
            return
        if not self._ready_event.is_set():
            self._pending_frame = frame.copy()
            self._ready_event.set()

    def get_latest(self) -> EnvironmentResult:
        with self._lock:
            return self._latest_env

    def _inference_loop(self):
        while self._running:
            self._ready_event.wait(timeout=2.0)
            if not self._running:
                break

            frame = self._pending_frame
            self._pending_frame = None
            self._ready_event.clear()

            if frame is None:
                continue

            try:
                result = self._run_inference(frame)
                with self._lock:
                    self._latest_env = result
            except Exception as e:
                log.error(f"EnvironmentClassifier inference error: {e}")

    def _run_inference(self, frame: np.ndarray) -> EnvironmentResult:
        import torch

        t0  = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top_conf, top_idx = float(probs.max()), int(probs.argmax())

        if self._classes and top_idx < len(self._classes):
            raw_label = self._classes[top_idx]
        else:
            raw_label = f"env_{top_idx}"

        clean = raw_label.lstrip("/")
        parts = clean.split("/", 1)
        clean = parts[1] if len(parts) > 1 else parts[0]

        label = self._LABEL_REMAP.get(clean, None)
        if label is None:
            indoor_hints = ["indoor", "room", "hall", "shop", "store",
                            "office", "kitchen", "studio", "theater", "salon", "gym"]
            label = "Indoor" if any(h in clean for h in indoor_hints) else "Unknown"

        if top_conf < ENV_TOP1_THRESH:
            label = "Unknown"

        elapsed_ms = (time.time() - t0) * 1000
        log.debug(f"Env: {label} ({top_conf:.0%}) — {elapsed_ms:.0f}ms")

        return EnvironmentResult(label=label, confidence=top_conf)


# ──────────────────────────────────────────────
#  MAIN CLASS
# ──────────────────────────────────────────────

class VisionProcessor:

    def __init__(
        self,
        camera_index:   int  = 0,
        use_yolo:       bool = True,
        use_env:        bool = True,
        device:         str  = "cpu",
        simulate:       bool = False,
        adaptive_conf        = None,
    ):
        self.camera_index  = camera_index
        self.use_yolo      = use_yolo
        self.use_env       = use_env
        self.device        = device
        self.simulate      = simulate
        self.adaptive_conf = adaptive_conf

        self._cap     = None
        self._model   = None
        self._running = False

        self._lock            = threading.Lock()
        self._latest_result   = DetectionResult()
        self._latest_frame    = None
        self._annotated_frame = None

        self._frame_count   = 0
        self._yolo_count    = 0
        self._pending_boxes = []

        self._capture_thread   = None
        self._yolo_thread      = None
        self._yolo_frame       = None
        self._yolo_ready       = threading.Event()
        self._yolo_result_lock = threading.Lock()

        self._last_yolo_time = 0.0

        # ── ROI cache (trapezoid points / contour / dim-mask) ──
        # بيتحسب مرة واحدة بس لكل (w, h) ويتخزن، بدل ما يتعاد حسابه
        # كل فريم في _run_yolo و _draw_annotations
        self._roi_cache_dims     = None
        self._roi_pts            = None
        self._roi_contour        = None
        self._roi_outside_mask   = None

        self._persistence = SlidingWindowPersistence()
        self._analyzer    = SmartObstacleAnalyzer()

        # ── Environment Classifier ──────────────
        self._env_classifier = EnvironmentClassifier(device=device)

        # ── Sign Detector ───────────────────────
        from sign_detector import SignDetector
        self._sign_detector           = SignDetector()
        self._sign_detector_available = False
        self._sign_thread             = None
        self._sign_frame              = None
        self._sign_ready              = threading.Event()
        self._sign_frame_lock         = threading.Lock()
        self._sign_prev_danger        = False  # state-change logging فقط

        # GStreamer push pipeline to MediaMTX
        self._gst_pipeline = None

    # ──────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────

    def start(self) -> bool:
        self._running = True

        # ── Load & start environment classifier ──
        if self.use_env:
            loaded = self._env_classifier.load()
            if loaded:
                self._env_classifier.start()
            else:
                log.warning("Environment classifier disabled (load failed)")
                self.use_env = False

        # ── Load & start sign detector ───────────
        self._sign_detector_available = self._sign_detector.load()
        if self._sign_detector_available:
            self._sign_detector.start()
            log.info("SignDetector started ✓")
        else:
            log.warning("SignDetector failed to load — sign detection disabled")

        if self.simulate:
            log.info("VisionProcessor — SIMULATION mode")
            self._capture_thread = threading.Thread(
                target=self._sim_loop, daemon=True, name="VisionSim"
            )
            self._capture_thread.start()
            self._start_gst_push()
            return True

        if not self._open_camera():
            log.error("Failed to open camera")
            return False

        if self.use_yolo:
            self._load_yolo()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="Vision-Capture"
        )
        self._capture_thread.start()

        if self.use_yolo and self._model:
            self._yolo_thread = threading.Thread(
                target=self._yolo_loop, daemon=True, name="Vision-YOLO"
            )
            self._yolo_thread.start()

        if self._sign_detector_available:
            self._sign_thread = threading.Thread(
                target=self._sign_loop, daemon=True, name="Vision-Sign"
            )
            self._sign_thread.start()

        log.info(
            f"VisionProcessor started | camera={self.camera_index} "
            f"yolo={self.use_yolo} env={self.use_env} sign={self._sign_detector_available}"
        )

        self._start_gst_push()
        return True

    def stop(self):
        self._running = False
        self._yolo_ready.set()
        self._sign_ready.set()   # ← wake sign thread عشان يخرج من الـ wait

        if self.use_env:
            self._env_classifier.stop()

        # ── Stop sign detector ───────────────────
        if self._sign_detector_available:
            self._sign_detector.stop()

        if self._gst_pipeline:
            self._gst_pipeline.release()
            self._gst_pipeline = None
            log.info("GStreamer push pipeline stopped")

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._yolo_thread:
            self._yolo_thread.join(timeout=2.0)
        if self._sign_thread:
            self._sign_thread.join(timeout=2.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        log.info("VisionProcessor stopped")

    def get_latest(self) -> DetectionResult:
        with self._lock:
            return self._latest_result

    def get_frame(self):
        with self._lock:
            return self._annotated_frame.copy() if self._annotated_frame is not None else None

    def get_raw_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_environment(self) -> EnvironmentResult:
        return self._env_classifier.get_latest()

    def get_latest_sign(self):
        """يرجع آخر نتيجة لكشف لوحات الخطر (SignDetectionResult)."""
        return self._sign_detector.get_latest()

    # ──────────────────────────────────────────
    #  GSTREAMER PUSH TO MEDIAMTX
    # ──────────────────────────────────────────

    def _start_gst_push(self):
        nvenc_pipeline = (
            'appsrc ! videoconvert '
            '! video/x-raw,format=I420 '
            '! nvv4l2h264enc bitrate=2000000 iframeinterval=30 '
            '! h264parse '
            f'! rtspclientsink location={MEDIAMTX_RTSP} protocols=tcp'
        )
        x264_pipeline = (
            'appsrc ! videoconvert '
            '! video/x-raw,format=I420 '
            '! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 '
            '! h264parse '
            f'! rtspclientsink location={MEDIAMTX_RTSP} protocols=tcp'
        )

        writer = cv2.VideoWriter(
            nvenc_pipeline, cv2.CAP_GSTREAMER, 0, 30,
            (FRAME_WIDTH, FRAME_HEIGHT), True
        )
        if writer.isOpened():
            self._gst_pipeline = writer
            log.info(f"GStreamer NVENC push → {MEDIAMTX_RTSP}")
            return
        writer.release()

        writer = cv2.VideoWriter(
            x264_pipeline, cv2.CAP_GSTREAMER, 0, 30,
            (FRAME_WIDTH, FRAME_HEIGHT), True
        )
        if writer.isOpened():
            self._gst_pipeline = writer
            log.info(f"GStreamer x264 push → {MEDIAMTX_RTSP}")
            return
        writer.release()

        log.error(
            "GStreamer push pipeline failed to open. "
            "Make sure MediaMTX is running and GStreamer is installed."
        )
        self._gst_pipeline = None

    def _push_frame(self, frame: np.ndarray):
        if self._gst_pipeline and self._gst_pipeline.isOpened():
            self._gst_pipeline.write(frame)

    # ──────────────────────────────────────────
    #  CAMERA
    # ──────────────────────────────────────────

    def _open_camera(self) -> bool:
        import platform
        if platform.system() == "Windows":
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            gst = self._gstreamer_pipeline()
            self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                log.warning("GStreamer failed → fallback V4L2")
                self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        return True

    @staticmethod
    def _gstreamer_pipeline(width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30) -> str:
        return (
            f"v4l2src device=/dev/video0 ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true"
        )

    # ──────────────────────────────────────────
    #  ROI CACHE
    # ──────────────────────────────────────────

    def _get_roi(self, w: int, h: int):
        """
        بيرجع (roi_pts, roi_contour, outside_mask) لمقاس فريم (w, h).
        بيتحسب مرة واحدة بس ويتخزن — لو نفس المقاس جه تاني (الحالة العادية
        لأن الكاميرا ثابتة على FRAME_WIDTH × FRAME_HEIGHT)، بيرجع نفس
        القيم المحفوظة من غير ما يعيد build_trapezoid_roi / fillPoly تاني.
        لو المقاس اتغير (نادر)، بيعيد الحساب تلقائي.
        """
        if self._roi_cache_dims != (w, h):
            roi_pts     = build_trapezoid_roi(w, h)
            roi_contour = roi_pts[0].astype(np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, roi_pts, 255)
            outside_mask = (mask == 0)

            self._roi_cache_dims   = (w, h)
            self._roi_pts          = roi_pts
            self._roi_contour      = roi_contour
            self._roi_outside_mask = outside_mask

        return self._roi_pts, self._roi_contour, self._roi_outside_mask

    # ──────────────────────────────────────────
    #  YOLO
    # ──────────────────────────────────────────

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            log.info(f"Loading YOLO: {YOLO_MODEL} on {self.device}")
            self._model = YOLO(YOLO_MODEL)
            dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            log.info("YOLO loaded and warmed up ✓")
        except ImportError:
            log.warning("ultralytics not installed — YOLO disabled")
            self.use_yolo = False
        except Exception as e:
            log.error(f"YOLO load failed: {e}")
            self.use_yolo = False

    # ──────────────────────────────────────────
    #  CAPTURE LOOP
    # ──────────────────────────────────────────

    def _capture_loop(self):
        while self._running:
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self._frame_count += 1

            # ── Sign detection — async (thread منفصل) ───
            if self._sign_detector_available:
                with self._sign_frame_lock:
                    self._sign_frame = frame   # latest frame فقط — القديم بيتبدل
                self._sign_ready.set()

            # ── YOLO every N frames ──────────────
            if (
                self.use_yolo
                and self._model
                and self._frame_count % YOLO_EVERY_N == 0
            ):
                with self._yolo_result_lock:
                    self._yolo_frame = frame.copy()
                self._yolo_ready.set()

            # ── Environment every ENV_EVERY_N frames ──
            if self.use_env and self._frame_count % ENV_EVERY_N == 0:
                self._env_classifier.submit_frame(frame)

            # clear stale detections
            if (time.time() - self._last_yolo_time) > MAX_DETECTION_AGE_S:
                with self._yolo_result_lock:
                    self._pending_boxes = []

            result    = self._compute_result(frame)
            annotated = self._draw_annotations(frame.copy(), result)

            with self._lock:
                self._latest_frame    = frame
                self._annotated_frame = annotated
                self._latest_result   = result

            self._push_frame(annotated)

    # ──────────────────────────────────────────
    #  YOLO THREAD
    # ──────────────────────────────────────────

    def _yolo_loop(self):
        while self._running:
            self._yolo_ready.wait(timeout=1.0)
            if not self._running:
                break

            with self._yolo_result_lock:
                frame            = self._yolo_frame
                self._yolo_frame = None
            self._yolo_ready.clear()

            if frame is None:
                continue

            try:
                boxes = self._run_yolo(frame)
                with self._yolo_result_lock:
                    self._pending_boxes = boxes
                self._yolo_count += 1
            except Exception as e:
                log.error(f"YOLO inference error: {e}")

    # ──────────────────────────────────────────
    #  SIGN DETECTOR THREAD
    # ──────────────────────────────────────────

    def _sign_loop(self):
        """
        Thread منفصل للـ sign detector — مش بيـblock الـ capture loop خالص.
        بياخد آخر frame متاح ويشتغل عليه. لو الـ processing بطيئة،
        الفريمات اللي بينهم بتتجاهل (latest-frame-wins).
        """
        while self._running:
            self._sign_ready.wait(timeout=1.0)
            if not self._running:
                break

            with self._sign_frame_lock:
                frame           = self._sign_frame
                self._sign_frame = None
            self._sign_ready.clear()

            if frame is None:
                continue

            try:
                sign_result = self._sign_detector.process_frame(frame)
                danger_now  = sign_result.danger_confirmed

                if danger_now and not self._sign_prev_danger:
                    log.warning(
                        f"[SignDetector] DANGER confirmed — "
                        f"reason={sign_result.reason} "
                        f"word='{sign_result.text_matched_word}'"
                    )
                elif not danger_now and self._sign_prev_danger:
                    log.info("[SignDetector] CLEAR — danger sign no longer detected")

                self._sign_prev_danger = danger_now

            except Exception as e:
                log.error(f"SignDetector process_frame error: {e}")

    def _run_yolo(self, frame) -> list:
        self._last_yolo_time = time.time()

        h, w       = frame.shape[:2]
        frame_area = w * h

        roi_pts, roi_contour, _ = self._get_roi(w, h)

        current_conf = (
            self.adaptive_conf.update(frame)
            if self.adaptive_conf is not None
            else YOLO_CONF
        )

        results = self._model(
            frame, imgsz=320, conf=current_conf,
            device=self.device, verbose=False,
        )

        boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = self._model.names[cls]

                if label in IGNORE_CLASSES:
                    continue
                if VALID_OBSTACLE_CLASSES and label not in VALID_OBSTACLE_CLASSES:
                    continue

                bw = x2 - x1
                bh = y2 - y1
                if bw < MIN_BOX_WIDTH or bh < MIN_BOX_HEIGHT:
                    continue
                box_area = bw * bh
                if box_area < frame_area * MIN_BOX_AREA_RATIO:
                    continue
                if bw > 0 and (bh / bw) < MIN_BOX_ASPECT:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                is_human = (label.lower() == HUMAN_CLASS) and (conf >= HUMAN_MIN_CONF)

                candidate = {
                    "label":      label,
                    "conf":       conf,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "cx": cx,  "cy": cy,
                    "area_ratio": box_area / frame_area,
                    "is_human":   is_human,
                    "danger_score": self._calculate_danger_score(
                        {"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                         "cx":cx,"cy":cy,"conf":conf},
                        frame.shape
                    ),
                    "_roi_contour": roi_contour,
                }
                boxes.append(candidate)

        boxes = [
            b for b in boxes
            if b["is_human"] or box_inside_trapezoid(b["_roi_contour"], b)
        ]

        stop_line_y = int(h * (STOP_LINE_PCT / 100.0))
        boxes = [
            b for b in boxes
            if b["is_human"] or b["y2"] >= stop_line_y
        ]

        boxes.sort(key=lambda b: (not b["is_human"], -b["danger_score"]))
        return boxes

    # ──────────────────────────────────────────
    #  DANGER SCORE
    # ──────────────────────────────────────────

    def _calculate_danger_score(self, box: dict, frame_shape) -> float:
        h, w         = frame_shape[:2]
        bw           = box["x2"] - box["x1"]
        bh           = box["y2"] - box["y1"]
        area_ratio   = (bw * bh) / float(w * h)
        center_dist  = abs(box["cx"] - (w // 2))
        center_norm  = 1.0 - min(center_dist / (w // 2), 1.0)
        bottom_norm  = box["y2"] / float(h)

        return (
            box["conf"]
            * (1.0 + area_ratio  * DANGER_AREA_WEIGHT)
            * (1.0 + center_norm * DANGER_CENTER_WEIGHT)
            * (1.0 + bottom_norm * DANGER_BOTTOM_WEIGHT)
        )

    # ──────────────────────────────────────────
    #  RESULT COMPUTATION
    # ──────────────────────────────────────────

    def _compute_result(self, frame) -> DetectionResult:
        with self._yolo_result_lock:
            boxes = list(self._pending_boxes)

        w             = frame.shape[1]
        h             = frame.shape[0]
        detection_age = time.time() - self._last_yolo_time

        detected   = len(boxes) > 0
        persistent = self._persistence.update(detected)

        if not boxes:
            self._analyzer.analyze([], w, h)
            return DetectionResult(
                position      = "FORWARD" if persistent else "NONE",
                frame_width   = w,
                detection_age = detection_age,
                persistent    = persistent,
            )

        threat = self._analyzer.analyze(boxes, w, h)
        best   = boxes[0]

        return DetectionResult(
            timestamp      = time.time(),
            position       = "FORWARD" if persistent else "NONE",
            confidence     = best["conf"],
            label          = best["label"],
            box            = (best["x1"], best["y1"], best["x2"], best["y2"]),
            center_x       = best["cx"],
            frame_width    = w,
            danger_score   = best["danger_score"],
            detection_age  = detection_age,
            persistent     = persistent,
            threat_level   = threat.threat,
            approx_dist    = threat.approx_dist,
            is_approaching = threat.is_approaching,
            is_human       = best.get("is_human", False),
        )

    # ──────────────────────────────────────────
    #  HUD DRAWING
    # ──────────────────────────────────────────

    def _draw_annotations(self, frame, result: DetectionResult) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. Dim outside trapezoid
        roi_pts, roi_contour, outside_mask = self._get_roi(w, h)
        overlay = frame.copy()
        overlay[outside_mask] = (overlay[outside_mask] * 0.35).astype(np.uint8)
        frame = overlay

        # 2. Trapezoid border
        roi_color = COLOR_HUMAN if result.is_human else (60, 60, 60)
        cv2.polylines(frame, roi_pts, True, roi_color, 2)

        # 3. Stop line
        stop_y     = int(h * STOP_LINE_PCT / 100.0)
        stop_color = COLOR_CRITICAL if result.obstacle_found() else (0, 220, 220)
        cv2.line(frame, (0, stop_y), (w, stop_y), stop_color, 1)
        cv2.putText(frame, "DETECTION LINE", (4, stop_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, stop_color, 1)

        # 4. All bounding boxes
        threat_colors = {
            "CRITICAL": COLOR_CRITICAL,
            "HIGH":     COLOR_HIGH,
            "MEDIUM":   COLOR_MEDIUM,
            "LOW":      COLOR_LOW,
            "NONE":     COLOR_CLEAR,
        }
        with self._yolo_result_lock:
            raw_boxes = list(self._pending_boxes)

        # ملحوظة: raw_boxes متفلترة بالفعل بالنسبة لعضوية الـ trapezoid
        # (الفلتر اتعمل مرة واحدة في _run_yolo) — مفيش داعي نعيد
        # build_trapezoid_roi ولا box_inside_trapezoid تاني هنا.
        all_boxes = raw_boxes

        for i, box in enumerate(all_boxes):
            is_human   = box.get("is_human", False)
            is_primary = (i == 0)

            if is_human:
                color = COLOR_HUMAN
            elif is_primary:
                color = threat_colors.get(result.threat_level, COLOR_CLEAR)
            else:
                color = (70, 70, 70)

            thickness = 2 if is_primary else 1
            cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, thickness)

            if is_primary:
                area_pct = box.get("area_ratio", 0) * 100
                tag = (
                    f"!! HUMAN {box['conf']:.0%}"
                    if is_human
                    else f"{box['label'].upper()}  {box['conf']:.0%}  {area_pct:.1f}%"
                )
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    frame,
                    (box["x1"],          box["y1"] - th - 8),
                    (box["x1"] + tw + 6, box["y1"]),
                    COLOR_BLACK, -1
                )
                cv2.putText(frame, tag, (box["x1"] + 3, box["y1"] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(frame, (box["cx"], box["cy"]), 4, COLOR_CLEAR, -1)

        # 5. Approaching arrow
        if result.is_approaching:
            ax = w - 35
            ay = int(h * TRAP_Y_TOP_PCT / 100) + 35
            cv2.arrowedLine(frame, (ax, ay+20), (ax, ay-5), COLOR_CRITICAL, 3, tipLength=0.4)
            cv2.putText(frame, "APPR", (ax-22, ay+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CRITICAL, 1)

        # 6. Human warning banner
        if result.is_human:
            cv2.putText(
                frame, "!! HUMAN — ROBOT STOPPED !!",
                (w // 2 - 160, int(h * TRAP_Y_TOP_PCT / 100) - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_HUMAN, 2
            )

        # 7. Top status bar
        cv2.rectangle(frame, (0, 0), (w, 32), COLOR_BLACK, -1)

        if result.obstacle_found():
            if result.is_human:
                status_text  = f"DETECT HUMAN  conf={result.confidence:.0%}"
                status_color = COLOR_HUMAN
            else:
                appr = " ↑APPR" if result.is_approaching else ""
                status_text  = (
                    f"DETECT {result.label.upper()}  "
                    f"{result.approx_dist}  "
                    f"[{result.threat_level}]  "
                    f"{result.confidence:.0%}{appr}"
                )
                status_color = threat_colors.get(result.threat_level, COLOR_CLEAR)
        else:
            status_text  = "AREA CLEAR"
            status_color = COLOR_CLEAR

        cv2.putText(frame, status_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        # 8. Bottom info bar
        cv2.rectangle(frame, (0, h - 22), (w, h), COLOR_BLACK, -1)
        persist_score = self._persistence.score
        cv2.putText(
            frame,
            f"DANGER:{result.danger_score:.2f}  "
            f"AGE:{result.detection_age*1000:.0f}ms  "
            f"PERSIST:{persist_score}/{PERSIST_WINDOW}  "
            f"CONF:{YOLO_CONF:.2f}",
            (8, h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_INFO, 1
        )

        # 9. Environment bar (مكبّرة وأوضح بناءً على طلبك)
        env = self._env_classifier.get_latest()
        env_bar_y1 = h - 50
        env_bar_y2 = h - 22

        cv2.rectangle(frame, (0, env_bar_y1), (w, env_bar_y2), (25, 25, 25), -1)

        if env.label != "Unknown" and env.is_valid():
            env_text  = f"ENV: {env.label}  ({env.confidence:.0%})"
            env_color = COLOR_ENV
        else:
            env_text  = "ENV: Scanning..."
            env_color = (150, 150, 150)

        cv2.putText(frame, env_text, (8, env_bar_y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, env_color, 2)

        # 10. ── Sign Danger Banner ────────────────────────────────────
        #     بانر أحمر في وسط الشاشة لو فيه لوحة خطر متأكدة
        sign = self._sign_detector.get_latest()
        if sign.danger_confirmed:
            banner_y1 = h // 2 - 30
            banner_y2 = h // 2 + 30
            cv2.rectangle(frame, (0, banner_y1), (w, banner_y2), (0, 0, 180), -1)
            cv2.putText(
                frame,
                f"!! DANGER SIGN [{sign.reason}] !!",
                (w // 2 - 170, h // 2 + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2
            )
            # لو في نص OCR كمان، بيظهر تحت البانر
            if sign.text_matched_word:
                cv2.putText(
                    frame,
                    f"TEXT: {sign.text_matched_word}",
                    (w // 2 - 80, h // 2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1
                )

        return frame

    # ──────────────────────────────────────────
    #  SIMULATION LOOP
    # ──────────────────────────────────────────

    def _sim_loop(self):
        t = 0.0
        sim_envs = ["Open Field", "Desert", "Forest", "Urban Road", "Yard"]
        sim_env_idx = 0

        while self._running:
            time.sleep(0.1)
            t += 0.1

            if int(t) % 10 == 0 and t % 10 < 0.15:
                sim_env_idx = (sim_env_idx + 1) % len(sim_envs)
                sim_env = EnvironmentResult(label=sim_envs[sim_env_idx], confidence=0.78)
                with self._env_classifier._lock:
                    self._env_classifier._latest_env = sim_env

            cycle = int(t) % 18
            if   cycle < 3:  pos, conf, label = "NONE",    0.0,  ""
            elif cycle < 6:  pos, conf, label = "FORWARD", 0.87, "chair"
            elif cycle < 10: pos, conf, label = "FORWARD", 0.91, "person"
            elif cycle < 13: pos, conf, label = "FORWARD", 0.78, "bottle"
            else:            pos, conf, label = "NONE",    0.0,  ""

            w, h  = FRAME_WIDTH, FRAME_HEIGHT
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = (25, 25, 25)

            for x in range(0, w, 40):
                cv2.line(frame, (x, 0), (x, h), (40, 40, 40), 1)
            for y in range(0, h, 40):
                cv2.line(frame, (0, y), (w, y), (40, 40, 40), 1)

            is_human   = (label == "person")
            detected   = (pos != "NONE")
            persistent = self._persistence.update(detected)

            result = DetectionResult(
                position     = pos if persistent else "NONE",
                confidence   = conf,
                label        = label,
                frame_width  = w,
                threat_level = "CRITICAL" if is_human else ("HIGH" if detected else "NONE"),
                approx_dist  = "CLOSE" if detected else "FAR",
                persistent   = persistent,
                is_human     = is_human,
                is_approaching = (cycle in (8, 9)),
            )

            if detected:
                cx  = 320
                cy  = 360
                box = (cx - 60, cy - 80, cx + 60, cy + 80)
                result.box      = box
                result.center_x = cx
                with self._yolo_result_lock:
                    self._pending_boxes = [{
                        "label": label, "conf": conf,
                        "x1": box[0], "y1": box[1],
                        "x2": box[2], "y2": box[3],
                        "cx": cx, "cy": cy,
                        "area_ratio": 0.06,
                        "is_human": is_human,
                        "danger_score": 3.0 if is_human else 2.0,
                    }]
            else:
                with self._yolo_result_lock:
                    self._pending_boxes = []

            annotated = self._draw_annotations(frame, result)
            cv2.putText(
                annotated, "[ SIMULATION ]",
                (w // 2 - 70, h // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 180), 1
            )

            with self._lock:
                self._latest_frame    = frame
                self._annotated_frame = annotated
                self._latest_result   = result

            self._push_frame(annotated)


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    )

    vp = VisionProcessor(simulate=True, use_yolo=False, use_env=False)
    vp.start()

    print("Vision Processor v9.2 — streaming to rtsp://127.0.0.1:8554/cam")
    print("Open browser: http://192.168.1.4:8889/cam")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
            result = vp.get_latest()
            env    = vp.get_environment()
            sign   = vp.get_latest_sign()
            if result.obstacle_found():
                print(
                    f"  {result.label:10s} | "
                    f"conf={result.confidence:.0%} | "
                    f"threat={result.threat_level:8s} | "
                    f"env={env.label} | "
                    f"sign={sign.danger_confirmed}"
                )
    except KeyboardInterrupt:
        pass

    vp.stop()