"""All tunable parameters in one place. Thresholds are tuned on real footage (tools/replay.py)."""

# ---------- model ----------
# yolo11m_bdd: YOLO11m fine-tuned on BDD100K (day + night dashcam frames, tools/train_detector.py).
# Fallbacks: yolo11s_fp32.onnx (COCO), yolo11n.onnx (weak laptops).
MODEL = "models/yolo11m_bdd.onnx"
MODEL_FALLBACK = "models/yolo11s_fp32.onnx"
ORT_ACCEL = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
NMS_IOU = 0.45
ORT_THREADS = 4                # CPU threads for the network; 0 = auto
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0
BICYCLE_CLASS = 1
TRAFFIC_LIGHT_CLASS = 9

# ---------- own lane without lane markings: trapezoid in front of the car (frame fractions) ----------
CORRIDOR_TOP_Y = 0.45         # top edge of the trapezoid (roughly the horizon)
CORRIDOR_TOP_HALF = 0.08      # half-width at the top
CORRIDOR_BOTTOM_HALF = 0.30   # half-width at the bottom (narrower for wide-angle cameras)
CORRIDOR_CENTER_X = 0.5       # shift if the camera is not centred
LEAD_MAX_OFFSET = 0.7         # car on our path: lateral offset at most 0.7 of its own width
LEAD_MAX_ASPECT = 1.9         # box 1.9+ times wider than tall: car seen from the side, not "ahead"
SIDE_MARGIN = 0.04            # lane-change / crossing rules need the whole car in view (not cut by the frame edge)
EDGE_MARGIN = 0.01            # boxes touching the left/right frame edge are not "the car ahead"
HOOD_Y = 0.85                 # boxes centred below this line are our own hood (default; calibrated per camera)
CALIB_WINDOW = 20.0           # camera auto-calibration (horizon, hood) from car boxes of the last N seconds
CALIB_MIN = 15                # ...at least this many boxes
CALIB_MIN_SPAN = 5.0          # ...spread over at least this many seconds (for the hood)
CALIB_FAR_WIDTH = 0.05        # boxes narrower than this are far cars standing near the horizon
CALIB_TOP_RANGE = (0.25, 0.6)

# ---------- camera (distance estimate) ----------
CAMERA_HFOV_DEG = 70          # horizontal field of view; set for your dashcam
VEHICLE_WIDTH_M = 1.8         # average car width
PERSON_HEIGHT_M = 1.7         # average person height
CAR_HEIGHT_M = 1.5            # car height: distance to a car seen from the side
TRUCK_HEIGHT_M = 3.0          # truck, bus
LEAD_MAX_CLOSING = 12.0       # closing faster than this, m/s (~43 km/h): most likely an oncoming car

# ---------- tracker ----------
TRACK_HIGH_CONF = 0.40        # confident detections create and extend tracks
TRACK_LOW_CONF = 0.15         # weak detections only extend existing tracks
TRACK_IOU = 0.30              # minimum overlap to match
TRACK_MAX_MISSED = 15         # frames a track survives without detections
TRACK_MIN_HITS = 2            # show a track after N confirmations

# ---------- track history ----------
HISTORY_SEC = 5.5             # seconds of history kept per object
TRACK_TTL = 2.0               # forget a track unseen for this long

# ---------- time to collision (TTC) ----------
TTC_WINDOW = 0.6              # window for the box growth rate, s
TTC_MIN_SPAN = 0.4            # minimum history for an estimate, s
TTC_MIN_RATE = 0.05           # growth slower than 5%/s: not closing in
TTC_MIN_WIDTH = 0.06          # ignore far cars (fraction of frame width)
TTC_WARNING = 2.5             # fast closing, s
TTC_CRITICAL = 1.5            # critical closing, s
MIN_CLOSING = 1.5             # yellow needs the gap to shrink at least this fast, m/s (creeping in a jam is not a risk)
BRAKE_MIN_CLOSING = 3.0       # red "Brake!" needs at least this closing speed, m/s (~11 km/h)
PED_RED_DIST = 12.0           # red for a pedestrian only on our path and closer than this, m
PED_COURSE_DIST = 15.0        # pedestrians on a collision course are judged only closer than this, m
PED_MIN_CLOSING = 2.5         # a pedestrian alert needs us to drive towards them at least this fast, m/s
CRITICAL_MIN_WIDTH = 0.12     # "Brake!" only when the car is already close
TTC_CERTAIN = 1.0             # TTC below this to a close car: red regardless of the risk model
CERTAIN_HOLD = 0.3            # ...and the critical TTC has held this long (s)
CLOSE_WIDTH = 0.40            # box wider than 40% of the frame: too close
CLOSE_MAX_TTC = 6.0           # ...and closing in (TTC < 6 s); standing bumper to bumper in a jam is fine

# ---------- hard braking of the car ahead ----------
# growth rate g = d ln(w)/dt = closing speed / distance. A car that starts braking hard makes g jump
# from ~0 to a large value within a second; steady following keeps g ~ 0.
BRAKE_RATE = 0.20             # g at least this (1/s)
BRAKE_JERK = 0.30             # and g rose by at least this within BRAKE_JERK_SEC
BRAKE_JERK_SEC = 1.0
BRAKE_MAX_DIST = 25.0         # only for cars closer than this, m
BRAKE_MAX_TTC = 3.0             # ...and only if it makes us close in (TTC below this)
BRAKE_CRIT_TTC = 2.0          # hard braking with TTC below this is red

# ---------- weaving of the car ahead ----------
WEAVE = False                 # off: our own steering and road curves make parked cars "weave" in the frame
WEAVE_WINDOW = 5.0            # analysis window, s
WEAVE_MIN_SPAN = 4.0          # minimum history, s
WEAVE_MIN_WIDTH = 0.05        # skip far cars
WEAVE_REL_STD = 0.15          # oscillation amplitude in car widths (~0.25 m)
WEAVE_MIN_SPEED = 0.15        # lateral speed below this (widths/s) is noise
WEAVE_MIN_REVERSALS = 2       # direction changes (lane change - 0, overtake and back - 1)

# ---------- crash of any car in view ----------
CRASH_WINDOW = 0.4            # compare the box over this many seconds
CRASH_RISE = 0.7              # box rose faster than 0.6 of its height per second (ego pitch removed)
CRASH_ASPECT = 0.5            # and became taller relative to width (ln(h/w) grows > 0.5 per second)
CRASH_MAX_SCALE = 0.5         # while the width barely changed (not driving away or approaching)
CRASH_BOTTOM_RISE = 0.25      # and the box bottom left the road too (box heights per second); a box that only
                              # stretches upward is the detector merging shadow or glare, not a lifted car
CRASH_MIN_WIDTH = 0.06        # small far boxes are too noisy
CRASH_HOLD = 3.0              # keep the alarm this long after crash signs
CRASH_MIN_AGE = 1.0           # only tracks seen at least this long (s) can show crash signs
CRASH_HITS = 2                # crash signs needed in the last 3 analyses of the same car
CRASH_CRIT_WIDTH = 0.08       # red only for a crash close enough (box wider than this share of the frame), else yellow

# ---------- object position relative to lanes (behavior.py) ----------
LANE_WIDTH_M = 3.5            # lane width: converts metres to "lanes" without markings
IN_LANE = 0.4                 # |u| <= 0.4 lanes: object is in our lane (0 = centre, 1 = centre of the next one)
IMPACT_U = 0.55               # |u| <= this: the object overlaps our car's track (car widths 1.8 m: (1.8 + 1.8) / 2 / 3.5 ~ 0.5)
PATH_CORRIDOR_FRAC = 0.55     # without markings, a box centre must lie within this share of the corridor half-width
IMPACT_U_CAR = 0.45           # a car is in our path if its centre is within ~1.6 m of our axis (parked cars we pass: 0.5+)
COURSE_MAX_DIST = 30.0        # collision-course rule looks at objects closer than this, m
COURSE_TTC = 2.5              # collision course: we reach the object within this many seconds
COURSE_MIN_TOWARD = 0.3       # a car must move towards our path at least this fast (lanes/s) - parked cars do not
KIN_WINDOW = 0.6              # window for lateral speed, s
KIN_MIN_SPAN = 0.3            # minimum history for it, s
YAW_MIN_TRACKS = 3            # own turning is estimated from the shift of at least 3 boxes
PREDICT_SEC = 1.0             # on-screen arrow: where the object will be in a second

# ---------- cut-in and crossing ----------
CUTIN_MAX_U = 1.6             # look at cars up to the neighbouring lane
CUTIN_MIN_SPEED = 0.8         # moving towards our lane faster than 0.6 lanes/s (~2 m/s)
CUTIN_HORIZON = 1.0           # and will be in it within 1 s
CUTIN_MAX_DIST = 20.0         # and closer than 25 m
CROSS_MIN_ASPECT = 1.6        # box 1.6+ times wider than tall: car seen from the side - "crossing", not a lane change
CUTIN_TTC = 2.0               # cut-in/crossing is dangerous if we close in on it (TTC <= 3 s) or it is within CRIT_DIST
CUTIN_CRIT_TTC = 1.0          # red only if the collision is about a second away
CUTIN_CRIT_DIST = 10.0        # ...and closer than this, m
CUTIN_MIN_AGE = 0.6           # the track must exist this long (s): new boxes have jumpy lateral speed
STATIC_TOL = 0.35             # closing speed within 35% of own speed: the object stands still (parked)
STATIC_MIN_EGO = 3.0          # only when we drive faster than this, m/s
CRIT_DIST = 10.0              # closer than 10 m: critical
WRONGWAY_TTC = 4.0            # oncoming car in our lane (seen by lane markings) with TTC below 4 s
WRONGWAY_U = 0.3              # ...steadily within this of our lane centre for 0.5 s
WRONGWAY_CLOSING = 16.0       # ...and closing faster than this, m/s (sum of two speeds)
YAW_SLACK = 5.0               # extra lanes/s of sideways speed required per frame-width/s of own turning
YAW_TURN = 0.12               # own turning faster than this (frame widths/s, ~8 deg/s at 70 deg FOV): sharp turn

# ---------- skid / loss of control of a car ahead ----------
SKID = False                  # off: own turning also changes how other cars look; not validated
SKID_MAX_U = 1.5              # cars up to the neighbouring lane
SKID_MAX_DIST = 50.0
SKID_ASPECT_RATE = 0.6        # |d ln(w/h)/dt| above this: the car is turning its body fast
SKID_MIN_DU = 0.15            # ...while moving sideways, lanes/s
SKID_CRIT_DIST = 25.0         # red closer than this (or when we close in on it)

# ---------- dense slow traffic (a jam) ----------
DENSE_CARS = 3                # this many cars closer than DENSE_DIST: we are in a jam
DENSE_DIST = 20.0

# ---------- slow drift of a neighbour towards our lane (attention, green) ----------
DRIFT_MIN_SPEED = 0.35        # lanes/s
DRIFT_HORIZON = 2.5           # reaches our lane within this many seconds
DRIFT_MAX_DIST = 35.0

# ---------- pedestrians, bicycles, motorcycles (vulnerable road users, VRU) ----------
VRU_NEAR = 25.0               # on our path closer than this...
PED_TTC = 3.5                 # ...and we drive towards them (time to reach them below this, s): warn in advance
VRU_NEAR_ALERT = False        # people at the road edge / on the pavement: no alert (only shown in the system view)
PED_XWALK_ALERT = False       # people waiting at a crosswalk: no alert
VRU_DIST = 20.0               # heading to our lane: look up to 25 m
VRU_MIN_SPEED = 0.5           # lanes/s (~1.7 m/s: fast walk, run, bicycle)
VRU_HORIZON = 1.0             # will be in our lane within 1.5 s
VRU_STEP_U = 1.6              # "stepping into the lane" only from at most this far (lanes)
VRU_ATTENTION_DIST = 30.0     # VRU near the road edge ahead (|u| < VRU_EDGE_U) closer than this: attention
VRU_EDGE_U = 1.3
VRU_SUDDEN_AGE = 1.0          # a VRU track younger than this...
VRU_SUDDEN_DIST = 15.0        # ...that appears this close and near our path: "sudden appearance"
VRU_SUDDEN_U = 0.9

# ---------- reputation: cars that already made risky manoeuvres ----------
MARK_WINDOW = 30.0            # a car is remembered for its risky manoeuvres this long, s
MARK_GAP = 2.0                # the same manoeuvre of the same car counts again after this gap, s
MARK_DIST = 40.0              # warn about a marked car while it is closer than this, m
MARK_YELLOW = 3.0             # this many episodes (lane change = LANECHANGE_WEIGHT) make it yellow, else green
LANECHANGE_WEIGHT = 0.34      # three lane changes within MARK_WINDOW = one risky manoeuvre
LANE_DEEP = 0.25              # a car is "in" a lane when it is within this of the lane centre (boundary jitter is not a change)
MARK_COUNTS = ("hardbrake", "cutin", "crossing", "weave", "lanechange")   # drift alone does not mark a car
LANE_STAY = 1.0               # a lane change counts if the car stayed in its previous lane at least this long, s

# ---------- crosswalk (crosswalk.py) ----------
CROSSWALK = True
CROSSWALK_EVERY = 2           # look for zebra stripes every N-th analysis
CROSSWALK_DIST = 40.0         # pedestrian on/near a crosswalk closer than this: attention

# ---------- ego motion (egomotion.py) ----------
EGO_STOPPED = 0.15            # normalised flow below this: we are (almost) standing still

# ---------- brake lights (experimental) ----------
BRAKE_LIGHTS = False
BRAKE_RED_MIN = 0.03          # minimum share of red pixels
BRAKE_RED_JUMP = 2.0          # how many times the share must exceed the usual one

# ---------- alert levels ----------
# 1 = GREEN  - pay attention (nothing dangerous yet, but worth watching)
# 2 = YELLOW - possible risk (be ready to brake)
# 3 = RED    - dangerous manoeuvre or crash: act now
LEVEL_NAMES = {0: "clear", 1: "attention", 2: "risk", 3: "danger"}

# kind -> (base level, screen text, voice phrase). Rules may raise the level of a particular threat.
LANG = "en"
ALERTS = {
    "ttc":        (3, "BRAKE!",            "Brake!"),
    "crash":      (3, "CRASH AHEAD",       "Crash ahead!"),
    "wrongway":   (3, "WRONG-WAY CAR",     "Oncoming car in your lane!"),
    "hardbrake":  (2, "HARD BRAKING",      "Car ahead is braking hard"),
    "cutin":      (2, "CUT-IN",            "Car cutting in"),
    "crossing":   (2, "CROSSING",          "Car crossing your path"),
    "ped":        (2, "PEDESTRIAN",        "Pedestrian on the road"),
    "danger":     (2, "DANGER",            "Danger ahead"),
    "closing":    (2, "CLOSING IN",        "Traffic slowing ahead"),
    "weave":      (2, "ERRATIC DRIVER",    "Erratic driver ahead, keep your distance"),
    "skid":       (2, "CAR SKIDDING",      "Car ahead is out of control!"),
    "aggressive": (1, "AGGRESSIVE DRIVER", "Careful, aggressive driver"),
    "ped_xwalk":  (1, "PED AT CROSSWALK",  "Pedestrian at the crossing"),
    "vru_near":   (1, "PEDESTRIAN NEAR",   "Watch the pedestrian"),
    "drift":      (1, "CAR DRIFTING",      "Car drifting into your lane"),
    "close":      (1, "TOO CLOSE",         "Keep your distance"),
    "brakelight": (1, "BRAKE LIGHTS",      "Brake lights ahead"),
}
# how long a condition must hold without interruption to fire (noise protection)
CONFIRM_SEC = {"danger": 0.2, "ttc": 0.1, "crash": 0.0, "wrongway": 0.1, "hardbrake": 0.1, "cutin": 0.2,
               "crossing": 0.4, "closing": 0.3, "weave": 0.0, "skid": 0.15, "aggressive": 0.3, "ped": 0.2, "ped_xwalk": 0.3, "vru_near": 0.5,
               "drift": 0.3, "brakelight": 0.2, "close": 0.5}
RAW_GAP = 0.3                 # a dropout shorter than this does not reset confirmation
ALERT_HOLD = 0.4              # an alert stays this long after the condition disappears (no flicker)
VOICE_COOLDOWN = {"danger": 4, "ttc": 2.5, "crash": 6, "wrongway": 4, "hardbrake": 4, "cutin": 4, "crossing": 4,
                  "closing": 6, "weave": 10, "skid": 4, "aggressive": 15, "ped": 4, "ped_xwalk": 15, "vru_near": 20, "drift": 10,
                  "brakelight": 8, "close": 15}
VOICE_MAX_DELAY = 1.5         # drop a phrase that waited longer: it is no longer relevant
BEEP = True                   # short chime before the phrase (sounds/beep.wav, beep_crit.wav)
BEEP_CRIT_LEVEL = 3           # level from which the "critical" chime plays
# Advisor (advisor.py): budget for non-urgent phrases so the driver does not learn to ignore them
HINT_GAP = {1: 20.0, 2: 3.0}  # min seconds between two spoken hints of this level (red has no budget)
SPEAK_GREEN = True            # speak green "attention" hints (else glow only)
SILENT_KINDS = ("drift",)     # shown (glow, box) but never spoken

# ---------- event recording ----------
RECORD_LEVEL = 3              # record a clip from this level
RECORD_PRE = 5.0              # seconds before the event
RECORD_POST = 5.0             # seconds after the event
EVENTS_DIR = "events"

# ---------- evaluation (evaluate.py, tools/replay.py) ----------
EVENT_GAP = 1.5               # pause after which the same alert counts as a new event, s
EVAL_EARLY = 3.0              # an alert up to this many seconds before the labelled danger counts

# ---------- lanes (lanes.py) ----------
LANE_MODEL = "models/ufldv2_culane_res18_320x1600.onnx"
LANE_ROWS = [round(0.45 + 0.025 * k, 3) for k in range(23)]   # rows where lines are stored (fraction of height)
LANE_EVERY = 1                # detect lanes on every N-th analysis (2-3 on a weak laptop)
LANE_MIN_SHARE = 0.25         # a line counts if found on at least a quarter of the rows
LANE_HOLD = 1.0               # seconds to remember a line that is no longer seen
LANE_SMOOTH = 0.6             # temporal smoothing (share of the old value)
LANE_FIT_RMS = 0.012          # scatter of line points around a smooth curve (frame widths); larger - drop the line
LANE_EXTEND = 0.05            # extend the line upward beyond the found part
LANE_MIN_WIDTH = 0.05         # own lane width in the frame (fraction), otherwise markings are wrong
LANE_MAX_WIDTH = 0.9

# ---------- learned risk model (risk.py, tools/train_risk.py) ----------
RISK_MODEL = "models/risk.json"
RISK_CONFIRM = 0.2            # probability above threshold must hold this long, s
