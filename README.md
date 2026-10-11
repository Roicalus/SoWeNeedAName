# Co-pilot — a calm second pair of eyes for any dashcam

**Mode: Startup.** WIUT Hackathon 2026, Computer Vision track (partners: Toyota Royal Motors, BYD Royal Motors, Ducati).

**Live prototype:** https://roicalus.github.io/SoWeNeedAName/ — open it on a laptop or a phone, choose
**Open a video** (any front dashcam clip) or **Use the camera**. The neural network runs inside the browser;
the video never leaves the device. Two views: **Driver** (what a driver gets) and **How the system sees it**.

Co-pilot watches the road and quietly hints the driver *before* things get dangerous. It is an assistant,
not an alarm: most of the time it says nothing.

| Colour | Meaning | Examples |
|---|---|---|
| 🟢 green | attention, worth watching | a car slowly drifting towards our lane, too close to the car ahead, a car that already made one risky manoeuvre is still next to us |
| 🟡 yellow | possible risk, be ready to brake | a car cutting in, the car ahead braking hard, a person stepping onto the road ahead, a car crossing our path |
| 🔴 red | danger, act now | collision about a second away, a crash in view, a wrong-way car |

**Driver reputation.** Co-pilot remembers every car that made a risky manoeuvre (hard braking in front of
us, a cut-in, crossing our path, weaving, hopping from lane to lane) and, while that car is still within
40 m during the next 30 s, says "Careful, aggressive driver on the left": green after one episode, yellow
after several.

People on the pavement or waiting at a crossing, parked cars, and traffic we are not driving into are
ignored on purpose. Hints are rationed (one per level every few seconds, the same phrase much less often),
because frequent false hints teach drivers to tune the assistant out.

## Who it is for
* **Drivers of cars without ADAS** (most of the car park in Uzbekistan): a phone on the windscreen or a
  cheap dashcam with a speaker.
* **Taxi and delivery fleets**: fewer incidents, event clips for coaching.
* **Car makers and dealers** (BYD, Toyota): a software feature for head units of models sold without
  forward-collision warning — the same ONNX models run on an automotive SoC.

## How it works
```
camera / video ─► newest frame ─► YOLO11m (fine-tuned on BDD100K) ─► tracker (ByteTrack-style)
                       │                                                    │
                       ├► lane lines (UFLDv2) ─────────────────────────────►│
                       ├► zebra crossing (own detector, crosswalk.py) ─────►│
                       └► own turning from far-background optical flow ────►│
                                                                            ▼
          behaviour (behavior.py): per object — time to collision from box growth, lateral position
          and speed in lanes, where it will be when we reach it; per camera — horizon and hood
          calibrated from the traffic itself
                                                                            ▼
          rules: cut-in · crossing · hard braking ahead · pedestrian on our path · crash in view ·
          wrong-way car · erratic driver · drift · too close · driver reputation (repeat offenders)
                                                                    ─► advisor: colour, side, phrase, budget
```
Analysis always takes the newest frame and never queues, so it may run less often than the video but
never lags behind the road (~10 analyses per second are enough for the logic).

**Single-camera geometry, no calibration:**
* **Time to collision** from how fast a box grows: `d ln(size)/dt = 1/TTC` — no distance needed.
* **Lateral position in lanes** from the box itself: offset in car widths (or person heights) is
  independent of distance; with lane markings, relative to the lines.
* **Collision course**: an object whose position `u + du·TTC` at the moment we reach it lies inside our
  track is on a collision course (constant bearing). Parked cars do not move sideways in the world, so
  they never qualify, even when they sweep through the frame while we turn.
* **Hard braking ahead**: the growth rate of the car ahead jumps within a second.
* **Crash in view**: a box that "takes off" — jumps up and gets taller while its width stays.
* **Per-camera calibration**: dashcams differ (horizon at 0.3–0.5 of the height, hoods covering up to
  40%); the horizon comes from where far cars stand, the hood from where cars are never seen.

## Results (honest numbers)
Thresholds were tuned on the *tune* half of the data; everything below is on the *test* half, which the
system never saw while tuning. A hint counts as in advance if it comes between 3 s before the labelled
start of the danger and the crash. In normal driving every yellow or red hint counts as false.

RESULTS_TABLE

**Detector** (BDD100K official val, 10 000 frames never used in training, AP50):

DETECTOR_TABLE

**Speed** (Intel i5-10400F + RTX 4070 Ti, ONNX Runtime + DirectML): SPEED_TEXT
In the browser the lighter YOLO11n runs on WebGPU/WASM; phones get a 320 px input.

**Known failure cases**
* A car spinning (loss of control) is hinted only when it brakes hard in front of us or ends up in our
  path — a dedicated skid rule produced false alarms while we turn and is switched off.
* Strong fisheye distortion: lateral speed is less accurate, cut-ins are hinted later.
* Motorbikes weaving through dense traffic: small, jumpy boxes; many are hinted only at close range.
* Own speed from video is rough (the hood and reflections pull it down); decisions that need it
  ("we stand still", "that car is parked") use it only when the speed comes from the vehicle (CAN).
* Night without street lights: beyond ~30 m the detector sees headlights rather than cars.

## Data and licences
| Source | Used for | Licence |
|---|---|---|
| [BDD100K](https://github.com/bdd100k/bdd100k) images + boxes (HF copies: `Xing1210/bdd100k`, `dronefreak/BDD100K`, `dronefreak/BDD100K-Period-Classification`) | detector fine-tuning (69 863 frames, 40 % night), held-out AP (10 000) | BAIR BDD100K licence: research/education; commercial use needs a licence from UC Berkeley |
| BDD100K videos (HF `linxxx3/bdd100k_videos`, byte ranges of the original zip) | 150 videos of normal driving: false-alarm measurement | same |
| [DADA-2000](https://github.com/JWFangit/LOTVS-DADA) (frames HF `yumuxie/dada`, labels HF `DBatinic/DADA2000`) | 1 714 crash clips with danger-start / crash labels: tuning and the held-out test | research use, citation requested |
| [YOLO11](https://github.com/ultralytics/ultralytics) COCO weights | starting point of the detector | AGPL-3.0 (this repository is AGPL-3.0 too) |
| [UFLDv2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2), CULane ResNet-18, ONNX from [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo) | lane lines | MIT |
| Wikimedia Commons: [Kraków city drive](https://commons.wikimedia.org/wiki/File:City_Driving_4K-_Krak%C3%B3w_Poland_2024.webm) (CC BY 3.0), [German city](https://commons.wikimedia.org/wiki/File:Dashcam_Recording_(urban).ogv) (CC0), [TeslaCam crash](https://commons.wikimedia.org/wiki/File:INSANE_Car_Crash_caught_on_TeslaCam-_Kia_Soul_flying_through_air!.webm) (PD) | samples and checks | as stated |
| Public social-media driving clips (7) | internal checks only (`tools/prepare_reels.py`), **not published** — they show plates and people who did not consent | not redistributed |

No personal data is collected: frames are processed and dropped on the device.

## Run it
**Web (what the jury tests):** any static server over `web/`, e.g. `python -m http.server 8099 -d web`, then
http://localhost:8099 (the camera needs HTTPS or localhost). GitHub Pages deploys `web/` on every push to
`main` (`.github/workflows/pages.yml`).

**Desktop app and evaluation (Python 3.10–3.12):**
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt            # onnxruntime-directml / onnxruntime-gpu for a GPU
python main.py data/videos/krakow_peds.mp4 # window with voice and event recording
python main.py 0                           # webcam
python main.py clip.mp4 --can-speed 50     # speed from the vehicle bus, km/h
python test_behavior.py                    # 27 logic tests on synthetic tracks
python debug_video.py clip.mp4 --demo      # render a demo video with sound (driver view)
python debug_video.py clip.mp4             # full debug view: tracks, plots, raw / confirmed alerts
```
The lane model (825 MB) is not in git: see the top of `lanes.py`. Without it the own lane is a trapezoid.

**Reproduce the numbers:**
```bash
python tools/get_dada.py --all                     # 1 714 DADA-2000 crash clips + labels
python tools/get_bdd_videos.py --n 150             # BDD100K normal driving
python tools/train_detector.py download && python tools/train_detector.py prep --train-n 0
python tools/train_detector.py train --arch yolo11m --epochs 8 --device 0
python tools/train_detector.py eval --arch yolo11m && python tools/train_detector.py export --arch yolo11m
python tools/cache_tracks.py data/dada/videos data/bdd --workers 3   # detector + lanes + crosswalk + flow, once
python tools/replay.py --split test                # metrics in seconds, from the cache
python tools/tune.py --iters 80                    # automatic threshold search on the tune half
```

## Files
| File | What it does |
|---|---|
| `web/` | the public prototype: `js/detector.js` (onnxruntime-web), `tracker.js`, `crosswalk.js`, `behavior.js`, `advisor.js`, `sound.js`, `app.js`; `index.html` the site |
| `config.py` | every threshold (exported to `web/js/config.json` by `tools/export_web_config.py`) |
| `detector.py`, `tracker.py` | YOLO (ONNX Runtime) + ByteTrack-style tracker |
| `lanes.py`, `crosswalk.py`, `egomotion.py` | lane lines, zebra crossings, own turning / speed from optical flow |
| `behavior.py` | geometry and rules: threats per object, camera calibration, confirmation with hysteresis |
| `advisor.py`, `voice.py`, `beeps.py`, `ui.py` | colour, side, phrase and budget; sound; screen |
| `main.py`, `recorder.py` | desktop app with event recording |
| `risk.py`, `tools/train_risk.py` | experimental learned filter (see "What did not work") |
| `debug_video.py`, `evaluate.py`, `tools/replay.py`, `tools/tune.py`, `tools/snapshots.py` | evaluation and debugging |
| `tools/get_*.py`, `tools/prepare_reels.py`, `tools/train_detector.py` | data and detector training |

## What did not work (and why it is not in the product)
* **A learned filter of rule alerts** (LightGBM on per-threat features, `tools/train_risk.py`): the crash
  clips come from China (DADA) and normal driving from the US (BDD), so a model learns *which dataset* a
  clip is from (track age, camera geometry) rather than danger. With dataset-revealing features removed it
  reached AUC ≈ 0.72 on held-out clips — not enough to ship. The fix is crash and normal clips from the
  same cameras (Nexar collision dataset, access pending).
* **Skid detection** from box proportions: our own turning changes how other cars look; switched off.
* **Own speed from optical flow**: too rough for decisions; kept as a feature only.

## Prior work (declared)
* The team's elimination-round project ([Roicalus/wiut-traffic-events](https://github.com/Roicalus/wiut-traffic-events))
  — experience with YOLO + ONNX Runtime and tracking; the site design follows the team site
  [wiut-traffic-site](https://github.com/Roicalus/wiut-traffic-site).
* Everything in this repository was built during the final (10–11 October 2026); the git history shows it.
* Coding assistants were used (allowed by the rules); every number in this README comes from the scripts above.
