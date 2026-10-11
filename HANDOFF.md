# Handoff — Co-pilot (state of 11 October 2026, final day of WIUT Hackathon 2026)

Everything needed to continue in a new chat or on another PC. Details of the code: `README.md`.

## The product
A driver assistant for any dashcam / phone / head unit: it watches the road and *quietly hints* before
things get dangerous. Three levels: green (attention), yellow (possible risk), red (danger). It is a
helper, not an alarm: people on the pavement, parked cars and traffic we are not driving into are ignored;
hints are rationed. Mode for the jury: **Startup** (drivers without ADAS, taxi/delivery fleets; OEM head
units of BYD / Toyota as a channel).

## How the user wants it
* Communication in Russian; code, comments, README, site and voice in English.
* Honest numbers only; known failures stated.
* Demo videos must stay as rendered by `debug_video.py --demo`; the site offers two views: Driver /
  How the system sees it. Site style follows https://github.com/Roicalus/wiut-traffic-site.
* Commits go to https://github.com/Roicalus/SoWeNeedAName (push via GitHub Desktop — the CLI has no
  credentials on this PC).

## This PC (Windows 11, i5-10400F, 32 GB, RTX 4070 Ti 12 GB)
* Repo: `E:\games\SoWeNeedAName\SoWeNeedAName` (E: is a **hard disk** — slow for 70k small files).
* Python venv: `C:\Users\user\Desktop\SoWeNeedAName-main\.venv` (torch cu128, ultralytics, onnxruntime-directml,
  lightgbm, yt-dlp). Run tools with that python.
* Git: `C:\Users\user\AppData\Local\GitHubDesktop\app-3.6.6\resources\app\git\cmd\git.exe`.
* Detector dataset (YOLO format, paths rewritten): `D:\SoWeNeedAName_data\bdd_yolo` (SSD).
* Long jobs must be started detached (`Invoke-CimMethod Win32_Process Create`), otherwise they die with the shell.

## Data on disk (not in git)
| Path | What |
|---|---|
| `data/dada/videos`, `labels.csv` | 1 714 DADA-2000 crash clips (10 fps, tune/test split) |
| `data/bdd` | 150 BDD100K videos of normal driving |
| `data/bdd_det` | BDD100K images zip + metadata (detector training) |
| `data/highway` | 3 long highway drives from Wikimedia (CC BY 3.0): Tehran, VW POV, Austria→Slovenia (first 300 MB) |
| `data/videos` | Kraków (CC BY), German city (CC0), TeslaCam crash (PD), Tashkent reel |
| `data/reels` | 7 Instagram clips + 20 labelled crops — internal only, never publish (faces/plates) |
| `cache*/` | detector+lanes+crosswalk+flow caches for `tools/replay.py` |
| `models/ufldv2_culane_res18_320x1600.onnx` | lane model (825 MB) |
| `runs/bdd/yolo11m_bdd*` | detector training runs |

## What was learned
* A learned filter of alerts trained on DADA (China) vs BDD (US) learns the dataset, not danger — not shipped.
* Video-only own speed is unreliable (hood, reflections); yaw from far background flow is usable.
* Camera mounting varies a lot: horizon/hood are auto-calibrated per camera (behavior._calibrate).
* Floating-point comparison in alert confirmation silently dropped short real alerts — fixed with an epsilon.
* Parked cars while turning were the main false-alarm source — yaw-dependent thresholds + world-motion checks.

## Next steps
1. Nexar collision dataset (needs the user to accept terms on Hugging Face + token): crash and normal clips
   from the same cameras — the right data for a learned false-alarm filter.
2. CAN speed/yaw input on a real head unit (`main.py --can-speed`).
3. Pilot with a taxi fleet: event clips + driver reputation stats.
