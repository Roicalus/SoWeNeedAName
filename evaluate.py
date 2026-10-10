"""Прогон по папке роликов без окна и звука: все события в CSV и метрики для питча.

    python evaluate.py data/videos
    python evaluate.py data/videos --labels data/labels.csv --tags bdd100k_labels_val.json

labels.csv: file,start,end,type — опасные моменты (секунды). type: brake, closing, weave, ped, close, any.
Если передан --labels, ролики без строк в нём считаются «чистыми» (любое предупреждение = ложное).
--tags: CSV (file,weather,timeofday,scene) или JSON с разметкой BDD100K (атрибуты кадров).
"""
import argparse
import csv
import glob
import json
import os
import time
from collections import defaultdict

import cv2

import config as C
from behavior import BehaviorEngine
from detector import Detector
import ui

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".ogv")
# какие предупреждения засчитываются за какой тип размеченной опасности
MATCH = {
    "brake": {"ttc", "closing", "brakelight"},
    "closing": {"ttc", "closing"},
    "weave": {"weave"},
    "ped": {"ped"},
    "crash": {"crash", "ttc", "closing"},
    "close": {"close", "closing", "ttc"},
    "any": set(C.ALERTS),
}


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def run_video(path, det, max_sec=None, stride=1, snap_dir=None):
    """stride — анализировать каждый N-й кадр (как на реальном устройстве, где анализ реже кадров)."""
    det.reset()
    engine = BehaviorEngine()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        fps = 30.0
    events, last_active, i, n = [], {}, 0, 0
    tic = time.perf_counter()
    while True:
        for _ in range(stride - 1):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        i += stride
        n += 1
        t = i / fps
        if max_sec and t > max_sec:
            break
        res = engine.update(t, frame.shape[1], frame.shape[0], det(frame), frame)
        new = False
        for a in res.alerts:
            if t - last_active.get(a.kind, -1e9) > C.EVENT_GAP:   # новое событие, а не продолжение
                events.append({"file": stem(path), "time": round(t, 2), "kind": a.kind, "level": a.level,
                               "ttc": round(res.ttc, 2) if res.ttc else ""})
                new = True
            last_active[a.kind] = t
        if new and snap_dir:
            os.makedirs(snap_dir, exist_ok=True)
            kinds = "+".join(sorted(a.kind for a in res.alerts))
            cv2.imwrite(os.path.join(snap_dir, f"{stem(path)}_{t:07.2f}_{kinds}.jpg"), ui.draw(frame, res))
    cap.release()
    return events, i / fps, n / max(time.perf_counter() - tic, 1e-6)


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{"file": stem(r["file"]), "start": float(r["start"]), "end": float(r["end"]),
                 "type": r.get("type", "any").strip() or "any"} for r in csv.DictReader(f)]


def load_tags(path):
    if path.endswith(".json"):        # разметка BDD100K: [{"name": ..., "attributes": {...}}, ...]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {stem(d["name"]): d.get("attributes", {}) for d in data}
    with open(path, newline="", encoding="utf-8") as f:
        return {stem(r["file"]): r for r in csv.DictReader(f)}


def score(files, events, labels, durations):
    """Метрики по набору роликов."""
    files = set(files)
    events = [e for e in events if e["file"] in files]
    labels = [L for L in labels if L["file"] in files]
    minutes = sum(durations[f] for f in files) / 60
    out = {"videos": len(files), "minutes": minutes, "events": len(events),
           "events_per_min": len(events) / minutes if minutes else 0}
    hits, leads = 0, []
    for L in labels:
        kinds = MATCH.get(L["type"], MATCH["any"])
        times = [e["time"] for e in events if e["file"] == L["file"] and e["kind"] in kinds
                 and L["start"] - C.EVAL_EARLY <= e["time"] <= L["end"]]
        if times:
            hits += 1
            leads.append(L["start"] - min(times))
    # ложное — предупреждение вне любого опасного интервала
    false = [e for e in events if not any(L["file"] == e["file"] and L["start"] - C.EVAL_EARLY <= e["time"] <= L["end"]
                                          for L in labels)]
    out.update(dangers=len(labels), caught=hits, recall=hits / len(labels) if labels else None,
               mean_lead=sum(leads) / len(leads) if leads else None,
               false=len(false), false_per_min=len(false) / minutes if minutes else 0)
    return out


def fmt(name, m, with_labels):
    line = f"{name:<22} роликов {m['videos']:>3}  мин {m['minutes']:>5.1f}  предупреждений/мин {m['events_per_min']:>5.2f}"
    if with_labels:
        recall = f"{m['caught']}/{m['dangers']}" if m["dangers"] else "-"
        lead = f"{m['mean_lead']:+.1f} с" if m["mean_lead"] is not None else "-"
        line += f"  поймано {recall:>6}  заранее {lead:>7}  ложных/мин {m['false_per_min']:.2f}"
    return line


def main():
    ap = argparse.ArgumentParser(description="Оценка «Второго пилота» на папке роликов")
    ap.add_argument("videos", help="папка с роликами или один файл")
    ap.add_argument("--labels", help="CSV с опасными моментами")
    ap.add_argument("--tags", help="условия съёмки: CSV или JSON-разметка BDD100K")
    ap.add_argument("--out", default="eval_events.csv")
    ap.add_argument("--limit", type=int, help="взять только первые N роликов")
    ap.add_argument("--max-sec", type=float, help="обрабатывать только первые N секунд ролика")
    ap.add_argument("--stride", type=int, default=3, help="анализировать каждый N-й кадр (3 = 10 раз/с при 30 fps)")
    ap.add_argument("--snapshots", help="папка для снимков кадров в момент событий")
    ap.add_argument("--model", default=C.MODEL, help="путь к .onnx модели")
    args = ap.parse_args()

    if os.path.isfile(args.videos):
        paths = [args.videos]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(args.videos, "*")) if p.lower().endswith(VIDEO_EXT))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"В папке {args.videos} нет роликов")

    det = Detector(args.model)
    events, durations, speeds = [], {}, []
    for n, p in enumerate(paths, 1):
        ev, dur, speed = run_video(p, det, args.max_sec, args.stride, args.snapshots)
        events += ev
        durations[stem(p)] = dur
        speeds.append(speed)
        print(f"[{n}/{len(paths)}] {os.path.basename(p)}: {len(ev)} событий, {speed:.0f} кадров/с")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "time", "kind", "level", "ttc"])
        w.writeheader()
        w.writerows(events)
    print(f"\nСобытия сохранены в {args.out}")

    labels = load_labels(args.labels) if args.labels else []
    with_labels = bool(args.labels)
    files = list(durations)
    print(f"\nСкорость анализа: {sum(speeds) / len(speeds):.1f} кадров/с (анализируется каждый {args.stride}-й кадр)")
    print("По типам:", ", ".join(f"{k}: {sum(e['kind'] == k for e in events)}" for k in C.ALERTS))
    print()
    print(fmt("ВСЕГО", score(files, events, labels, durations), with_labels))

    if args.tags:
        tags = load_tags(args.tags)
        for attr in ("timeofday", "weather", "scene"):
            groups = defaultdict(list)
            for f in files:
                groups[str(tags.get(f, {}).get(attr, "unknown"))].append(f)
            if len(groups) > 1 or "unknown" not in groups:
                print(f"\n--- {attr} ---")
                for val, fs in sorted(groups.items()):
                    print(fmt(val, score(fs, events, labels, durations), with_labels))


if __name__ == "__main__":
    main()
