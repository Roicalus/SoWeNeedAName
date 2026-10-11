"""Export thresholds and the risk model for the web version, so the browser uses exactly the same numbers.

    python tools/export_web_config.py      ->  web/js/config.json, web/js/risk.json
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import advisor  # noqa: E402
import config as C  # noqa: E402
import crosswalk  # noqa: E402

WEB_MODEL = "yolo11m_bdd"   # web/models/<name>_<size>.onnx on laptops (ours, fine-tuned on BDD100K)
WEB_MODEL_PHONE = "yolo11n" # phones: the small COCO model keeps real time


def plain(v):
    if isinstance(v, (set, frozenset, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [plain(x) for x in v]
    return v


def main():
    data = lambda mod: {k: plain(getattr(mod, k)) for k in dir(mod)
                        if k.isupper() and len(k) > 1 and not callable(getattr(mod, k)) and type(getattr(mod, k)).__name__ != "module"}
    out = data(C)
    out["ADVISOR"] = data(advisor)
    out["XWALK_HOLD"] = crosswalk.HOLD
    out["WEB_MODEL"] = WEB_MODEL
    out["WEB_MODEL_PHONE"] = WEB_MODEL_PHONE
    web = ROOT / "web" / "js"
    web.mkdir(parents=True, exist_ok=True)
    (web / "config.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(web / "config.json")

    path = ROOT / C.RISK_MODEL
    if not path.exists():
        (web / "risk.json").unlink(missing_ok=True)
        print("no risk model - web version runs on rules only")
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    if m["kind"] == "lgbm":
        import lightgbm
        booster = lightgbm.Booster(model_file=os.path.join(path.parent, m.pop("model_file")))
        dump = booster.dump_model()
        assert all(t.get("shrinkage", 1) for t in dump["tree_info"])
        m["trees"] = [t["tree_structure"] for t in dump["tree_info"]]
    (web / "risk.json").write_text(json.dumps(m), encoding="utf-8")
    print(web / "risk.json", m["kind"])


if __name__ == "__main__":
    main()
