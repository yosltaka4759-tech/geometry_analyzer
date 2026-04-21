# src/plugins/geometry_analyzer.py
from __future__ import annotations

import cv2, json
import numpy as np
from skimage import measure
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Any, List #tang added
import redis.asyncio as aioredis #tang added


from src.core.plugin_base import PluginBase
from src.core.model import Frame, Mask, GeometryStats
from src.core.utils import get_logger
#tang added
from src.core.event_bus import on
from src.core.config_runtime import RuntimeConfigManager as RCM
from gui.img_utils import arr_to_b64

log = get_logger()

# このプラグイン内だけで使う簡易カラーマップ（他プラグインには干渉しない）
DEFAULT_COLORMAP = {
    0: (0, 0, 0),     # background
    1: (0, 0, 150),   # class-1 (red系)
    2: (0, 150, 0),   # class-2 (green系)
    3: (0, 255, 255), # class-3
}
#tang added , local default, usually overridden by cfg
DEFAULT_GEOM_CFG = {
    "save_dir": "outputs/geometry_viz",
    "pixel_size_nm": None,
}

# ---------- ヘルパ ----------

def _extract_frame_mask(item: Any) -> Tuple[Optional[Frame], Optional[np.ndarray], str]:
    """
    上流 payload から (frame, mask_array(H,W), id_str) を安全に取り出す。
    - (frame, mask) / (frame, mask, stats...) / Mask 単体 / (frame, ndarray) などに対応
    失敗したら (None, None, "unknown")
    """
    frame: Optional[Frame] = None
    mask_arr: Optional[np.ndarray] = None
    id_str = "unknown"

    if isinstance(item, tuple):
        for x in item:
            if frame is None and isinstance(x, Frame):
                frame = x
        for x in item:
            if isinstance(x, Mask):
                m = getattr(x, "mask", None)
                if m is None:
                    m = getattr(x, "data", None)
                if m is not None:
                    mask_arr = np.asarray(m)
                    id_str = getattr(x, "id", getattr(frame, "id", "unknown"))
                    break
        if mask_arr is None:
            for x in item:
                if isinstance(x, np.ndarray):
                    mask_arr = x.astype("uint8")
                    id_str = getattr(frame, "id", "unknown")
                    break

    elif isinstance(item, Mask):  #	•	Mask 単体の場合の取り出し。
        m = getattr(item, "mask", None)
        if m is None:
            m = getattr(item, "data", None)
        if m is not None:
            mask_arr = np.asarray(m)
            id_str = getattr(item, "id", "unknown")

    elif isinstance(item, Frame):  #Frameにmask属性がぶら下がっているケースにも対応。
        mobj = getattr(item, "mask", None)
        if isinstance(mobj, Mask):
            m = getattr(mobj, "mask", None) or getattr(mobj, "data", None)
            if m is not None:
                frame = item
                mask_arr = np.asarray(m)
                id_str = getattr(mobj, "id", getattr(item, "id", "unknown"))

    return frame, mask_arr, id_str


def _guess_roi_scale(frame: Optional[Frame], mask_arr: np.ndarray) -> float:
    """
    ROIの「元画像上の幅」と「U-Net出力マスクの幅」からスケール倍率を推定。
    返り値は倍率（例: 2.0 なら 'マスク1px=元画像2px'）。
    情報が無ければ 1.0 を返す。
    1枚のフレーム（画像）とそのマスク配列から、
   「ROIのスケール（倍率）」を推定して小数値で返す関数。
    """
    try:
        Hm, Wm = mask_arr.shape[:2]
        if Wm <= 0:
            return 1.0

        def _fetch_box(host, key): #オブジェクトや辞書から値を安全・共通の方法で取り出す
            if host is None: return None
            if hasattr(host, key):
                return getattr(host, key)
            if isinstance(host, dict):
                return host.get(key)
            return None

        candidates = []
        host_list = []
        if frame is not None:
            host_list = [getattr(frame, "extra", None), frame]

        for host in host_list:
            for key in ("roi", "roi_box", "bbox", "roi_xywh", "roi_xyxy"):
                box = _fetch_box(host, key)
                if box is None:
                    continue
                # (x,y,w,h) or (x1,y1,x2,y2) or dict を吸収,
                #このコードで倍率を推定
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    x, y, a, b = box
                    if a > x and b > y:  # (x1,y1,x2,y2)
                        Wroi = float(a - x)
                    else:                # (x,y,w,h)
                        Wroi = float(a)
                    candidates.append(Wroi)
                elif isinstance(box, dict):
                    if {"x","y","w","h"}.issubset(box.keys()):
                        candidates.append(float(box["w"]))
                    elif {"x1","y1","x2","y2"}.issubset(box.keys()):
                        candidates.append(float(box["x2"] - box["x1"]))

        if candidates:
            return float(candidates[0]) / float(Wm)
    except Exception:
        pass
    return 1.0


def _colorize(mask_arr: np.ndarray, cmap: dict[int, tuple[int,int,int]]) -> np.ndarray:
    h, w = mask_arr.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for cid, color in cmap.items():
        rgb[mask_arr == cid] = color
    return rgb


def _clamp_pt(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))


# ---------- 本体 ----------
class GeometryAnalyzer(PluginBase):
    """
    U-Netラベルマップから幾何量を計算し、画像に最小限の注記を描画して保存。
    - クラス1(赤): PCA主軸(白線) + 長さ（nm優先 / 無ければpx）
    - クラス2(緑): 面積のみ（nm²優先 / 無ければpx²）
    下流へは **受け取ったpayloadをそのまま返す**（互換維持）。
    """
    name = "geometry_analyzer"

    def __init__(self, pixel_size_nm: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pixel_size_nm = pixel_size_nm
        self._out_dir: Optional[Path] = None
        self._cmap = DEFAULT_COLORMAP.copy()
        self._rds: aioredis.Redis | None = None #tang added

        # hotfix オプション（将来 GUI で設定可能にするかも）
        self.draw_pca_major: bool = True
        self.green_show_area: bool = True
        self.save_csv: bool = False
        self.save_image: bool      = True     #tang added
        self.push_image: bool = True
        self.push_stats: bool = True 

    async def setup(self):
        # 1) tang added
        cfg = getattr(self, "cfg", {}) or {}
        save_root = Path(cfg.get("save_dir", DEFAULT_GEOM_CFG["save_dir"]))
        self.pixel_size_nm = cfg.get("pixel_size_nm", DEFAULT_GEOM_CFG["pixel_size_nm"]) # get from cfg, if None, use local default

        #out_root = Path("outputs/geometry_viz")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_dir = save_root / ts
        self._out_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[{self.name}] save dir = {self._out_dir}")
        
        #read from cfg
        self.draw_pca_major  = bool(cfg.get("draw_pca_major", True))
        self.green_show_area = bool(cfg.get("green_show_area", True))
        self.save_image      = bool(cfg.get("save_image", True))
        self.save_csv        = bool(cfg.get("save_csv", False))

      # 2) RuntimeConfig  /tang  added
        rc = RCM.get()
        self._apply_cfg(rc)
        on("config_updated", self._on_cfg)   # 订阅热更新事件

       # 3) Redis  /tang added
        self._rds = aioredis.from_url("redis://localhost:6379/0")

    async def close(self): 
        if self._rds is not None:
            await self._rds.aclose()
    
    def _apply_cfg(self, rc):
        """copy from RuntimeConfig to local settings"""
        self.draw_pca_major  = getattr(rc, "geom_draw_pca_major", self.draw_pca_major)
        self.green_show_area = getattr(rc, "geom_green_show_area", self.green_show_area)
        self.save_csv        = getattr(rc, "geom_save_csv", self.save_csv)
        self.save_image      = getattr(rc, "geom_save_image", self.save_image)   
        self.push_image      = getattr(rc, "geom_push_image", self.push_image)
        self.push_stats      = getattr(rc, "geom_push_stats", self.push_stats)
    
    async def _on_cfg(self, rc):
        """event handler for runtime config update"""
        self._apply_cfg(rc)
        log.info(f"[{self.name}] runtime cfg updated: "
                 f"draw_pca={self.draw_pca_major}, green_area={self.green_show_area}, "
                 f"push_img={self.push_image}, push_stats={self.push_stats}")

    async def process(self, item):
        """
        何が来ても解析 → 画像保存。返り値は受け取った item（本線互換）。
        """
        frame, mask_arr, id_str = _extract_frame_mask(item)
        if mask_arr is None:
            log.warning(f"[{self.name}] no mask found in payload ({type(item)}); skip analyze")
            return item

        label_img = mask_arr.astype("uint8")

        # スケール（nm/px）を推定（ROIリサイズ対策）
        base_px2nm = getattr(frame, "pixel_size_nm", None) or self.pixel_size_nm #get from frame, if None, use local.
        scale_roi = _guess_roi_scale(frame, label_img)
        eff_px2nm = base_px2nm * scale_roi if base_px2nm is not None else None

        results = []
        rgb = _colorize(label_img, self._cmap)
        H, W = rgb.shape[:2]

        class_ids = np.unique(label_img)
        class_ids = class_ids[class_ids != 0]

        for cls_id in class_ids:
            binary = (label_img == cls_id).astype(np.uint8)
            labeled = measure.label(binary, connectivity=2)

            for p in measure.regionprops(labeled):
                area_px = float(p.area)
                area_nm2 = float(area_px) * (eff_px2nm ** 2) if eff_px2nm is not None else None

                # ラベルのピクセル座標群（画像全体座標）
                rr_cc = np.column_stack(np.nonzero(p.image))
                rr_cc[:, 0] += p.bbox[0]  # y
                rr_cc[:, 1] += p.bbox[1]  # x

                cx_px = float(p.centroid[1])
                cy_px = float(p.centroid[0])

                # --- 表示 ---
                if cls_id == 1:
                   # PCA主軸
                    pc = rr_cc.astype(np.float32)
                    mu = pc.mean(axis=0, keepdims=True)
                    cov = np.cov((pc - mu).T)
                    vals, vecs = np.linalg.eigh(cov)
                    v = vecs[:, int(np.argmax(vals))]  # (dy, dx)

                    t = (pc - mu).dot(v)
                    tmin, tmax = float(t.min()), float(t.max())

                    # 端点（x,y）に変換
                    x1 = float(mu[0, 1] + v[1] * tmin)
                    y1 = float(mu[0, 0] + v[0] * tmin)
                    x2 = float(mu[0, 1] + v[1] * tmax)
                    y2 = float(mu[0, 0] + v[0] * tmax)

                    x1i, y1i = _clamp_pt(int(round(x1)), int(round(y1)), W, H)
                    x2i, y2i = _clamp_pt(int(round(x2)), int(round(y2)), W, H)

                    # 主軸長
                    L_px = float(np.hypot(x2 - x1, y2 - y1))
                    L_nm = (L_px * eff_px2nm) if eff_px2nm is not None else None

                    if self.draw_pca_major:
                        # 白線を描く
                        cv2.line(rgb, (x1i, y1i), (x2i, y2i), (255, 255, 255), 1, cv2.LINE_AA)

                        # ラベル（長さ）
                        if L_nm is not None:
                            text_len = f"A={L_nm:.1f} nm"
                        else:
                            text_len = f"A={L_px:.1f} px"

                        mx = int(round((x1i + x2i) * 0.5))
                        my = int(round((y1i + y2i) * 0.5)) - 6
                        cv2.putText(rgb, text_len, (mx, my),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        # ラベル（面積）
                        if area_nm2 is not None:
                            text_area = f"S1={area_nm2:.1f} nm^2"
                        else:
                            text_area = f"S1={area_px:.1f} px^2"

                        cv2.putText(rgb, text_area, (mx, my + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                    # 結果登録
                    results.append({
                        "class_id": int(cls_id),
                        "area_px": area_px,
                        "area_nm2": area_nm2,
                        "pca_major_len_px": L_px,
                        "pca_major_len_nm": float(L_nm) if L_nm is not None else None,
                        "pca_line_p1": (x1, y1),
                        "pca_line_p2": (x2, y2),
                        "centroid_x_px": cx_px,
                        "centroid_y_px": cy_px,
                    })
                elif cls_id == 2:
                    if self.green_show_area:
                        # 緑は面積のみ
                        if area_nm2 is not None:
                            text = f"S2={area_nm2:.1f} nm^2"
                        else:
                            text = f"S2={area_px:.1f} px^2"

                        # bbox 右上付近を初期アンカーに
                        x0, y0, x1b, y1b = p.bbox[1], p.bbox[0], p.bbox[3], p.bbox[2]
                        tx = x1b + 4
                        ty = y0 + 14

                        # ---- ここがポイント：テキストサイズで内側に収める ----
                        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        # 右端・左端はみ出し対策（幅ぶん引いてクランプ）
                        tx = max(0, min(W - tw - 1, tx))
                        # 上下はみ出し対策（ベースラインと高さを考慮）
                        ty = max(th + base + 1, min(H - base - 1, ty))
                        
                        cv2.putText(rgb, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


                    results.append({
                        "class_id": int(cls_id),
                        "area_px": area_px,
                        "area_nm2": area_nm2,
                        "centroid_x_px": cx_px,
                        "centroid_y_px": cy_px,
                    })

                else:
                    # 他クラスは何も描かないが、面積などは保持
                    results.append({
                        "class_id": int(cls_id),
                        "area_px": area_px,
                        "area_nm2": area_nm2,
                        "centroid_x_px": cx_px,
                        "centroid_y_px": cy_px,
                    })

        # 保存 if true
        if self.save_image and self._out_dir is not None:
            out_path = self._out_dir / f"{id_str}.png"
            cv2.imwrite(str(out_path), rgb)
            log.info(f"[{self.name}] saved → {out_path}")
        else:
            log.debug(f"[{self.name}] save_image=False → skip PNG write")


        # 統計は frame.extra に積んでおく（任意）
        stats = GeometryStats(pixel_size_nm=base_px2nm, objects=results)
        try:
            if frame is not None:
                if not hasattr(frame, "extra") or frame.extra is None:
                    frame.extra = {}
                frame.extra["geometry_stats"] = stats
        except Exception:
            pass

        # ---- 推流到 GUI（按开关） ----
        if self._rds is not None:
            # --- 1) 尝试从 item 中找 Mask：对齐 root_id ---
            mask_obj = None
            if isinstance(item, Mask):
                mask_obj = item
            elif isinstance(item, tuple):
                for x in item:
                    if isinstance(x, Mask):
                        mask_obj = x
                        break

            root_id: str | None = None
            roi_id: int | None = None

            if mask_obj is not None:
                # 和 MaskVisualizer 保持一致
                root_id = mask_obj.meta.get("root_id", mask_obj.frame_id)
                # roi_id 如果上游有就用，没有就先 0
                roi_id = getattr(mask_obj, "roi_id", None)
                if roi_id is None:
                    roi_id = mask_obj.meta.get("roi_id", 0)

            # --- 2) 如果没有 Mask 或 meta 中没 root_id，就退回 Frame ---
            if root_id is None and frame is not None:
                # 优先 Frame.frame_id（如果有）
                root_id = getattr(frame, "frame_id", None)
                # 其次 Frame.extra["root_id"]
                if root_id is None and hasattr(frame, "extra") and isinstance(frame.extra, dict):
                    root_id = frame.extra.get("root_id")
                # 最后退回 frame.id 或 id_str
                if root_id is None:
                    root_id = getattr(frame, "id", id_str)

                # roi_id 也尽量从 frame 上取
                if roi_id is None:
                    roi_id = getattr(frame, "roi_id", 0)

            # --- 3) 最后的兜底 ---
            if root_id is None:
                root_id = id_str
            if roi_id is None:
                roi_id = 0

            if self.push_image:
                payload_img = json.dumps({
                    "frame_id": root_id,
                    "roi_id"  : roi_id,
                    "source"  : "geom_img",
                    "data"    : arr_to_b64(rgb),
                })
                await self._rds.publish("geometry_img", payload_img)

            if self.push_stats:
                payload_stats = json.dumps({
                    "frame_id": root_id,
                    "roi_id"  : roi_id,
                    "source"  : "geom_stats",
                    "data"    : {
                        "pixel_size_nm": base_px2nm,
                        "objects": results,
                    },
                })
                await self._rds.publish("geometry_stats", payload_stats)
                
        log.info(f"[{self.name}] emit stats n_objs={len(results)}")
        return item  # 下流互換