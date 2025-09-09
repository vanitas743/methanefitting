#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODTRAN 入力 JSON 量産スクリプト
- 背景LUT（SURREF × H2OSTR）
- テストLUT（SURREF × H2OSTR × ΔCH4；下層 layer_km のみ増加）
雛形 sample.json を基に必要箇所を書き換えて出力。
"""

import argparse
import json
import math
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any

def parse_list_or_range(s: str) -> List[float]:
    """ "a,b,c" または "start:end:step" を List[float] に変換（end を含める） """
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2,3):
            raise ValueError("Range指定は start:end[:step] 形式です: {}".format(s))
        start = float(parts[0]); end = float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 1.0
        if step <= 0:
            raise ValueError("step は正である必要があります")
        n = int(math.floor((end - start) / step + 1e-9)) + 1
        vals = [round(start + i*step, 10) for i in range(max(n,0))]
        if len(vals)==0 or abs(vals[-1]-end)>1e-9:
            if (end - (vals[-1] if vals else start)) > 1e-9:
                vals.append(end)
        return vals
    else:
        return [float(x) for x in s.split(",") if x.strip()!=""]

def token_A(a: float) -> str:  # SURREF=0.123 → A123
    return "{:03d}".format(int(round(a*1000)))

def token_W(w: float) -> str:  # H2OSTR=1.416 → W1416
    return "{:04d}".format(int(round(w*1000)))

def token_D(d: float) -> str:  # ΔCH4=0.5 ppm → d050
    return "{:03d}".format(int(round(d*100)))

def load_sample(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_template(sample: Dict[str, Any]) -> Dict[str, Any]:
    """ sample から MODTRANINPUT 部分（辞書）を取り出す """
    if isinstance(sample, dict) and "MODTRAN" in sample:
        m = sample["MODTRAN"]
        if isinstance(m, list) and len(m)>0 and "MODTRANINPUT" in m[0]:
            return deepcopy(m[0]["MODTRANINPUT"])
    if isinstance(sample, dict) and "MODTRANINPUT" in sample:
        return deepcopy(sample["MODTRANINPUT"])
    raise ValueError("sample.json の構造が想定外です（'MODTRAN' または 'MODTRANINPUT' が必要）")

def wrap_to_json(modin: Dict[str, Any]) -> Dict[str, Any]:
    return {"MODTRAN": [{"MODTRANINPUT": modin}]}

def set_spectral(modin: Dict[str, Any], v1: float, v2: float, dv: float, fwhm: float) -> None:
    spec = modin.setdefault("SPECTRAL", {})
    spec["V1"] = float(v1); spec["V2"] = float(v2)
    spec["DV"] = float(dv); spec["FWHM"] = float(fwhm)
    spec["YFLAG"] = "R"  # at-sensor radiance

def set_surface_albedo(modin: Dict[str, Any], a: float) -> None:
    surf = modin.setdefault("SURFACE", {})
    surf["SURFTYPE"] = "REFL_CONSTANT"
    surf["SURREF"] = float(a)
    surf["NSURF"] = 1

def set_water_column(modin: Dict[str, Any], w: float) -> None:
    atm = modin.setdefault("ATMOSPHERE", {})
    atm["H2OSTR"] = float(w)  # H2OUNIT は雛形のまま利用

def ensure_fileoptions_csv(modin: Dict[str, Any], basename: str) -> None:
    fops = modin.setdefault("FILEOPTIONS", {})
    fops["CSVPRNT"] = "{}.csv".format(basename)

def find_profile(modin: Dict[str, Any], ptype: str) -> Dict[str, Any]:
    profs = modin.get("ATMOSPHERE", {}).get("PROFILES", [])
    for pr in profs:
        if pr.get("TYPE")==ptype:
            return pr
    raise KeyError("ATMOSPHERE.PROFILES 内に TYPE='{}' が見つかりません".format(ptype))

def set_bg_ch4_constant(modin: Dict[str, Any], bg_ppm: float) -> None:
    """背景用：CH4 プロファイル全高度を bg_ppm に置換"""
    ch4 = find_profile(modin, "PROF_CH4")
    ch4["PROFILE"] = [float(bg_ppm) for _ in ch4["PROFILE"]]

def bump_ch4_lower_layer(modin: Dict[str, Any], delta_ppm: float, layer_km: float) -> None:
    """テスト用：下層 layer_km 以内の CH4 を（現在値 + delta_ppm）に"""
    alt = find_profile(modin, "PROF_ALTITUDE")["PROFILE"]
    ch4 = find_profile(modin, "PROF_CH4")["PROFILE"]
    if len(alt) != len(ch4):
        raise ValueError("ALTITUDE と CH4 の配列長が一致しません")
    changed = 0
    for i, a in enumerate(alt):
        if a <= layer_km + 1e-9:
            ch4[i] = float(ch4[i] + delta_ppm)
            changed += 1
    if changed == 0:
        raise ValueError("layer_km={} 以下の高度点が見つかりません（ALT グリッドを確認）".format(layer_km))

# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser(description="Generate MODTRAN input JSONs for LUTs.")
    ap.add_argument("--sample", required=True, type=Path, help="雛形 sample.json のパス")
    ap.add_argument("--outdir", required=True, type=Path, help="出力ディレクトリ")
    ap.add_argument("--mode", choices=["bg","test","both"], default="both", help="生成対象（背景/テスト/両方）")
    ap.add_argument("--surref", required=True, help="SURREF 値。'0:0.5:0.01' または '0,0.05,0.1'")
    ap.add_argument("--h2ostr", required=True, help="H2OSTR 値（可降水量）。範囲/リスト対応")
    ap.add_argument("--ch4-delta", default="0.1,0.3,0.5,1.0,2.0", help="テストLUT用 ΔCH4 ppm（mode=test/both）")
    ap.add_argument("--layer-km", type=float, default=1.0, help="下層の層厚[km]（例：1.0）")
    ap.add_argument("--spectral", default="", help="V1,V2,DV,FWHM を上書き（例：'2100,2400,0.1,0.2'）")
    ap.add_argument("--name-prefix", default="", help="出力ファイル名の先頭（例：'hisui_'）")
    ap.add_argument("--bg-ch4", type=float, default=None, help="背景CH4(ppm)。指定時は背景LUTで全高度をこの値に固定")
    ap.add_argument("--dry-run", action="store_true", help="出力せず計画のみ表示")
    args = ap.parse_args()

    if not args.sample.exists():
        ap.error("--sample が見つかりません: {}".format(args.sample))
    args.outdir.mkdir(parents=True, exist_ok=True)

    surrefs = parse_list_or_range(args.surref)
    wvals   = parse_list_or_range(args.h2ostr)
    dvals   = parse_list_or_range(args.ch4_delta) if args.mode in ("test","both") else []

    sample = load_sample(args.sample)
    template = extract_template(sample)

    spec_override = None
    if args.spectral:
        toks = [float(x) for x in args.spectral.split(",")]
        if len(toks)!=4:
            ap.error("--spectral は 'V1,V2,DV,FWHM' の4要素です")
        spec_override = toks

    total = 0

    # 背景LUT
    if args.mode in ("bg","both"):
        for A in surrefs:
            for W in wvals:
                modin = deepcopy(template)
                set_surface_albedo(modin, A)
                set_water_column(modin, W)
                if args.bg_ch4 is not None:
                    set_bg_ch4_constant(modin, args.bg_ch4)
                if spec_override:
                    set_spectral(modin, *spec_override)
                base = "{}bg_A{}_W{}".format(args.name_prefix, token_A(A), token_W(W))
                ensure_fileoptions_csv(modin, base)
                outjson = wrap_to_json(modin)
                outpath = args.outdir / ("{}.json".format(base))
                if args.dry_run:
                    print("[BG]", outpath.name)
                else:
                    with outpath.open("w", encoding="utf-8") as f:
                        json.dump(outjson, f, ensure_ascii=False, indent=2)
                total += 1

    # テストLUT（下層 layer_km のみ CH4 増加）
    if args.mode in ("test","both"):
        for A in surrefs:
            for W in wvals:
                for d in dvals:
                    modin = deepcopy(template)
                    set_surface_albedo(modin, A)
                    set_water_column(modin, W)
                    if args.bg_ch4 is not None:
                        set_bg_ch4_constant(modin, args.bg_ch4)
                    if spec_override:
                        set_spectral(modin, *spec_override)
                    bump_ch4_lower_layer(modin, d, args.layer_km)
                    base = "{}test_A{}_W{}_d{}ppm_L{}m".format(
                        args.name_prefix, token_A(A), token_W(W), token_D(d),
                        int(round(args.layer_km*1000))
                    )
                    ensure_fileoptions_csv(modin, base)
                    outjson = wrap_to_json(modin)
                    outpath = args.outdir / ("{}.json".format(base))
                    if args.dry_run:
                        print("[TEST]", outpath.name)
                    else:
                        with outpath.open("w", encoding="utf-8") as f:
                            json.dump(outjson, f, ensure_ascii=False, indent=2)
                    total += 1

    print("生成完了: {} ファイル → {}".format(total, args.outdir))

if __name__ == "__main__":
    main()
