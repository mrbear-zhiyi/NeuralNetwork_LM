#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_data.py
------------------------------------------------------
本文件包含以下函数：
0. 提取值和单位      _extract_value_unit()
1. 读取数据         get_data()
2. 单位统一         unit_unified()
3. 特定范围的核      specific_ranged()
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import json
from pandas.api.types import is_numeric_dtype


# 0. 提取值和单位(私有函数)
def _extract_value_unit(entry: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
    if entry is None:
        return None, None
    return entry.get("value"), entry.get("unit")

# 0.1 提取值和类型(私有函数)
def _extract_value_type(entry: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
    if entry is None:
        return None, None
    unc = entry.get("uncertainty")
    return entry.get("value"), unc.get("type") if unc else None


# 0.2 提取值, 单位, 类型(私有函数)
def _extract_value_unit_type(entry: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if entry is None:
        return None, None, None
    unc = entry.get("uncertainty")
    return entry.get("value"), entry.get("unit"), unc.get("type") if unc else None


# 1. 读取数据
def get_data(path: str) -> pd.DataFrame:
    # 1) 读取json文件
    with open(path, "r", encoding="utf-8") as f:
        data_json: Dict[str, Any] = json.load(f)

    # 2) 确定行表头rows、列表头columns
    table: List[List[Any]] = list()
    rows = list(data_json.keys())
    columns = [
        'Z', 'N', 'A','half_life', 'half_life_unit','half_life_type','br_alpha', 'br_alpha_type','half_life_alpha', 'half_life_alpha_unit','Q_alpha', 'Q_alpha_unit'
    ]
    # columns = [
    #     'Z', 'N', 'A', 'br_alpha', 'br_beta_p', 'br_beta_m', 'br_SF', 'br_EC',
    #     'half_life', 'half_life_unit', 'half_life_alpha', 'half_life_alpha_unit',
    #     'half_life_beta_p', 'half_life_beta_p_unit', 'half_life_beta_m', 'half_life_beta_m_unit',
    #     'half_life_SF', 'half_life_SF_unit', 'half_life_EC', 'half_life_EC_unit',
    #     'Q_alpha', 'Q_alpha_unit', 'Q_beta_p', 'Q_beta_p_unit', 'Q_beta_m', 'Q_beta_m_unit',
    #     'Q_EC', 'Q_EC_unit'
    # ]

    # 3) 数据库与列表头的映射关系
    mode_col_map: Dict[str, Tuple[str, str,str]] = {
        "A": ("br_alpha",'half_life_alpha', 'half_life_alpha_unit'),
        # "B+": ("br_beta_p", "half_life_beta_p", "half_life_beta_p_unit"),
        # "B-": ("br_beta_m", "half_life_beta_m", "half_life_beta_m_unit"),
        # "SF": ("br_SF", "half_life_SF", "half_life_SF_unit"),
        # "EC": ("br_EC", "half_life_EC", "half_life_EC_unit"),
        # # 对于分支比EC+B+，当作B+
        # "EC+B+": ("br_beta_p", "half_life_beta_p", "half_life_beta_p_unit")
    }

    # 4) 获取核素信息
    for nuclide_name in rows:
        nuclide = data_json[nuclide_name]
        record: Dict[str, Any] = {key: None for key in columns}

        # --- 基本核素信息 ---------------------------------------------------
        record["Z"] = nuclide.get("z")
        record["N"] = nuclide.get("n")
        record["A"] = nuclide.get("a")
        record["Q_alpha"], record["Q_alpha_unit"] = _extract_value_unit(nuclide.get("alpha"))
        # record["Q_EC"], record["Q_EC_unit"] = _extract_value_unit(nuclide.get("electronCapture"))
        # record["Q_beta_p"], record["Q_beta_p_unit"] = _extract_value_unit(nuclide.get("positronEmission"))
        # record["Q_beta_m"], record["Q_beta_m_unit"] = _extract_value_unit(nuclide.get("betaMinus"))

        # --- 总半衰期 --------------------------------------------------------
        levels = nuclide.get("levels", [])
        level0 = next(
            (lv for lv in levels if lv.get("energy", {}).get("value") == 0),
            levels[0] if levels else None
        )
        if level0 and "halflife" in level0:
            (record['half_life'], record['half_life_unit'],
             record['half_life_type']) = _extract_value_unit_type(level0['halflife'])
        total_T = record["half_life"]

        # --- 分支比与各模式半衰期 ---------------------------------------------
        observed = level0.get("decayModes", {}).get("observed", []) if level0 else []
        branch_accumulator: Dict[str, float] = {}
        for mode_info in observed:
            raw_tag: str = mode_info.get("mode")
            br_val, br_typ = _extract_value_type(mode_info)  # 这里提取分支比类型
            if br_val is None:
                continue
            #canonical_tag = "A" if raw_tag == "A" else raw_tag
            if raw_tag not in mode_col_map:
                continue
            branch_accumulator[raw_tag] = branch_accumulator.get(raw_tag, 0.0) + br_val
            record[f"{mode_col_map[raw_tag][0]}_type"] = br_typ

        # 写入分支比和对应半衰期
        for tag, br_total in branch_accumulator.items():
            br_col, hl_col, hl_unit_col = mode_col_map[tag]
            record[br_col] = br_total
            if total_T is not None and br_total:
                record[hl_col] = total_T / (br_total / 100.0)
                record[hl_unit_col] = record["half_life_unit"]

        # --- 追加到总表 -----------------------------------------------------
        table.append([record[col] for col in columns])

    # 5) 生成并返回 DataFrame
    df = pd.DataFrame(table, index=rows, columns=columns)
    return df


# 2. 单位统一
def unit_unified(data: pd.DataFrame) -> pd.DataFrame:
    """统一单位为能量 keV、时间 s, 基于统一后半衰期计算主导衰变模式"""
    df = data.copy()
    # --- 能量单位转换 ------------------------------------------------------
    energy_cols = [
        ("Q_alpha", "Q_alpha_unit"),
    ]
    # energy_cols = [
    #     ("Q_alpha", "Q_alpha_unit"),
    #     ("Q_EC", "Q_EC_unit"),
    #     ("Q_beta_p", "Q_beta_p_unit"),
    #     ("Q_beta_m", "Q_beta_m_unit")
    # ]
    energy_factor: Dict[str, float] = {"keV": 1e-3, "MeV": 1, "eV": 1e-6, "GeV": 1e3}
    for val_col, unit_col in energy_cols:
        def _convert_energy(row):
            val, unit = row[val_col], row[unit_col]
            if val is None or unit is None:
                return val
            if unit not in energy_factor:
                return val
            return val * energy_factor[unit]
        df[val_col] = df.apply(_convert_energy, axis=1)
        df[unit_col] = df[unit_col].apply(lambda u: "MeV" if u in energy_factor else u)

    # --- 时间单位转换 ------------------------------------------------------
    time_cols = [("half_life_alpha", "half_life_alpha_unit")]
    # time_cols = [
    #     ("half_life", "half_life_unit"),
    #     ("half_life_SF", "half_life_SF_unit"),
    #     ("half_life_alpha", "half_life_alpha_unit"),
    #     ("half_life_EC", "half_life_EC_unit"),
    #     ("half_life_beta_p", "half_life_beta_p_unit"),
    #     ("half_life_beta_m", "half_life_beta_m_unit")
    # ]
    time_factor: Dict[str, float] = {
        "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
        "m": 60.0, "h": 3600.0, "d": 86400.0, "y": 31557600.0
    }
    for val_col, unit_col in time_cols:
        def _convert_time(row):
            val, unit = row[val_col], row[unit_col]
            if val is None or unit is None:
                return val
            if unit not in time_factor:
                return val
            return val * time_factor[unit]
        df[val_col] = df.apply(_convert_time, axis=1)
        df[unit_col] = df[unit_col].apply(lambda u: "s" if u in time_factor else u)

    # --- 基于统一后半衰期计算主导衰变模式 -------------------------------------
    decay_cols = {"alpha": "half_life_alpha",}
    # decay_cols = {"alpha": "half_life_alpha",
    #               "beta_plus": "half_life_beta_p",
    #               "beta_minus": "half_life_beta_m",
    #               "SF": "half_life_SF",
    #               "EC": "half_life_EC"}

    # def _calc_dominant(row):
    #     vals = {mode: row[col] for mode, col in decay_cols.items() if pd.notnull(row[col])}
    #     if not vals:
    #         return None
    #     return min(vals, key=lambda m: vals[m])
    # df["Dominant_decay_mode"] = df.apply(_calc_dominant, axis=1)
    return df

# 2. 能量半衰期标签筛选
def Data_modified(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    column_to_check = ['Q_alpha']

    #将负Q标注为None
    for col in column_to_check:
        df[col] = df[col].apply(lambda x: None if pd.notna(x) and is_numeric_dtype(type(x)) and x < 0 else x)

    #删除空值
    df = df.dropna(subset=['Q_alpha', 'half_life_alpha'])

    # 删除 br_alpha_type 为 "unreported" 或 "limit" 的行
    # 删除 half_life_type 为 "unreported" 或 "limit" 的行
    df = df[~df["br_alpha_type"].isin(["unreported", "limit"])]
    df = df[~df["half_life_type"].isin(["unreported", "limit"])]
    return df

