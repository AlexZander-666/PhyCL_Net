#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SisFall 子群体公平性分析脚本

输入: 预测结果 DataFrame/CSV，需包含 subject_id(1-38), y_true(0/1), y_pred(0/1), fall_type(F01-F15 或 ADL)
输出: 分层统计表 (CSV + Markdown) 与 3 张箱线图

使用示例:
    python code/scripts/sisfall_subgroup_fairness.py \
        --preds-path outputs/sisfall_preds.csv \
        --output-dir outputs/subgroup_fairness \
        --figure-dir figures/subgroup_fairness

指标说明:
    Accuracy = (TP + TN) / 所有样本
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    所有指标先在 subject 层面计算，再在子群体内做均值±标准差，避免样本数过大的受试者主导结果。
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 来自 SisFall 官方 Readme 的受试者元数据 (年龄、身高、体重、性别)
# 通过 subject_id 1-23 -> SA01-SA23 (Youth), 24-38 -> SE01-SE15 (Elderly)
_SUBJECT_ROWS = [
    # Youth (SA)
    ("SA01", 26, 165, 53, "F"),
    ("SA02", 23, 176, 58.5, "M"),
    ("SA03", 19, 156, 48, "F"),
    ("SA04", 23, 170, 72, "M"),
    ("SA05", 22, 172, 69.5, "M"),
    ("SA06", 21, 169, 58, "M"),
    ("SA07", 21, 156, 63, "F"),
    ("SA08", 21, 149, 41.5, "F"),
    ("SA09", 24, 165, 64, "M"),
    ("SA10", 21, 177, 67, "M"),
    ("SA11", 19, 170, 80.5, "M"),
    ("SA12", 25, 153, 47, "F"),
    ("SA13", 22, 157, 55, "F"),
    ("SA14", 27, 160, 46, "F"),
    ("SA15", 25, 160, 52, "F"),
    ("SA16", 20, 169, 61, "F"),
    ("SA17", 23, 182, 75, "M"),
    ("SA18", 23, 181, 73, "M"),
    ("SA19", 30, 170, 76, "M"),
    ("SA20", 30, 150, 42, "F"),
    ("SA21", 30, 183, 68, "M"),
    ("SA22", 19, 158, 50.5, "F"),
    ("SA23", 24, 156, 48, "F"),
    # Elderly (SE)
    ("SE01", 71, 171, 102, "M"),
    ("SE02", 75, 150, 57, "F"),
    ("SE03", 62, 150, 51, "F"),
    ("SE04", 63, 160, 59, "F"),
    ("SE05", 63, 165, 72, "M"),
    ("SE06", 60, 163, 79, "M"),
    ("SE07", 65, 168, 76, "M"),
    ("SE08", 68, 163, 72, "F"),
    ("SE09", 66, 167, 65, "M"),
    ("SE10", 64, 156, 66, "F"),
    ("SE11", 66, 169, 63, "F"),
    ("SE12", 69, 164, 56.5, "M"),
    ("SE13", 65, 171, 72.5, "M"),
    ("SE14", 67, 163, 58, "M"),
    ("SE15", 64, 150, 50, "F"),
]

FALL_TYPES = [f"F{str(i).zfill(2)}" for i in range(1, 16)]


def build_metadata() -> pd.DataFrame:
    """构建受试者元数据 DataFrame，附加 age_group 与 gender 全称。"""
    rows: List[Dict[str, object]] = []
    for idx, (code, age, height, weight, gender) in enumerate(_SUBJECT_ROWS, start=1):
        age_group = "Youth" if code.startswith("SA") else "Elderly"
        rows.append(
            {
                "subject_id": idx,
                "subject_code": code,
                "age": age,
                "height_cm": height,
                "weight_kg": weight,
                "gender": gender,
                "gender_full": "Male" if gender.upper() == "M" else "Female",
                "age_group": age_group,
            }
        )
    return pd.DataFrame(rows)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """计算 Accuracy / Sensitivity / Specificity，分母为 0 时返回 NaN."""
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    tp = np.sum((y_true_np == 1) & (y_pred_np == 1))
    tn = np.sum((y_true_np == 0) & (y_pred_np == 0))
    fp = np.sum((y_true_np == 0) & (y_pred_np == 1))
    fn = np.sum((y_true_np == 1) & (y_pred_np == 0))

    total = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / total if total > 0 else np.nan

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _format_mean_std(series: pd.Series) -> str:
    """将均值±标准差格式化为字符串，空集时返回 'nan'。"""
    if len(series.dropna()) == 0:
        return "nan"
    return f"{series.mean():.3f}±{series.std(ddof=0):.3f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """无依赖生成简单 Markdown 表格。"""
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join([header_line, sep_line, *body])


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    """确保必需字段存在，否则抛出异常。"""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def attach_metadata(df_preds: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """合并预测与受试者元数据，并填补 fall_type 中的 ADL 占位。"""
    df = df_preds.copy()
    df["fall_type"] = df["fall_type"].fillna("ADL").str.upper()
    df["subject_id"] = df["subject_id"].astype(int)
    merged = df.merge(meta_df, on="subject_id", how="left")
    if merged["subject_code"].isna().any():
        raise ValueError("subject_id 未在内置元数据中找到，请检查编号是否为 1-38。")
    return merged


def compute_subject_metrics(
    df: pd.DataFrame, fall_type: Optional[str] = None
) -> pd.DataFrame:
    """
    在 subject 层面计算指标。
    - fall_type=None: 使用该 subject 的全部样本。
    - fall_type=FXX: 使用该 subject 的 FXX + 所有 ADL 样本，保证 specificity 有负样本。
    """
    records: List[Dict[str, object]] = []
    for subject_id, sdf in df.groupby("subject_id"):
        if fall_type:
            sdf_pos = sdf[sdf["fall_type"] == fall_type]
            if sdf_pos.empty:
                # 该 subject 未执行此跌倒类型，跳过以免稀释统计
                continue
            # 仅针对该 subject，加入 ADL 作为负样本，以免只看单一跌倒类型时特异度分母为 0
            sdf_adl = sdf[sdf["y_true"] == 0]
            sdf_use = pd.concat([sdf_pos, sdf_adl], ignore_index=True)
        else:
            sdf_use = sdf

        metrics = calculate_metrics(sdf_use["y_true"], sdf_use["y_pred"])
        records.append(
            {
                "subject_id": subject_id,
                "fall_type": fall_type or "ALL",
                "age_group": sdf["age_group"].iloc[0],
                "gender_full": sdf["gender_full"].iloc[0],
                "n_samples": len(sdf_use),
                "n_events": int((sdf_use["y_true"] == 1).sum()),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def aggregate_groups(
    subj_metrics: pd.DataFrame, group_cols: List[str], dimension: str
) -> pd.DataFrame:
    """对 subject 指标做子群体聚合，输出均值±标准差。"""
    rows: List[Dict[str, object]] = []
    for keys, gdf in subj_metrics.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label_parts = [f"{col}={val}" for col, val in zip(group_cols, keys)]
        rows.append(
            {
                "dimension": dimension,
                "group": ", ".join(label_parts),
                "n_samples": int(gdf["n_samples"].sum()),
                "n_events": int(gdf["n_events"].sum()),
                "Accuracy (Mean±Std)": _format_mean_std(gdf["accuracy"]),
                "Sensitivity (Mean±Std)": _format_mean_std(gdf["sensitivity"]),
                "Specificity (Mean±Std)": _format_mean_std(gdf["specificity"]),
            }
        )
    return pd.DataFrame(rows)


def build_stratified_table(subj_all: pd.DataFrame, subj_ft: pd.DataFrame) -> pd.DataFrame:
    """拼接单维与交叉维度的分层统计表。"""
    tables = [
        aggregate_groups(subj_all, ["age_group"], "age_group"),
        aggregate_groups(subj_all, ["gender_full"], "gender"),
        aggregate_groups(subj_all, ["age_group", "gender_full"], "age_group x gender"),
        aggregate_groups(subj_ft, ["fall_type"], "fall_type"),
    ]
    return pd.concat(tables, ignore_index=True)


def plot_boxplots(
    subj_all: pd.DataFrame, subj_ft: pd.DataFrame, figure_dir: Path
) -> None:
    """生成 3 张箱线图，对比不同子群体的指标分布。"""
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = ["accuracy", "sensitivity", "specificity"]
    long_all = subj_all.melt(
        id_vars=["subject_id", "age_group", "gender_full"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    # 图 1: Youth vs Elderly
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=long_all,
        x="metric",
        y="value",
        hue="age_group",
        palette="Set2",
    )
    sns.despine()
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Age Group: Metric Distributions")
    plt.legend(title="Age Group")
    plt.tight_layout()
    plt.savefig(figure_dir / "box_age_group.png", dpi=300)
    plt.close()

    # 图 2: Male vs Female
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=long_all,
        x="metric",
        y="value",
        hue="gender_full",
        palette="Set2",
    )
    sns.despine()
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Gender: Metric Distributions")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(figure_dir / "box_gender.png", dpi=300)
    plt.close()

    # 图 3: 不同跌倒类型的敏感度分布
    sens_ft = subj_ft[["fall_type", "sensitivity"]].dropna()
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=sens_ft,
        x="fall_type",
        y="sensitivity",
        palette="Set2",
    )
    sns.despine()
    plt.xlabel("Fall Type")
    plt.ylabel("Sensitivity")
    plt.title("Fall-Type Sensitivity Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figure_dir / "box_fall_type_sensitivity.png", dpi=300)
    plt.close()


def run_analysis(
    df_preds: pd.DataFrame,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    """核心执行函数: 读取数据 -> 元数据注入 -> 计算子群体指标 -> 导出表格与图像。"""
    ensure_columns(df_preds, ["subject_id", "y_true", "y_pred", "fall_type"])
    meta_df = build_metadata()
    df_with_meta = attach_metadata(df_preds, meta_df)

    # 在 subject 层面计算总体指标
    subj_all = compute_subject_metrics(df_with_meta, fall_type=None)

    # 针对每个跌倒类型计算 subject 层面的敏感度/特异度
    subj_ft_list = []
    for ft in FALL_TYPES:
        if (df_with_meta["fall_type"] == ft).any():
            df_ft = compute_subject_metrics(df_with_meta, fall_type=ft)
            if not df_ft.empty:
                subj_ft_list.append(df_ft)
    subj_ft = pd.concat(subj_ft_list, ignore_index=True) if subj_ft_list else pd.DataFrame()

    # 汇总表格
    table_df = build_stratified_table(subj_all, subj_ft)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_csv = output_dir / "subgroup_report.csv"
    table_md = output_dir / "subgroup_report.md"
    table_df.to_csv(table_csv, index=False)
    table_md.write_text(dataframe_to_markdown(table_df), encoding="utf-8")
    logging.info("Saved stratified table to %s and %s", table_csv, table_md)

    # 绘图
    if not subj_ft.empty:
        plot_boxplots(subj_all, subj_ft, figure_dir)
        logging.info("Saved box plots to %s", figure_dir)
    else:
        logging.warning("未找到跌倒类型数据，跳过 fall_type 敏感度箱线图。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SisFall 子群体公平性分析 (Age/Gender/Fall Type)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preds-path",
        type=Path,
        required=True,
        help="预测结果 CSV，包含 subject_id, y_true, y_pred, fall_type",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/subgroup_fairness"),
        help="分层统计表输出目录",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/subgroup_fairness"),
        help="箱线图输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_preds = pd.read_csv(args.preds_path)
    run_analysis(df_preds, args.output_dir, args.figure_dir)


if __name__ == "__main__":
    main()
