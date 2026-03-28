from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


EXPECTED_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
    "Exam_Score",
]

NUMERIC_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Exam_Score",
]

CATEGORICAL_COLUMNS = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

DEFAULT_RISK_THRESHOLDS: dict[str, float] = {
    "pass_score_threshold": 60.0,
    "high_risk_score_threshold": 50.0,
    "attendance_threshold": 75.0,
    "study_hours_threshold": 12.0,
    "previous_score_threshold": 65.0,
    "sleep_hours_min": 6.0,
    "tutoring_min_sessions": 1.0,
}


@dataclass(frozen=True)
class KPIBundle:
    total_students: int
    avg_exam_score: float
    median_exam_score: float
    pass_rate_percent: float
    avg_attendance_percent: float
    avg_hours_studied: float


@dataclass(frozen=True)
class DataQualityReport:
    input_rows: int
    output_rows: int
    rows_dropped: int
    missing_critical_rows: int
    missing_numeric_cells: int
    missing_categorical_cells: int
    attendance_out_of_range_rows: int
    score_out_of_range_rows: int
    negative_hours_rows: int
    negative_sleep_rows: int
    negative_tutoring_rows: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "rows_dropped": self.rows_dropped,
            "missing_critical_rows": self.missing_critical_rows,
            "missing_numeric_cells": self.missing_numeric_cells,
            "missing_categorical_cells": self.missing_categorical_cells,
            "attendance_out_of_range_rows": self.attendance_out_of_range_rows,
            "score_out_of_range_rows": self.score_out_of_range_rows,
            "negative_hours_rows": self.negative_hours_rows,
            "negative_sleep_rows": self.negative_sleep_rows,
            "negative_tutoring_rows": self.negative_tutoring_rows,
        }


def _prepare_data_with_report(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Prepare student performance data and return cleaned data plus quality diagnostics."""
    input_rows = int(len(df))
    working = df.copy()

    for col in NUMERIC_COLUMNS:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    missing_numeric_cells = int(working[NUMERIC_COLUMNS].isna().sum().sum())
    missing_categorical_cells = int(working[CATEGORICAL_COLUMNS].isna().sum().sum())

    critical_cols = ["Exam_Score", "Attendance", "Hours_Studied"]
    missing_critical_mask = working[critical_cols].isna().any(axis=1)
    missing_critical_rows = int(missing_critical_mask.sum())

    working = working.loc[~missing_critical_mask].copy()

    for col in NUMERIC_COLUMNS:
        if working[col].isna().any():
            working[col] = working[col].fillna(working[col].median())

    for col in CATEGORICAL_COLUMNS:
        working[col] = working[col].fillna("Unknown").astype(str)

    attendance_out_of_range_rows = int(((working["Attendance"] < 0) | (working["Attendance"] > 100)).sum())
    score_out_of_range_rows = int(((working["Exam_Score"] < 0) | (working["Exam_Score"] > 100)).sum())
    negative_hours_rows = int((working["Hours_Studied"] < 0).sum())
    negative_sleep_rows = int((working["Sleep_Hours"] < 0).sum())
    negative_tutoring_rows = int((working["Tutoring_Sessions"] < 0).sum())

    # Clamp ranges to keep downstream visuals stable and interpretable.
    working["Attendance"] = working["Attendance"].clip(0, 100)
    working["Exam_Score"] = working["Exam_Score"].clip(0, 100)
    working["Hours_Studied"] = working["Hours_Studied"].clip(lower=0)
    working["Sleep_Hours"] = working["Sleep_Hours"].clip(lower=0)
    working["Tutoring_Sessions"] = working["Tutoring_Sessions"].clip(lower=0)

    working["pass_flag"] = working["Exam_Score"] >= 60
    working["attendance_band"] = pd.cut(
        working["Attendance"],
        bins=[0, 70, 80, 90, 100],
        labels=["<=70", "71-80", "81-90", "91-100"],
        include_lowest=True,
    )
    working["study_band"] = pd.cut(
        working["Hours_Studied"],
        bins=[0, 10, 20, 30, np.inf],
        labels=["<=10", "11-20", "21-30", ">30"],
        include_lowest=True,
    )
    working["score_band"] = pd.cut(
        working["Exam_Score"],
        bins=[0, 50, 60, 70, 80, 90, 100],
        labels=["<=50", "51-60", "61-70", "71-80", "81-90", "91-100"],
        include_lowest=True,
    )

    output_rows = int(len(working))
    report = DataQualityReport(
        input_rows=input_rows,
        output_rows=output_rows,
        rows_dropped=input_rows - output_rows,
        missing_critical_rows=missing_critical_rows,
        missing_numeric_cells=missing_numeric_cells,
        missing_categorical_cells=missing_categorical_cells,
        attendance_out_of_range_rows=attendance_out_of_range_rows,
        score_out_of_range_rows=score_out_of_range_rows,
        negative_hours_rows=negative_hours_rows,
        negative_sleep_rows=negative_sleep_rows,
        negative_tutoring_rows=negative_tutoring_rows,
    )
    return working, report


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    prepared_df, _ = _prepare_data_with_report(df)
    return prepared_df


def load_data_with_quality(csv_path: str) -> tuple[pd.DataFrame, DataQualityReport]:
    df = pd.read_csv(csv_path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return _prepare_data_with_report(df)


def quality_report_to_frame(report: DataQualityReport) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": list(report.as_dict().keys()),
            "value": list(report.as_dict().values()),
        }
    )


def apply_filters(
    df: pd.DataFrame,
    attendance_range: tuple[float, float] | None = None,
    hours_range: tuple[float, float] | None = None,
    genders: Iterable[str] | None = None,
    school_types: Iterable[str] | None = None,
    motivation_levels: Iterable[str] | None = None,
    internet_access_values: Iterable[str] | None = None,
    family_income_levels: Iterable[str] | None = None,
    parental_involvement_levels: Iterable[str] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()

    if attendance_range is not None:
        filtered = filtered[
            (filtered["Attendance"] >= attendance_range[0])
            & (filtered["Attendance"] <= attendance_range[1])
        ]

    if hours_range is not None:
        filtered = filtered[
            (filtered["Hours_Studied"] >= hours_range[0])
            & (filtered["Hours_Studied"] <= hours_range[1])
        ]

    if genders:
        filtered = filtered[filtered["Gender"].isin(genders)]
    if school_types:
        filtered = filtered[filtered["School_Type"].isin(school_types)]
    if motivation_levels:
        filtered = filtered[filtered["Motivation_Level"].isin(motivation_levels)]
    if internet_access_values:
        filtered = filtered[filtered["Internet_Access"].isin(internet_access_values)]
    if family_income_levels:
        filtered = filtered[filtered["Family_Income"].isin(family_income_levels)]
    if parental_involvement_levels:
        filtered = filtered[filtered["Parental_Involvement"].isin(parental_involvement_levels)]

    return filtered


def compute_kpis(df: pd.DataFrame) -> KPIBundle:
    if df.empty:
        return KPIBundle(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_students = int(len(df))
    avg_exam_score = float(df["Exam_Score"].mean())
    median_exam_score = float(df["Exam_Score"].median())
    pass_rate_percent = float(df["pass_flag"].mean() * 100)
    avg_attendance_percent = float(df["Attendance"].mean())
    avg_hours_studied = float(df["Hours_Studied"].mean())

    return KPIBundle(
        total_students=total_students,
        avg_exam_score=avg_exam_score,
        median_exam_score=median_exam_score,
        pass_rate_percent=pass_rate_percent,
        avg_attendance_percent=avg_attendance_percent,
        avg_hours_studied=avg_hours_studied,
    )


def score_distribution_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, as_index=False)
        .agg(
            avg_exam_score=("Exam_Score", "mean"),
            median_exam_score=("Exam_Score", "median"),
            pass_rate_percent=("pass_flag", lambda s: s.mean() * 100),
            student_count=("Exam_Score", "size"),
        )
        .sort_values("avg_exam_score", ascending=False)
    )


def study_attendance_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["study_band", "attendance_band"], observed=False, as_index=False)
        .agg(
            avg_exam_score=("Exam_Score", "mean"),
            pass_rate_percent=("pass_flag", lambda s: s.mean() * 100),
            student_count=("Exam_Score", "size"),
        )
        .sort_values(["study_band", "attendance_band"])
    )


def support_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Internet_Access", "Access_to_Resources", "Tutoring_Sessions"], as_index=False)
        .agg(
            avg_exam_score=("Exam_Score", "mean"),
            pass_rate_percent=("pass_flag", lambda s: s.mean() * 100),
            student_count=("Exam_Score", "size"),
        )
        .sort_values("avg_exam_score", ascending=False)
    )


def factor_importance(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    feature_cols = [
        "Hours_Studied",
        "Attendance",
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Sleep_Hours",
        "Previous_Scores",
        "Motivation_Level",
        "Internet_Access",
        "Tutoring_Sessions",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Physical_Activity",
        "Learning_Disabilities",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Gender",
    ]

    model_df = df[feature_cols + ["Exam_Score"]].copy()
    x = pd.get_dummies(model_df[feature_cols], drop_first=False)
    y = model_df["Exam_Score"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(x, y)

    importance = (
        pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
        "Exam_Score",
    ]
    return df[numeric_cols].corr(numeric_only=True)


def build_at_risk_profile(
    df: pd.DataFrame,
    pass_score_threshold: float = DEFAULT_RISK_THRESHOLDS["pass_score_threshold"],
    high_risk_score_threshold: float = DEFAULT_RISK_THRESHOLDS["high_risk_score_threshold"],
    attendance_threshold: float = DEFAULT_RISK_THRESHOLDS["attendance_threshold"],
    study_hours_threshold: float = DEFAULT_RISK_THRESHOLDS["study_hours_threshold"],
    previous_score_threshold: float = DEFAULT_RISK_THRESHOLDS["previous_score_threshold"],
    sleep_hours_min: float = DEFAULT_RISK_THRESHOLDS["sleep_hours_min"],
    tutoring_min_sessions: float = DEFAULT_RISK_THRESHOLDS["tutoring_min_sessions"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return student-level risk table and intervention summary using explicit thresholds."""
    risk = df.copy().reset_index(drop=True)
    risk["student_id"] = risk.index + 1

    risk["below_pass_score"] = risk["Exam_Score"] < pass_score_threshold
    risk["high_risk_score"] = risk["Exam_Score"] < high_risk_score_threshold
    risk["low_attendance"] = risk["Attendance"] < attendance_threshold
    risk["low_study_hours"] = risk["Hours_Studied"] < study_hours_threshold
    risk["weak_previous_scores"] = risk["Previous_Scores"] < previous_score_threshold
    risk["low_motivation"] = risk["Motivation_Level"] == "Low"
    risk["no_internet"] = risk["Internet_Access"] == "No"
    risk["low_access"] = risk["Access_to_Resources"] == "Low"
    risk["no_tutoring"] = risk["Tutoring_Sessions"] < tutoring_min_sessions
    risk["learning_disability"] = risk["Learning_Disabilities"] == "Yes"
    risk["short_sleep"] = risk["Sleep_Hours"] < sleep_hours_min

    flag_cols = [
        "below_pass_score",
        "high_risk_score",
        "low_attendance",
        "low_study_hours",
        "weak_previous_scores",
        "low_motivation",
        "no_internet",
        "low_access",
        "no_tutoring",
        "learning_disability",
        "short_sleep",
    ]
    risk["risk_score"] = risk[flag_cols].sum(axis=1)

    critical_mask = (
        risk["high_risk_score"]
        | (risk["below_pass_score"] & risk["low_attendance"])
        | (risk["risk_score"] >= 5)
    )
    high_mask = (~critical_mask) & (risk["below_pass_score"] | (risk["risk_score"] >= 4))
    moderate_mask = (~critical_mask) & (~high_mask) & (risk["risk_score"] >= 2)

    risk["risk_tier"] = "Low"
    risk.loc[moderate_mask, "risk_tier"] = "Moderate"
    risk.loc[high_mask, "risk_tier"] = "High"
    risk.loc[critical_mask, "risk_tier"] = "Critical"

    risk["primary_concerns"] = risk[flag_cols].apply(
        lambda row: ", ".join([col for col, val in row.items() if val]) if row.any() else "None",
        axis=1,
    )

    intervention_rows = [
        {
            "priority": "Critical",
            "intervention": "Intensive academic recovery plan",
            "trigger": f"Exam score < {high_risk_score_threshold}",
            "student_count": int(risk["high_risk_score"].sum()),
            "recommended_actions": "Daily remediation, parent meeting, and weekly progress checks.",
        },
        {
            "priority": "High",
            "intervention": "Attendance mentorship plan",
            "trigger": f"Attendance < {attendance_threshold}",
            "student_count": int(risk["low_attendance"].sum()),
            "recommended_actions": "Attendance tracking, mentor call, and family outreach within 7 days.",
        },
        {
            "priority": "High",
            "intervention": "Pass-threshold support group",
            "trigger": f"Exam score < {pass_score_threshold}",
            "student_count": int(risk["below_pass_score"].sum()),
            "recommended_actions": "Small-group instruction on weak topics and biweekly mock assessments.",
        },
        {
            "priority": "Medium",
            "intervention": "Study-habits coaching",
            "trigger": f"Hours studied < {study_hours_threshold}",
            "student_count": int(risk["low_study_hours"].sum()),
            "recommended_actions": "Structured study planner, accountability check-ins, and goal setting.",
        },
        {
            "priority": "Medium",
            "intervention": "Foundational skill reinforcement",
            "trigger": f"Previous scores < {previous_score_threshold}",
            "student_count": int(risk["weak_previous_scores"].sum()),
            "recommended_actions": "Bridge lessons and adaptive practice for prerequisite gaps.",
        },
        {
            "priority": "Medium",
            "intervention": "Tutoring enrollment drive",
            "trigger": f"Tutoring sessions < {tutoring_min_sessions}",
            "student_count": int(risk["no_tutoring"].sum()),
            "recommended_actions": "Enroll in tutoring and review participation after 2 weeks.",
        },
        {
            "priority": "Medium",
            "intervention": "Digital and resource equity support",
            "trigger": "No internet or low access to resources",
            "student_count": int((risk["no_internet"] | risk["low_access"]).sum()),
            "recommended_actions": "Provide device/library access and printed learning kits.",
        },
        {
            "priority": "Support",
            "intervention": "Motivation and wellbeing counseling",
            "trigger": "Low motivation or short sleep",
            "student_count": int((risk["low_motivation"] | risk["short_sleep"]).sum()),
            "recommended_actions": "Counselor sessions, sleep hygiene guidance, and motivation coaching.",
        },
        {
            "priority": "Support",
            "intervention": "Inclusive learning accommodations",
            "trigger": "Learning disabilities = Yes",
            "student_count": int(risk["learning_disability"].sum()),
            "recommended_actions": "Individualized supports and assistive accommodations in class.",
        },
    ]

    interventions = pd.DataFrame(intervention_rows)
    interventions["share_of_students_percent"] = (
        interventions["student_count"] / max(len(risk), 1) * 100
    ).round(2)
    interventions = interventions.sort_values("student_count", ascending=False).reset_index(drop=True)

    return risk, interventions


def at_risk_tier_summary(risk_df: pd.DataFrame) -> pd.DataFrame:
    return (
        risk_df.groupby("risk_tier", as_index=False)
        .agg(student_count=("student_id", "count"))
        .assign(
            share_percent=lambda d: (d["student_count"] / max(d["student_count"].sum(), 1) * 100).round(2)
        )
        .sort_values(
            "risk_tier",
            key=lambda s: s.map({"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}),
        )
    )


def format_number(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:,.2f}"
