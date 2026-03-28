from __future__ import annotations

import argparse
from pathlib import Path

import plotly.express as px

from bi_utils import (
    at_risk_tier_summary,
    build_at_risk_profile,
    compute_kpis,
    correlation_matrix,
    DEFAULT_RISK_THRESHOLDS,
    factor_importance,
    format_number,
    load_data_with_quality,
    quality_report_to_frame,
    score_distribution_by_group,
    study_attendance_summary,
    support_summary,
)


def create_report(data_path: str, output_dir: str) -> Path:
    df, quality_report = load_data_with_quality(data_path)
    kpis = compute_kpis(df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    school = score_distribution_by_group(df, "School_Type")
    gender = score_distribution_by_group(df, "Gender")
    motivation = score_distribution_by_group(df, "Motivation_Level")
    parental_education = score_distribution_by_group(df, "Parental_Education_Level")
    distance = score_distribution_by_group(df, "Distance_from_Home")
    support = support_summary(df)
    study_attendance = study_attendance_summary(df)
    factors = factor_importance(df, top_n=20)
    corr = correlation_matrix(df)
    risk_df, interventions = build_at_risk_profile(
        df,
        pass_score_threshold=DEFAULT_RISK_THRESHOLDS["pass_score_threshold"],
        high_risk_score_threshold=DEFAULT_RISK_THRESHOLDS["high_risk_score_threshold"],
        attendance_threshold=DEFAULT_RISK_THRESHOLDS["attendance_threshold"],
        study_hours_threshold=DEFAULT_RISK_THRESHOLDS["study_hours_threshold"],
        previous_score_threshold=DEFAULT_RISK_THRESHOLDS["previous_score_threshold"],
        sleep_hours_min=DEFAULT_RISK_THRESHOLDS["sleep_hours_min"],
        tutoring_min_sessions=DEFAULT_RISK_THRESHOLDS["tutoring_min_sessions"],
    )
    risk_tiers = at_risk_tier_summary(risk_df)
    at_risk_students = risk_df[risk_df["risk_tier"].isin(["Critical", "High", "Moderate"])].copy()
    priority_students = risk_df[risk_df["risk_tier"].isin(["Critical", "High"])].copy()
    priority_students = priority_students.sort_values(
        ["risk_tier", "risk_score", "Exam_Score"],
        ascending=[True, False, True],
    )
    quality = quality_report_to_frame(quality_report)

    school.to_csv(out / "school_type_performance.csv", index=False)
    gender.to_csv(out / "gender_performance.csv", index=False)
    motivation.to_csv(out / "motivation_performance.csv", index=False)
    parental_education.to_csv(out / "parental_education_performance.csv", index=False)
    distance.to_csv(out / "distance_performance.csv", index=False)
    support.to_csv(out / "support_summary.csv", index=False)
    study_attendance.to_csv(out / "study_attendance_summary.csv", index=False)
    factors.to_csv(out / "factor_importance.csv", index=False)
    corr.to_csv(out / "correlation_matrix.csv", index=True)
    at_risk_students.to_csv(out / "at_risk_students.csv", index=False)
    interventions.to_csv(out / "at_risk_interventions.csv", index=False)
    risk_tiers.to_csv(out / "at_risk_tier_summary.csv", index=False)
    priority_students.to_csv(out / "priority_students.csv", index=False)
    quality.to_csv(out / "data_quality_summary.csv", index=False)

    fig_hist = px.histogram(
        df,
        x="Exam_Score",
        color="School_Type",
        nbins=25,
        barmode="overlay",
        opacity=0.65,
        title="Exam Score Distribution by School Type",
    )
    fig_hist.write_html(out / "exam_score_distribution.html", include_plotlyjs="cdn")

    fig_importance = px.bar(
        factors.sort_values("importance", ascending=True).tail(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Top Predictive Features for Exam Score",
    )
    fig_importance.write_html(out / "feature_importance.html", include_plotlyjs="cdn")

    fig_support = px.scatter(
        df,
        x="Attendance",
        y="Exam_Score",
        color="Tutoring_Sessions",
        size="Hours_Studied",
        hover_data=["Motivation_Level", "Family_Income", "Internet_Access", "Teacher_Quality"],
        title="Attendance vs Exam Score (size = hours studied)",
    )
    fig_support.write_html(out / "attendance_vs_score.html", include_plotlyjs="cdn")

    top_school = school.iloc[0]["School_Type"] if not school.empty else "N/A"
    best_motivation = motivation.iloc[0]["Motivation_Level"] if not motivation.empty else "N/A"
    top_factor = factors.iloc[0]["feature"] if not factors.empty else "N/A"
    top_intervention = interventions.iloc[0]["intervention"] if not interventions.empty else "N/A"
    top_intervention_count = int(interventions.iloc[0]["student_count"]) if not interventions.empty else 0
    at_risk_count = int(len(at_risk_students))
    critical_count = int((risk_df["risk_tier"] == "Critical").sum())
    high_count = int((risk_df["risk_tier"] == "High").sum())

    report_md = out / "executive_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Student Performance Executive Report",
                "",
                "## KPI Snapshot",
                f"- Students analyzed: {format_number(kpis.total_students)}",
                f"- Average exam score: {kpis.avg_exam_score:.2f}",
                f"- Median exam score: {kpis.median_exam_score:.2f}",
                f"- Pass rate (>= 60): {kpis.pass_rate_percent:.2f}%",
                f"- Average attendance: {kpis.avg_attendance_percent:.2f}%",
                f"- Average hours studied: {kpis.avg_hours_studied:.2f}",
                "",
                "## Key Insights",
                f"- Highest-performing school type: {top_school}.",
                f"- Highest-performing motivation segment: {best_motivation}.",
                f"- Most influential modeled factor: {top_factor}.",
                "- Study-time and attendance combinations show clear score separation across student segments.",
                f"- Top intervention target: {top_intervention} ({top_intervention_count:,} students).",
                "",
                "## At-Risk Students (Threshold-Based)",
                f"- Pass score threshold: {DEFAULT_RISK_THRESHOLDS['pass_score_threshold']:.0f}",
                f"- High-risk score threshold: {DEFAULT_RISK_THRESHOLDS['high_risk_score_threshold']:.0f}",
                f"- Attendance threshold: {DEFAULT_RISK_THRESHOLDS['attendance_threshold']:.0f}%",
                f"- Study-hours threshold: {DEFAULT_RISK_THRESHOLDS['study_hours_threshold']:.0f}",
                f"- Previous-score threshold: {DEFAULT_RISK_THRESHOLDS['previous_score_threshold']:.0f}",
                f"- At-risk students (Moderate/High/Critical): {at_risk_count:,}",
                f"- Critical: {critical_count:,} | High: {high_count:,}",
                "",
                "### Intervention List",
                "1. Intensive academic recovery plan for students below the high-risk score threshold.",
                "2. Attendance mentorship plan for students below the attendance threshold.",
                "3. Pass-threshold support groups for students below minimum passing score.",
                "4. Study-habits coaching and tutoring enrollment for low-effort cohorts.",
                "5. Resource equity and wellbeing support for access and motivation barriers.",
                "",
                "## Data Quality Summary",
                f"- Input rows: {quality_report.input_rows:,}",
                f"- Output rows: {quality_report.output_rows:,}",
                f"- Rows dropped: {quality_report.rows_dropped:,}",
                f"- Missing critical rows: {quality_report.missing_critical_rows:,}",
                f"- Missing numeric cells: {quality_report.missing_numeric_cells:,}",
                f"- Missing categorical cells: {quality_report.missing_categorical_cells:,}",
                f"- Attendance out-of-range rows: {quality_report.attendance_out_of_range_rows:,}",
                f"- Exam score out-of-range rows: {quality_report.score_out_of_range_rows:,}",
                "",
                "## Generated Artifacts",
                "- school_type_performance.csv",
                "- gender_performance.csv",
                "- motivation_performance.csv",
                "- parental_education_performance.csv",
                "- distance_performance.csv",
                "- support_summary.csv",
                "- study_attendance_summary.csv",
                "- factor_importance.csv",
                "- correlation_matrix.csv",
                "- at_risk_students.csv",
                "- at_risk_interventions.csv",
                "- at_risk_tier_summary.csv",
                "- priority_students.csv",
                "- data_quality_summary.csv",
                "- exam_score_distribution.html",
                "- feature_importance.html",
                "- attendance_vs_score.html",
                "",
                "## Recommended Next Actions",
                "1. Assign counselors and teachers to all Critical-tier students this week.",
                "2. Launch attendance + tutoring intervention cycles for High-tier students.",
                "3. Monitor intervention conversion and score lift monthly using at_risk_interventions.csv.",
            ]
        ),
        encoding="utf-8",
    )

    return report_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a BI summary report from student performance data")
    parser.add_argument(
        "--data",
        default="StudentPerformanceFactors.csv",
        help="Path to student dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Directory where report outputs should be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = create_report(args.data, args.output)
    print(f"Report created: {report_path}")


if __name__ == "__main__":
    main()
