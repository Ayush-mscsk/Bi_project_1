from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bi_utils import (
    apply_filters,
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

st.set_page_config(page_title="Student Performance Intelligence Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def get_data(path: str):
    return load_data_with_quality(path)


def build_sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    attendance_min = float(df["Attendance"].min())
    attendance_max = float(df["Attendance"].max())
    hours_min = float(df["Hours_Studied"].min())
    hours_max = float(df["Hours_Studied"].max())

    attendance_range = st.sidebar.slider(
        "Attendance range (%)",
        min_value=attendance_min,
        max_value=attendance_max,
        value=(attendance_min, attendance_max),
    )

    hours_range = st.sidebar.slider(
        "Hours studied range",
        min_value=hours_min,
        max_value=hours_max,
        value=(hours_min, hours_max),
    )

    selected_genders = st.sidebar.multiselect(
        "Gender",
        options=sorted(df["Gender"].dropna().unique().tolist()),
        default=sorted(df["Gender"].dropna().unique().tolist()),
    )

    selected_school_types = st.sidebar.multiselect(
        "School type",
        options=sorted(df["School_Type"].dropna().unique().tolist()),
        default=sorted(df["School_Type"].dropna().unique().tolist()),
    )

    selected_motivation = st.sidebar.multiselect(
        "Motivation level",
        options=sorted(df["Motivation_Level"].dropna().unique().tolist()),
        default=sorted(df["Motivation_Level"].dropna().unique().tolist()),
    )

    selected_internet = st.sidebar.multiselect(
        "Internet access",
        options=sorted(df["Internet_Access"].dropna().unique().tolist()),
        default=sorted(df["Internet_Access"].dropna().unique().tolist()),
    )

    selected_family_income = st.sidebar.multiselect(
        "Family income",
        options=sorted(df["Family_Income"].dropna().unique().tolist()),
        default=sorted(df["Family_Income"].dropna().unique().tolist()),
    )

    selected_parental_involvement = st.sidebar.multiselect(
        "Parental involvement",
        options=sorted(df["Parental_Involvement"].dropna().unique().tolist()),
        default=sorted(df["Parental_Involvement"].dropna().unique().tolist()),
    )

    return (
        attendance_range,
        hours_range,
        selected_genders,
        selected_school_types,
        selected_motivation,
        selected_internet,
        selected_family_income,
        selected_parental_involvement,
    )


def render_kpi_row(df: pd.DataFrame) -> None:
    kpis = compute_kpis(df)
    top1, top2, top3 = st.columns(3)
    bot1, bot2, bot3 = st.columns(3)

    top1.metric("Students", format_number(kpis.total_students))
    top2.metric("Average Exam Score", f"{kpis.avg_exam_score:.2f}")
    top3.metric("Pass Rate", f"{kpis.pass_rate_percent:.1f}%")
    bot1.metric("Median Exam Score", f"{kpis.median_exam_score:.2f}")
    bot2.metric("Avg Attendance", f"{kpis.avg_attendance_percent:.1f}%")
    bot3.metric("Avg Hours Studied", f"{kpis.avg_hours_studied:.1f}")


def overview_insights(df: pd.DataFrame) -> list[str]:
    insights = []

    school = score_distribution_by_group(df, "School_Type")
    if len(school) >= 2:
        top_school = school.iloc[0]
        low_school = school.iloc[-1]
        gap = top_school["avg_exam_score"] - low_school["avg_exam_score"]
        insights.append(
            f"School type gap: {top_school['School_Type']} averages {top_school['avg_exam_score']:.1f} vs "
            f"{low_school['School_Type']} at {low_school['avg_exam_score']:.1f} (gap {gap:.1f})."
        )

    corr_attendance = df["Attendance"].corr(df["Exam_Score"])
    corr_hours = df["Hours_Studied"].corr(df["Exam_Score"])
    insights.append(
        f"Correlation snapshot: Attendance vs score = {corr_attendance:.2f}, "
        f"Hours studied vs score = {corr_hours:.2f}."
    )

    by_internet = score_distribution_by_group(df, "Internet_Access")
    if len(by_internet) >= 2:
        with_net = by_internet.loc[by_internet["Internet_Access"] == "Yes"]
        without_net = by_internet.loc[by_internet["Internet_Access"] == "No"]
        if not with_net.empty and not without_net.empty:
            gap = with_net.iloc[0]["avg_exam_score"] - without_net.iloc[0]["avg_exam_score"]
            insights.append(f"Digital access gap: students with internet score {gap:+.1f} points vs those without.")

    return insights


def factors_insights(df: pd.DataFrame) -> list[str]:
    insights = []

    importance = factor_importance(df, top_n=5)
    if not importance.empty:
        top_factor = importance.iloc[0]
        insights.append(
            f"Most influential factor from model: {top_factor['feature']} (importance {top_factor['importance']:.3f})."
        )

    motivation = score_distribution_by_group(df, "Motivation_Level")
    if len(motivation) >= 2:
        high = motivation.iloc[0]
        low = motivation.iloc[-1]
        insights.append(
            f"Motivation impact: {high['Motivation_Level']} students outperform {low['Motivation_Level']} "
            f"by {high['avg_exam_score'] - low['avg_exam_score']:.1f} points on average."
        )

    disabilities = score_distribution_by_group(df, "Learning_Disabilities")
    if len(disabilities) >= 2:
        yes_row = disabilities.loc[disabilities["Learning_Disabilities"] == "Yes"]
        no_row = disabilities.loc[disabilities["Learning_Disabilities"] == "No"]
        if not yes_row.empty and not no_row.empty:
            diff = no_row.iloc[0]["pass_rate_percent"] - yes_row.iloc[0]["pass_rate_percent"]
            insights.append(f"Pass-rate difference by disability status: {diff:+.1f} percentage points (No - Yes).")

    return insights


def support_insights(df: pd.DataFrame) -> list[str]:
    insights = []

    support = support_summary(df)
    if not support.empty:
        best = support.iloc[0]
        insights.append(
            f"Best support mix: Internet={best['Internet_Access']}, Resources={best['Access_to_Resources']}, "
            f"Tutoring Sessions={int(best['Tutoring_Sessions'])} with avg score {best['avg_exam_score']:.1f}."
        )

    parent_edu = score_distribution_by_group(df, "Parental_Education_Level")
    if len(parent_edu) >= 2:
        insights.append(
            f"Parental education spread: top group ({parent_edu.iloc[0]['Parental_Education_Level']}) leads by "
            f"{parent_edu.iloc[0]['avg_exam_score'] - parent_edu.iloc[-1]['avg_exam_score']:.1f} points."
        )

    distance = score_distribution_by_group(df, "Distance_from_Home")
    if len(distance) >= 2:
        insights.append(
            f"Commute effect: {distance.iloc[0]['Distance_from_Home']} students have best pass rate "
            f"at {distance.iloc[0]['pass_rate_percent']:.1f}%."
        )

    return insights


def display_insights(insights: list[str]) -> None:
    if insights:
        st.markdown("**Key Insights**")
        for insight in insights:
            st.markdown(f"- {insight}")


def render_overview_tab(df: pd.DataFrame) -> None:
    st.subheader("Student Outcome Overview")

    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            df,
            x="Exam_Score",
            color="School_Type",
            nbins=25,
            title="Exam Score Distribution by School Type",
            barmode="overlay",
            opacity=0.65,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        motivation = score_distribution_by_group(df, "Motivation_Level")
        fig_motivation = px.bar(
            motivation,
            x="Motivation_Level",
            y="avg_exam_score",
            color="pass_rate_percent",
            title="Average Score by Motivation Level",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_motivation, use_container_width=True)

    fig_scatter = px.scatter(
        df,
        x="Attendance",
        y="Exam_Score",
        color="Tutoring_Sessions",
        size="Hours_Studied",
        hover_data=["Previous_Scores", "Sleep_Hours", "Gender", "Family_Income"],
        title="Attendance vs Exam Score (size = study hours, color = tutoring sessions)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    display_insights(overview_insights(df))


def render_factors_tab(df: pd.DataFrame) -> None:
    st.subheader("Learning Factors Deep Dive")

    c1, c2 = st.columns(2)

    with c1:
        corr = correlation_matrix(df)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Numeric Factor Correlation Matrix",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with c2:
        importance = factor_importance(df, top_n=12)
        fig_importance = px.bar(
            importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Predictive Features for Exam Score",
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    factor_options = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Motivation_Level",
        "Family_Income",
        "Teacher_Quality",
        "Peer_Influence",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Learning_Disabilities",
        "Gender",
    ]
    selected_factor = st.selectbox("Compare score distribution by factor", factor_options)

    fig_box = px.box(
        df,
        x=selected_factor,
        y="Exam_Score",
        color=selected_factor,
        title=f"Exam Score Distribution by {selected_factor}",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    display_insights(factors_insights(df))


def render_support_tab(df: pd.DataFrame) -> None:
    st.subheader("Support, Access, and Context")

    c1, c2 = st.columns(2)

    with c1:
        resource_income = (
            df.groupby(["Family_Income", "Access_to_Resources"], as_index=False)
            .agg(avg_exam_score=("Exam_Score", "mean"))
            .pivot(index="Family_Income", columns="Access_to_Resources", values="avg_exam_score")
            .fillna(0)
        )
        fig_heatmap = px.imshow(
            resource_income,
            text_auto=True,
            aspect="auto",
            title="Average Exam Score Heatmap: Family Income x Access to Resources",
            color_continuous_scale="YlOrBr",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with c2:
        tutoring = (
            df.groupby("Tutoring_Sessions", as_index=False)
            .agg(avg_exam_score=("Exam_Score", "mean"), pass_rate_percent=("pass_flag", lambda s: s.mean() * 100))
            .sort_values("Tutoring_Sessions")
        )
        fig_tutoring = px.line(
            tutoring,
            x="Tutoring_Sessions",
            y=["avg_exam_score", "pass_rate_percent"],
            markers=True,
            title="Tutoring Sessions vs Score and Pass Rate",
        )
        st.plotly_chart(fig_tutoring, use_container_width=True)

    support = study_attendance_summary(df)
    support_matrix = support.pivot(index="study_band", columns="attendance_band", values="avg_exam_score").fillna(0)
    fig_support = px.imshow(
        support_matrix,
        text_auto=True,
        aspect="auto",
        title="Average Score by Study Band x Attendance Band",
        color_continuous_scale="Teal",
    )
    st.plotly_chart(fig_support, use_container_width=True)

    fig_sunburst = px.sunburst(
        df,
        path=["Parental_Education_Level", "Peer_Influence", "Distance_from_Home"],
        values="Exam_Score",
        color="Exam_Score",
        color_continuous_scale="Viridis",
        title="Contextual Performance Map",
    )
    st.plotly_chart(fig_sunburst, use_container_width=True)

    st.divider()
    display_insights(support_insights(df))


def render_export_tab(df: pd.DataFrame) -> None:
    st.subheader("Data Export")
    st.write("Download the filtered student dataset currently in view.")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered CSV",
        data=csv_data,
        file_name="filtered_student_performance_data.csv",
        mime="text/csv",
    )

    st.dataframe(df.head(100), use_container_width=True)


def render_at_risk_tab(df: pd.DataFrame) -> None:
    st.subheader("At-Risk Students")
    st.caption("Threshold-based risk targeting with intervention-ready student lists")

    t1, t2, t3 = st.columns(3)
    with t1:
        pass_score_threshold = st.slider(
            "Pass score threshold",
            min_value=40.0,
            max_value=75.0,
            value=float(DEFAULT_RISK_THRESHOLDS["pass_score_threshold"]),
            step=1.0,
        )
        high_risk_score_threshold = st.slider(
            "High-risk score threshold",
            min_value=30.0,
            max_value=65.0,
            value=float(DEFAULT_RISK_THRESHOLDS["high_risk_score_threshold"]),
            step=1.0,
        )
    with t2:
        attendance_threshold = st.slider(
            "Attendance threshold (%)",
            min_value=60.0,
            max_value=95.0,
            value=float(DEFAULT_RISK_THRESHOLDS["attendance_threshold"]),
            step=1.0,
        )
        study_hours_threshold = st.slider(
            "Study-hours threshold",
            min_value=5.0,
            max_value=25.0,
            value=float(DEFAULT_RISK_THRESHOLDS["study_hours_threshold"]),
            step=1.0,
        )
    with t3:
        previous_score_threshold = st.slider(
            "Previous-score threshold",
            min_value=40.0,
            max_value=85.0,
            value=float(DEFAULT_RISK_THRESHOLDS["previous_score_threshold"]),
            step=1.0,
        )
        sleep_hours_min = st.slider(
            "Minimum sleep-hours",
            min_value=4.0,
            max_value=9.0,
            value=float(DEFAULT_RISK_THRESHOLDS["sleep_hours_min"]),
            step=0.5,
        )

    tutoring_min_sessions = st.slider(
        "Minimum tutoring sessions",
        min_value=0,
        max_value=4,
        value=int(DEFAULT_RISK_THRESHOLDS["tutoring_min_sessions"]),
        step=1,
    )

    risk_df, interventions = build_at_risk_profile(
        df,
        pass_score_threshold=pass_score_threshold,
        high_risk_score_threshold=high_risk_score_threshold,
        attendance_threshold=attendance_threshold,
        study_hours_threshold=study_hours_threshold,
        previous_score_threshold=previous_score_threshold,
        sleep_hours_min=sleep_hours_min,
        tutoring_min_sessions=float(tutoring_min_sessions),
    )
    tier_summary = at_risk_tier_summary(risk_df)

    at_risk_total = int(risk_df["risk_tier"].isin(["Critical", "High", "Moderate"]).sum())
    critical_total = int((risk_df["risk_tier"] == "Critical").sum())
    high_total = int((risk_df["risk_tier"] == "High").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("At-Risk Students", format_number(at_risk_total))
    k2.metric("Critical", format_number(critical_total))
    k3.metric("High", format_number(high_total))
    k4.metric("Average Risk Score", f"{risk_df['risk_score'].mean():.2f}")

    c1, c2 = st.columns(2)
    with c1:
        fig_tier = px.bar(
            tier_summary,
            x="risk_tier",
            y="student_count",
            color="share_percent",
            title="Risk Tier Distribution",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with c2:
        top_interventions = interventions.head(6)
        fig_intervention = px.bar(
            top_interventions.sort_values("student_count", ascending=True),
            x="student_count",
            y="intervention",
            color="priority",
            orientation="h",
            title="Top Intervention Targets",
        )
        st.plotly_chart(fig_intervention, use_container_width=True)

    st.markdown("**Threshold-Based Intervention List**")
    st.dataframe(
        interventions[
            [
                "priority",
                "intervention",
                "trigger",
                "student_count",
                "share_of_students_percent",
                "recommended_actions",
            ]
        ],
        use_container_width=True,
    )

    st.markdown("**Priority Student List (Critical and High)**")
    priority_students = risk_df[risk_df["risk_tier"].isin(["Critical", "High"])].copy()
    priority_students = priority_students.sort_values(["risk_tier", "risk_score", "Exam_Score"], ascending=[True, False, True])
    st.dataframe(
        priority_students[
            [
                "student_id",
                "risk_tier",
                "risk_score",
                "Exam_Score",
                "Attendance",
                "Hours_Studied",
                "Tutoring_Sessions",
                "Motivation_Level",
                "Internet_Access",
                "Access_to_Resources",
                "primary_concerns",
            ]
        ].head(250),
        use_container_width=True,
    )


def main() -> None:
    st.title("Student Performance Intelligence Dashboard")
    st.caption("Interactive analytics for achievement, study behavior, access, and academic context")

    data_path = st.sidebar.text_input("Dataset path", value="StudentPerformanceFactors.csv")

    try:
        df, quality_report = get_data(data_path)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    with st.sidebar.expander("Data quality summary", expanded=False):
        st.caption("Validation snapshot from ingestion step")
        st.write(f"Input rows: {quality_report.input_rows:,}")
        st.write(f"Output rows: {quality_report.output_rows:,}")
        st.write(f"Rows dropped: {quality_report.rows_dropped:,}")
        st.write(f"Missing critical rows: {quality_report.missing_critical_rows:,}")
        st.write(f"Missing numeric cells: {quality_report.missing_numeric_cells:,}")
        st.write(f"Missing categorical cells: {quality_report.missing_categorical_cells:,}")
        st.write(f"Attendance out-of-range rows: {quality_report.attendance_out_of_range_rows:,}")
        st.write(f"Exam score out-of-range rows: {quality_report.score_out_of_range_rows:,}")
        st.write(f"Negative studied-hours rows: {quality_report.negative_hours_rows:,}")
        st.write(f"Negative sleep-hours rows: {quality_report.negative_sleep_rows:,}")
        st.write(f"Negative tutoring rows: {quality_report.negative_tutoring_rows:,}")
        st.dataframe(quality_report_to_frame(quality_report), use_container_width=True)

    (
        attendance_range,
        hours_range,
        genders,
        school_types,
        motivation_levels,
        internet_access_values,
        family_income_levels,
        parental_involvement_levels,
    ) = build_sidebar_filters(df)

    filtered_df = apply_filters(
        df,
        attendance_range=attendance_range,
        hours_range=hours_range,
        genders=genders,
        school_types=school_types,
        motivation_levels=motivation_levels,
        internet_access_values=internet_access_values,
        family_income_levels=family_income_levels,
        parental_involvement_levels=parental_involvement_levels,
    )

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        return

    render_kpi_row(filtered_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Learning Factors",
        "Support and Context",
        "At-Risk Students",
        "Export",
    ])

    with tab1:
        render_overview_tab(filtered_df)
    with tab2:
        render_factors_tab(filtered_df)
    with tab3:
        render_support_tab(filtered_df)
    with tab4:
        render_at_risk_tab(filtered_df)
    with tab5:
        render_export_tab(filtered_df)


if __name__ == "__main__":
    main()
