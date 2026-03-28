# Student Performance BI Project

End-to-end Business Intelligence project built on StudentPerformanceFactors.csv with:

- Interactive dashboard (Streamlit + Plotly)
- Automated executive reporting pipeline
- Reusable analytics code for KPIs, segmentation, and factor analysis

## Project Goals

- Track exam-score outcomes and pass-rate performance
- Compare performance by school type, motivation, and demographics
- Understand how attendance, study habits, and support conditions shape outcomes
- Identify high-impact academic factors using model-based feature importance

## Project Structure

```text
.
├── StudentPerformanceFactors.csv
├── bi_utils.py
├── dashboard.py
├── generate_report.py
├── requirements.txt
└── README.md
```

## Dataset Fields Used

- Hours_Studied
- Attendance
- Parental_Involvement
- Access_to_Resources
- Extracurricular_Activities
- Sleep_Hours
- Previous_Scores
- Motivation_Level
- Internet_Access
- Tutoring_Sessions
- Family_Income
- Teacher_Quality
- School_Type
- Peer_Influence
- Physical_Activity
- Learning_Disabilities
- Parental_Education_Level
- Distance_from_Home
- Gender
- Exam_Score

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Interactive Dashboard

```bash
streamlit run dashboard.py
```

The dashboard includes:

- Sidebar filters for attendance, study hours, and key categorical segments
- Data quality summary panel (missing, dropped, and out-of-range records)
- Executive KPI row for student outcomes
- Overview visuals for score distribution and outcome patterns
- Learning-factors tab with correlation and feature-importance analysis
- Support/context tab covering resource, tutoring, family, and contextual effects
- At-risk tab with threshold controls, risk tiers, and intervention target lists
- Filtered data export to CSV

## Generate Automated Report

```bash
python generate_report.py --data StudentPerformanceFactors.csv --output reports
```

This creates:

- reports/executive_report.md
- CSV summary tables for BI reporting
- reports/at_risk_students.csv and reports/at_risk_interventions.csv
- reports/data_quality_summary.csv
- HTML charts for shareable visuals, including risk-tier and intervention-target charts

## KPI Definitions

- Students: Number of rows after quality filtering
- Average Exam Score: Mean of Exam_Score
- Median Exam Score: Median of Exam_Score
- Pass Rate: Share of students with Exam_Score >= 60
- Average Attendance: Mean of Attendance
- Average Hours Studied: Mean of Hours_Studied

## Business Questions Answered

- Which student segments perform best and worst?
- How strongly do attendance and study hours relate to exam performance?
- Which support conditions (internet, resources, tutoring) are linked to better outcomes?
- Which factors are most predictive of exam score?

## Next BI Enhancements

1. Add confidence intervals for segment-level performance comparisons.
2. Add model explainability views (e.g., SHAP) for factor-level interpretation.
3. Add unit tests for KPI and transformation functions.
4. Add time-aware tracking once a dated student-progress dataset is available.
