# 🎓 Student Career Readiness Predictor

A **production-ready** Streamlit application that predicts **junior/entry-level job readiness** and **tech-domain fit** for students using advanced machine learning with **honest data quality**.

Built with ❤️ using **Streamlit, Scikit-learn, XGBoost, LightGBM, SHAP, and Plotly**.

---

## ℹ️ About

This application leverages **6 advanced ML models** (Random Forest, Gradient Boosting, SVR, Linear Regression, XGBoost, LightGBM) with **SHAP explainability** to help educators and career counselors:

- Predict **junior/entry-level job readiness** (0-100% score)
- Match students to **tech domains** (Web Dev, Data, DevOps, Mobile)
- Provide **skill gap analysis** for target roles
- Explain predictions with **SHAP feature importance**
- Support **batch predictions** for multiple students

---

## 🎯 Purpose

To support **career counseling and junior-level hiring** by predicting **real job readiness** (not just willingness) from:

- **Academic Performance**: 10th, 12th, college marks
- **Engagement Metrics**: Certification courses, degree satisfaction
- **Study Habits**: Daily study time, stress management
- **Junior Profile**: GitHub activity, side projects, internships, tech skills
- **Real Outcomes**: Campus placement records with actual results

---

## 📊 Features

### 🎯 Core Prediction
- **Junior Readiness Score**: 0-100% with confidence intervals
- **Career Domain Matching**: Web Dev, Data, DevOps, Mobile
- **Skill Gap Analysis**: Per-role gap identification
- **SHAP Explainability**: Feature importance visualization

### 📁 Data Quality
- **450 honest records**: 235 self-reported + 215 real outcomes
- **No fabricated data**: Only honestly alignable datasets
- **Transparent provenance**: Clear dataset documentation

### 🤖 Machine Learning
- **6 model candidates**: Auto-selected by cross-validated MAE
- **Real outcomes**: Campus placement data (not synthetic)
- **Confidence intervals**: Residual-based uncertainty estimation

### 📊 Analytics
- Single student prediction with radar charts
- Batch CSV upload and download
- Aggregate analytics dashboard
- Model selection metrics display

---

## 🧠 Model Information

### 🎯 Current Model (Auto-Selected)
- **Algorithm**: SVR (Support Vector Regression)
- **Performance**: Test MAE = 12.89 (improved from 14.31)
- **Selection**: Cross-validated MAE across 6 candidates
- **Features**: 17 engineered features (12 raw + 5 derived)
- **Output**: Junior readiness % with confidence interval

### 📊 Model Candidates
1. **Random Forest Regressor** - Robust, handles non-linearity
2. **Gradient Boosting Regressor** - High accuracy, feature importance
3. **SVR** - Best performance for this dataset
4. **Linear Regression** - Baseline interpretability
5. **XGBoost** - Optimized gradient boosting
6. **LightGBM** - Fast training, memory efficient

### 🔬 Feature Engineering
- **Academic Relevance Score**: Weighted academic performance
- **Experience Relevance Score**: Project/internship proxy
- **Engagement Relevance Score**: Study habits balance
- **Domain Scores**: Web Dev, Data, DevOps, Mobile fit  

---

## 🖥️ Installation & Setup

**Modern Python project using UV for fast, reproducible dependency management.**

### 1. Clone the repository
```bash
git clone https://github.com/rupinmunjal/student-career-willingness-predictor.git
cd student-career-willingness-predictor
```

### 2. Install dependencies
```bash
uv sync
```
Creates `.venv/` with all dependencies from `pyproject.toml` (no `requirements.txt` needed).

### 3. Install new dependencies
```bash
uv add package_name  # Automatically updates pyproject.toml and uv.lock
```

---

## 🚀 Running the Application

### Containerized development
```bash
docker compose up --build
```

The Streamlit interface is available at [http://localhost:8501](http://localhost:8501).

The FastAPI service exposes health, prediction, batch prediction, skill-gap, and prediction
history endpoints. Interactive API documentation is available at
[http://localhost:8000/docs](http://localhost:8000/docs) when the API service is running.

Local development stores prediction history in `student_career.db`. Set `DATABASE_URL` to an
async SQLAlchemy PostgreSQL URL for a shared database. Schema changes are managed with Alembic:

```bash
uv run alembic upgrade head
```

### Start the app
```bash
uv run streamlit run app.py
```

### Access the app
Open [http://localhost:8501](http://localhost:8501) in your browser.

### App Views
- **Prediction**: Individual student analysis — readiness gauge, domain-fit radar chart,
  confidence interval, skill-gap analysis for a target role, and top feature importances.
- **Batch Upload**: Process many students via CSV upload, view results, download predictions,
  and inspect batch metrics (mean readiness, level distribution, best-domain distribution).
- **Analytics Dashboard**: Training-data provenance, cross-validation scores, and global
  feature importance for the selected model.
- **About Page**: Model details and data sources.

---

## 📂 Project Structure

```
student-career-willingness-predictor/
├── app.py                          # Main Streamlit app (4 views: Prediction, Batch Upload, Analytics, About)
├── pyproject.toml                  # Modern dependency management
├── uv.lock                         # Locked dependency versions
├── .gitignore                      # File exclusion patterns
├── data/                           # Quality datasets (450 records)
│   ├── Student Attitude and Behavior.csv  # 235 self-reported surveys
│   └── Placement_Data_Full_Class.csv    # 215 real placement outcomes
├── models/                         # Trained models
│   └── student_career_willingness_model.pkl  # Auto-selected best model
├── src/                            # Modular source code
│   ├── __init__.py                     # Package imports
│   ├── data_pipeline.py               # Multi-dataset pipeline + canonical features
│   └── student_career_willingness_predictor.py  # Core predictor (train/predict/batch/skill-gap/SHAP)
├── tests/                          # Test suite
│   ├── test_predictor.py              # Unit tests (single, batch, skill-gap)
│   ├── test_visual_output.py          # Visual output generation checks
│   └── validate_pipeline.py          # Data validation
└── README.md                       # This file
```

---

## 📈 Interpretation Guide

### Junior Readiness Score
- **85–100%**: High readiness → Strong candidate for junior roles
- **70–84%**: Moderate readiness → Good fit with minor skill gaps
- **40–69%**: Developing readiness → Needs targeted preparation
- **0–39%**: Low readiness → Significant support required

### Career Domain Match
Radar chart shows fit across:
- **Web Dev**: Frontend/backend development
- **Data**: Analytics, ML, data engineering  
- **DevOps**: Cloud, CI/CD, infrastructure
- **Mobile**: iOS/Android development

### Skill Gap Analysis
Identifies specific areas needing improvement for target role.

---

## 📊 Data Sources

### Honest Quality Data (450 Records)

| Dataset | Records | Target Type | New Signal |
|---------|---------|-------------|------------|
| **Student Attitude and Behavior** | 235 | Self-reported willingness | Baseline academic & engagement |
| **Campus Placement Records** | 215 | REAL placement outcomes | Professional experience, test scores |

### Previously Rejected Datasets

❌ **Maths Performance** - No career target
❌ **Portuguese Performance** - No career target  
❌ **Job Placement (old)** - Fabricated targets
❌ **Students Performance** - No career outcomes

### Quality Principles

✅ **Real Outcomes**: Only datasets with genuine career-related targets
✅ **Honest Alignment**: Features must have true semantic equivalence
✅ **No Fabrication**: No invented targets or synthetic padding
✅ **Transparent Provenance**: Clear documentation of each dataset's contribution

## 🧪 Testing

### Test Coverage
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_api.py

# Static analysis and formatting
uv run ruff check .
uv run mypy src/ api.py app.py
uv run pre-commit run --all-files
```

### Test Results
- ✅ Pytest coverage for the API, data pipeline, feature engineering, and model persistence
- ✅ Edge case validation
- ✅ Batch prediction testing
- ✅ Pipeline integrity checks

## 🤝 Contributing

**Quality-first development welcome!** Pull requests should:

### ✅ Acceptance Criteria
- Add **real value** (not just volume)
- Maintain **honest data alignment** (no fabrication)
- Include **tests** for new functionality
- Update **documentation**
- Preserve **backward compatibility**

### 🎯 Suggested Enhancements
- Additional **quality datasets** with real outcomes
- **Model explainability** improvements
- **Performance optimization** for large batches
- **Deployment** scripts for cloud platforms

### 🚀 Process
1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

**No fabricated data or synthetic padding will be accepted.**

## 📦 Deployment

### Streamlit Cloud
```bash
# Deploy directly from GitHub
# 1. Push to main branch
# 2. Connect repository to Streamlit Cloud
# 3. Select Python 3.9+ environment
# 4. Deploy!
```

### Local Production
```bash
# Build for production
gunicorn app:app --bind 0.0.0.0:8501 --workers 4

# Or use Streamlit directly
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📄 License

**MIT License** - Open source for education and career development.

**Attribution**: Please cite this repository if used in research or commercial applications.

**No Warranty**: Provided "AS IS" without warranty of any kind.
