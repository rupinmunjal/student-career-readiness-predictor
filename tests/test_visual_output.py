import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src import StudentCareerWillingnessPredictor


def test_visual_outputs():
    print("🎨 Testing visual output generation...")
    
    predictor = StudentCareerWillingnessPredictor()
    predictor.load_model("models/student_career_willingness_model.pkl")
    
    test_input = {
        "certification_course": 1.0,
        "tenth_mark": 85.0,
        "twelfth_mark": 80.0,
        "college_mark": 82.0,
        "daily_studying_time": 3.0,
        "likes_degree": 1.0,
        "travel_time": 1.0,
        "stress_level": 2.0,
        "part_time_job": 0.0,
        "github_activity_proxy": 6.0,
        "side_projects_count": 4.0,
        "internship_experience": 1.0,
        "tech_skills_self_rating": 7.0,
        "languages_known_count": 4.0,
    }
    
    profile = predictor.predict_profile(test_input)
    
    print(f"✅ Prediction: {profile['readiness_score']:.1f}% readiness")
    print(f"🎯 Best domain: {profile['best_domain']}")
    print(f"📊 Domain scores: {profile['domain_scores']}")
    
    from app import _render_radar_chart
    radar_fig = _render_radar_chart(profile['domain_scores'])
    print("✅ Radar chart generated successfully")
    print(f"   Chart has {len(radar_fig.data)} traces")
    print(f"   Layout height: {radar_fig.layout.height}px")
    
    shap_df = predictor.get_shap_feature_importance(pd.DataFrame([test_input]))
    if shap_df is not None:
        fig = px.bar(
            shap_df.head(8),
            x="importance",
            y="feature",
            orientation="h",
            title="SHAP Feature Impact (Top 8)",
        )
        fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"})
        print("✅ SHAP chart generated successfully")
        print(f"   Top features: {list(shap_df.head(3)['feature'].values)}")
    else:
        print("⚠️  SHAP not available")
    
    batch_results = predictor.predict_batch(pd.DataFrame([test_input, test_input]))
    fig_hist = px.histogram(
        batch_results,
        x="predicted_junior_readiness",
        nbins=20,
        title="Distribution of Junior Readiness Scores",
    )
    print("✅ Histogram generated successfully")
    print(f"   Data range: {batch_results['predicted_junior_readiness'].min():.1f}% - {batch_results['predicted_junior_readiness'].max():.1f}%")
    
    print("\n🎉 All visual outputs generated successfully!")


if __name__ == "__main__":
    test_visual_outputs()