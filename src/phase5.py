import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ====================================
# PAGE CONFIGURATION
# ====================================
st.set_page_config(
    page_title="HR Analytics BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
    .improvement-positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .improvement-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================
# LOAD MODELS AND DATA
# ====================================
@st.cache_resource
def load_models_and_data():
    """Load trained models and metadata"""
    try:
        best_model = joblib.load('data/best_model.pkl')
        metadata = joblib.load('data/model_metadata.pkl')
        
        # Load baseline and tuned results
        baseline_results = pd.read_csv('data/baseline_model_comparison.csv')
        model_results = pd.read_csv('data/model_comparison.csv')
        comparison_results = pd.read_csv('data/baseline_vs_tuned_comparison.csv')
        tuning_results = pd.read_csv('data/hyperparameter_tuning_results.csv')
        feature_importance = pd.read_csv('data/feature_importance.csv')
        best_params = pd.read_csv('data/best_hyperparameters.csv')
        
        return (best_model, metadata, baseline_results, model_results, 
                comparison_results, tuning_results, feature_importance, best_params)
    except FileNotFoundError as e:
        st.error(f"⚠️ Error loading files: {e}")
        st.error("Please run Phases 1-4 first!")
        st.stop()

@st.cache_data
def load_sample_data():
    """Load sample dataset"""
    try:
        df = pd.read_csv('data/hr_ibm.csv')
        return df
    except FileNotFoundError:
        return None

# Load resources
(best_model, metadata, baseline_results, model_results, 
 comparison_results, tuning_results, feature_importance, best_params) = load_models_and_data()
sample_data = load_sample_data()

# ====================================
# SIDEBAR NAVIGATION
# ====================================
st.sidebar.markdown("# 🎯 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Data Explorer", "🤖 Model Insights", 
     "⚡ Hyperparameter Tuning", "🔮 Make Predictions", "📈 Visualizations"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Upload Your Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with the same structure as the training data"
)

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded: {user_data.shape[0]} rows × {user_data.shape[1]} columns")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {e}")
        user_data = None
else:
    user_data = None

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")

# Calculate improvement
baseline_best = baseline_results[baseline_results['Model'] == metadata['best_model_name']]['ROC-AUC'].values[0]
tuned_best = metadata['metrics']['ROC-AUC']
improvement = tuned_best - baseline_best
improvement_pct = (improvement / baseline_best) * 100

st.sidebar.info(f"""
**HR Analytics BI Dashboard**

**Best Model:** {metadata['best_model_name']}

**Performance:**
- Baseline: {baseline_best:.3f}
- Tuned: {tuned_best:.3f}
- Improvement: +{improvement:.3f} ({improvement_pct:+.1f}%)

**Optimal Threshold:** {metadata.get('optimal_threshold', 0.5):.3f}
""")

# ====================================
# PAGE 1: HOME
# ====================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📊 HR Analytics Business Intelligence Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the HR Analytics BI Platform
    
    This comprehensive dashboard leverages machine learning with **hyperparameter optimization** to provide 
    actionable insights into employee attrition. Navigate through different sections to explore data, 
    analyze models, and make predictions.
    """)
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🏆 Best Model",
            value=metadata['best_model_name'].split()[0],
            delta="After Tuning"
        )
    
    with col2:
        st.metric(
            label="🎯 Baseline ROC-AUC",
            value=f"{baseline_best:.3f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="⚡ Tuned ROC-AUC",
            value=f"{tuned_best:.3f}",
            delta=f"+{improvement:.3f}"
        )
    
    with col4:
        st.metric(
            label="📈 Improvement",
            value=f"{improvement_pct:+.1f}%",
            delta="From Tuning"
        )
    
    with col5:
        st.metric(
            label="📊 Features",
            value=metadata['n_features'],
            delta="Predictive Variables"
        )
    
    st.markdown("---")
    
    # Project Overview
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown('<p class="sub-header">🎯 Project Workflow</p>', unsafe_allow_html=True)
        st.markdown("""
        **Phase 1: Dataset Selection** ✅
        - Compared 2 HR datasets
        - Selected IBM HR Analytics (1,470 employees)
        
        **Phase 2: Data Preprocessing** ✅
        - Cleaned and encoded data
        - Created 6 engineered features
        - StandardScaler normalization
        
        **Phase 3: Model Development** ✅
        - **Step 1:** Trained 4 baseline models
        - **Step 2:** GridSearchCV optimization (594 combinations)
        - **Step 3:** Evaluated improvement
        
        **Phase 4: Advanced Visualization** ✅
        - 17 comprehensive visualizations
        - Baseline vs Tuned analysis
        
        **Phase 5: Streamlit Deployment** ✅
        - Interactive web application
        - Real-time predictions
        """)
    
    with col_right:
        st.markdown('<p class="sub-header">📈 Baseline vs Tuned Performance</p>', unsafe_allow_html=True)
        
        # Comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=comparison_results['Model'],
            y=comparison_results['ROC-AUC_Baseline'],
            marker_color='#e74c3c',
            text=comparison_results['ROC-AUC_Baseline'].round(3),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Tuned',
            x=comparison_results['Model'],
            y=comparison_results['ROC-AUC_Tuned'],
            marker_color='#2ecc71',
            text=comparison_results['ROC-AUC_Tuned'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Model Performance: Before vs After Tuning',
            xaxis_title='Model',
            yaxis_title='ROC-AUC Score',
            barmode='group',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Technologies Used
    st.markdown('<p class="sub-header">🔧 Technologies & Methodology</p>', unsafe_allow_html=True)
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.markdown("**Data Processing**")
        st.markdown("- Pandas")
        st.markdown("- NumPy")
        st.markdown("- Scikit-learn")
    
    with tech_col2:
        st.markdown("**Machine Learning**")
        st.markdown("- 4 Classification Models")
        st.markdown("- GridSearchCV Tuning")
        st.markdown("- 5-Fold Cross-Validation")
    
    with tech_col3:
        st.markdown("**Visualization**")
        st.markdown("- Plotly (Interactive)")
        st.markdown("- Matplotlib (Static)")
        st.markdown("- Seaborn")
    
    with tech_col4:
        st.markdown("**Deployment**")
        st.markdown("- Streamlit")
        st.markdown("- Joblib")
        st.markdown("- Python 3.8+")

# ====================================
# PAGE 2: DATA EXPLORER
# ====================================
elif page == "📊 Data Explorer":
    st.markdown('<h1 class="main-header">📊 Data Explorer</h1>', unsafe_allow_html=True)
    
    # Choose data source
    data_source = st.radio(
        "Select Data Source",
        ["Sample Dataset", "Uploaded Data"] if user_data is not None else ["Sample Dataset"],
        horizontal=True
    )
    
    if data_source == "Sample Dataset":
        df_display = sample_data
    else:
        df_display = user_data
    
    if df_display is not None:
        # Data Overview
        st.markdown('<p class="sub-header">📋 Dataset Overview</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{df_display.shape[0]:,}")
        col2.metric("Total Columns", df_display.shape[1])
        col3.metric("Missing Values", df_display.isnull().sum().sum())
        col4.metric("Duplicates", df_display.duplicated().sum())
        
        # Data Preview
        st.markdown('<p class="sub-header">👀 Data Preview</p>', unsafe_allow_html=True)
        
        col_preview, col_stats = st.columns([2, 1])
        
        with col_preview:
            st.dataframe(df_display.head(20), use_container_width=True, height=400)
        
        with col_stats:
            st.markdown("**Data Types**")
            dtype_counts = df_display.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
            
            if 'Attrition' in df_display.columns:
                st.markdown("**Target Distribution**")
                attrition_counts = df_display['Attrition'].value_counts()
                st.write(attrition_counts)
                
                fig_pie = px.pie(
                    values=attrition_counts.values,
                    names=attrition_counts.index,
                    title='Attrition Distribution',
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                fig_pie.update_layout(height=250)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Statistical Summary
        st.markdown('<p class="sub-header">📊 Statistical Summary</p>', unsafe_allow_html=True)
        
        summary_option = st.selectbox(
            "Select columns to analyze",
            ["All Numerical", "Specific Columns"]
        )
        
        if summary_option == "All Numerical":
            st.dataframe(df_display.describe().T, use_container_width=True)
        else:
            selected_cols = st.multiselect(
                "Choose columns",
                df_display.columns.tolist(),
                default=df_display.columns[:5].tolist()
            )
            if selected_cols:
                st.dataframe(df_display[selected_cols].describe().T, use_container_width=True)
    else:
        st.warning("⚠️ No data available. Please upload a CSV file or check sample data.")

# ====================================
# PAGE 3: MODEL INSIGHTS
# ====================================
elif page == "🤖 Model Insights":
    st.markdown('<h1 class="main-header">🤖 Model Insights</h1>', unsafe_allow_html=True)
    
    # Tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Comparison", "🎯 Feature Importance", 
                                       "📈 ROC Analysis", "🔍 Confusion Matrices"])
    
    with tab1:
        st.markdown('<p class="sub-header">Model Performance Metrics</p>', unsafe_allow_html=True)
        
        # Show both baseline and tuned results side by side
        col_base, col_tuned = st.columns(2)
        
        with col_base:
            st.markdown("**Baseline Models (Default Parameters)**")
            st.dataframe(
                baseline_results.style.highlight_max(
                    axis=0, 
                    subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                ),
                use_container_width=True
            )
        
        with col_tuned:
            st.markdown("**Tuned Models (Optimized)**")
            st.dataframe(
                model_results.style.highlight_max(
                    axis=0,
                    subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                ),
                use_container_width=True
            )
        
        # Improvement Analysis
        st.markdown("**Performance Improvement Analysis**")
        st.dataframe(comparison_results, use_container_width=True)
        
        # Radar chart comparison
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig_radar = go.Figure()
        
        # Add baseline models (dashed lines)
        for idx, row in baseline_results.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], 
                   row['F1-Score'], row['ROC-AUC']],
                theta=categories,
                fill='toself',
                name=row['Model'] + ' (Base)',
                opacity=0.4,
                line=dict(dash='dash')
            ))
        
        # Add tuned models (solid lines)
        for idx, row in model_results.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], 
                   row['F1-Score'], row['ROC-AUC']],
                theta=categories,
                fill='toself',
                name=row['Model'] + ' (Tuned)',
                line=dict(width=2)
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Model Performance Radar: Baseline vs Tuned",
            height=600
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Best model highlight
        st.success(f"""
        🏆 **Best Performing Model: {metadata['best_model_name']}**
        
        **Baseline Performance:**
        - ROC-AUC: {baseline_best:.3f}
        - Accuracy: {baseline_results[baseline_results['Model'] == metadata['best_model_name']]['Accuracy'].values[0]:.3f}
        
        **Tuned Performance:**
        - ROC-AUC: {tuned_best:.3f} **(+{improvement:.3f})**
        - Accuracy: {metadata['metrics']['Accuracy']:.3f}
        - F1-Score: {metadata['metrics']['F1-Score']:.3f}
        
        **Improvement:** {improvement_pct:+.2f}% gain from hyperparameter tuning
        """)
    
    with tab2:
        st.markdown('<p class="sub-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
        
        # Number of features to display
        n_features = st.slider("Number of top features to display", 5, 20, 15)
        
        top_features = feature_importance.head(n_features)
        
        fig_importance = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {n_features} Most Important Features',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(
            height=max(400, n_features * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Feature importance table
        st.markdown("**Feature Importance Values**")
        st.dataframe(top_features, use_container_width=True, height=300)
        
        # Insights
        st.info(f"""
        **Key Insights:**
        
        The top 3 most predictive features for employee attrition are:
        1. **{top_features.iloc[0]['Feature']}**: {top_features.iloc[0]['Importance']:.4f}
        2. **{top_features.iloc[1]['Feature']}**: {top_features.iloc[1]['Importance']:.4f}
        3. **{top_features.iloc[2]['Feature']}**: {top_features.iloc[2]['Importance']:.4f}
        
        These features should be prioritized in retention strategies.
        """)
    
    with tab3:
        st.markdown('<p class="sub-header">ROC Curve Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        The ROC (Receiver Operating Characteristic) curve illustrates the diagnostic ability of our classifier.
        A higher AUC (Area Under Curve) indicates better model performance.
        """)
        
        # Display model comparison by ROC-AUC
        fig_auc = px.bar(
            model_results.sort_values('ROC-AUC', ascending=True),
            x='ROC-AUC',
            y='Model',
            orientation='h',
            title='Model Comparison by ROC-AUC Score (Tuned Models)',
            color='ROC-AUC',
            color_continuous_scale='RdYlGn',
            range_color=[0.7, 1.0],
            text='ROC-AUC'
        )
        fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_auc.update_layout(height=400)
        st.plotly_chart(fig_auc, use_container_width=True)
        
        # Optimal threshold info
        st.info(f"""
        **Optimal Decision Threshold:** {metadata.get('optimal_threshold', 0.5):.3f}
        
        This threshold was calculated using Youden's J statistic to maximize the difference 
        between True Positive Rate and False Positive Rate.
        
        Using this threshold provides the best balance between sensitivity and specificity.
        """)
        
        # CV Scores
        if 'cv_scores' in metadata:
            st.markdown("**Cross-Validation Stability**")
            cv_scores = metadata['cv_scores']
            cv_df = pd.DataFrame({
                'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                'ROC-AUC': cv_scores
            })
            
            fig_cv = px.bar(
                cv_df,
                x='Fold',
                y='ROC-AUC',
                title='5-Fold Cross-Validation Scores',
                text='ROC-AUC'
            )
            fig_cv.add_hline(
                y=np.mean(cv_scores), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {np.mean(cv_scores):.3f}"
            )
            fig_cv.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_cv.update_layout(height=400)
            st.plotly_chart(fig_cv, use_container_width=True)
    
    with tab4:
        st.markdown('<p class="sub-header">Confusion Matrices</p>', unsafe_allow_html=True)
        
        st.info("""
        Confusion matrices show the performance of the classification model:
        - **True Negatives (TN)**: Correctly predicted no attrition
        - **False Positives (FP)**: Incorrectly predicted attrition
        - **False Negatives (FN)**: Missed attrition cases
        - **True Positives (TP)**: Correctly predicted attrition
        """)
        
        # Note: Actual confusion matrices would require test data
        st.warning("💡 Detailed confusion matrices are available in Phase 3 output: `phase3_model_evaluation_tuned.png`")
        
        # Show sample metrics
        st.markdown("**Best Model Performance Breakdown:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metadata['metrics']['Accuracy']:.3f}")
        col2.metric("Precision", f"{metadata['metrics']['Precision']:.3f}")
        col3.metric("Recall", f"{metadata['metrics']['Recall']:.3f}")
        col4.metric("F1-Score", f"{metadata['metrics']['F1-Score']:.3f}")

# ====================================
# PAGE 4: HYPERPARAMETER TUNING
# ====================================
elif page == "⚡ Hyperparameter Tuning":
    st.markdown('<h1 class="main-header">⚡ Hyperparameter Tuning Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This page shows the impact of systematic hyperparameter optimization using GridSearchCV.
    We tested **594 parameter combinations** across 4 models to find optimal configurations.
    """)
    
    # Tuning Results Summary
    st.markdown('<p class="sub-header">📊 Tuning Results Summary</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tuning results table
        st.dataframe(tuning_results, use_container_width=True)
    
    with col2:
        # Summary metrics
        total_time = tuning_results['Tuning_Time'].sum()
        avg_improvement = tuning_results['Improvement'].mean()
        best_improvement = tuning_results['Improvement'].max()
        
        st.metric("Total Tuning Time", f"{total_time:.1f}s")
        st.metric("Average Improvement", f"+{avg_improvement:.4f}")
        st.metric("Best Improvement", f"+{best_improvement:.4f}")
    
    # Improvement Visualization
    st.markdown('<p class="sub-header">📈 Performance Improvement from Tuning</p>', unsafe_allow_html=True)
    
    fig_improvement = go.Figure()
    
    colors_improvement = ['#2ecc71' if x > 0 else '#e74c3c' 
                          for x in tuning_results['Improvement_%']]
    
    fig_improvement.add_trace(go.Bar(
        x=tuning_results['Improvement_%'],
        y=tuning_results['Model'],
        orientation='h',
        marker_color=colors_improvement,
        text=tuning_results['Improvement_%'].round(2).astype(str) + '%',
        textposition='outside'
    ))
    
    fig_improvement.update_layout(
        title="Percentage Improvement from Hyperparameter Tuning",
        xaxis_title="Improvement (%)",
        yaxis_title="Model",
        height=400
    )
    
    st.plotly_chart(fig_improvement, use_container_width=True)
    
    # Tuning Efficiency Analysis
    st.markdown('<p class="sub-header">⚡ Tuning Efficiency: Time vs Improvement</p>', unsafe_allow_html=True)
    
    fig_efficiency = px.scatter(
        tuning_results,
        x='Tuning_Time',
        y='Improvement',
        size=abs(tuning_results['Improvement']) * 1000,
        color='Improvement',
        color_continuous_scale='RdYlGn',
        text='Model',
        title='Hyperparameter Tuning Efficiency Analysis',
        labels={'Tuning_Time': 'Tuning Time (seconds)', 
                'Improvement': 'ROC-AUC Improvement'}
    )
    
    fig_efficiency.update_traces(textposition='top center')
    fig_efficiency.add_hline(y=0, line_dash="dash", line_color="red", 
                            annotation_text="No Improvement Line")
    fig_efficiency.update_layout(height=500)
    
    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Best Hyperparameters
    st.markdown('<p class="sub-header">🏆 Optimal Hyperparameters</p>', unsafe_allow_html=True)
    
    # Group parameters by model
    for model_name in best_params['Model'].unique():
        with st.expander(f"**{model_name}** - Best Parameters"):
            model_params = best_params[best_params['Model'] == model_name]
            
            param_col1, param_col2 = st.columns(2)
            
            for idx, row in model_params.iterrows():
                if idx % 2 == 0:
                    param_col1.markdown(f"**{row['Parameter']}:** `{row['Value']}`")
                else:
                    param_col2.markdown(f"**{row['Parameter']}:** `{row['Value']}`")
    
    # Key Insights
    st.markdown('<p class="sub-header">💡 Key Insights</p>', unsafe_allow_html=True)
    
    best_model_tuning = tuning_results[tuning_results['Model'] == metadata['best_model_name']].iloc[0]
    
    st.success(f"""
    **Hyperparameter Tuning Impact:**
    
    - **Best Model:** {metadata['best_model_name']}
    - **Baseline ROC-AUC:** {best_model_tuning['Baseline_ROC_AUC']:.4f}
    - **Tuned ROC-AUC:** {best_model_tuning['Tuned_ROC_AUC']:.4f}
    - **Improvement:** +{best_model_tuning['Improvement']:.4f} ({best_model_tuning['Improvement_%']:+.2f}%)
    - **Tuning Time:** {best_model_tuning['Tuning_Time']:.1f} seconds
    
    **Conclusion:** Hyperparameter tuning provided a {best_model_tuning['Improvement_%']:.2f}% performance improvement,
    validating the time investment of {best_model_tuning['Tuning_Time']:.1f} seconds.
    """)

# ====================================

# PAGE 5: MAKE PREDICTIONS
elif page == "???? Make Predictions":
    st.markdown(
        '<h1 class="main-header">Make Predictions</h1>',
        unsafe_allow_html=True
    )

    st.warning(
        "Upload a PREPROCESSED CSV containing the exact 53 model features. "
        "Raw employee records require the original encoding and scaling "
        "pipeline, which is not saved as a reusable transformer."
    )

    feature_names = metadata["feature_names"]
    threshold = float(metadata.get("optimal_threshold", 0.5))

    st.markdown("### Upload preprocessed employee data")
    prediction_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="prediction_csv"
    )

    if prediction_file is not None:
        try:
            prediction_data = pd.read_csv(prediction_file)

            missing = [c for c in feature_names if c not in prediction_data.columns]
            extra = [c for c in prediction_data.columns if c not in feature_names]

            if missing:
                st.error(
                    f"Missing {len(missing)} required feature columns: "
                    + ", ".join(missing)
                )
                st.info(
                    "Use the feature names listed in model_metadata.pkl. "
                    "The file must contain the same encoded and scaled features "
                    "used to train the model."
                )
            elif prediction_data.empty:
                st.error("The uploaded CSV contains no rows.")
            else:
                X_input = prediction_data.loc[:, feature_names].copy()

                for column in feature_names:
                    X_input[column] = pd.to_numeric(
                        X_input[column], errors="coerce"
                    )

                if X_input.isna().any().any():
                    bad_columns = X_input.columns[
                        X_input.isna().any()
                    ].tolist()
                    st.error(
                        "Missing or non-numeric values found in: "
                        + ", ".join(bad_columns)
                    )
                elif not np.isfinite(X_input.to_numpy(dtype=float)).all():
                    st.error("The CSV contains infinite or invalid numeric values.")
                else:
                    predictions = best_model.predict(X_input)
                    probabilities = best_model.predict_proba(X_input)[:, 1]

                    results = pd.DataFrame({
                        "Prediction": predictions,
                        "Attrition_Risk_Probability": probabilities,
                        "Risk_Level": np.where(
                            probabilities >= 0.70, "High",
                            np.where(probabilities >= 0.40, "Medium", "Low")
                        )
                    })

                    st.success(
                        f"Generated predictions for {len(results):,} employees."
                    )

                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Employees analyzed", f"{len(results):,}")
                    metric2.metric(
                        "Predicted attrition",
                        int((predictions == 1).sum())
                    )
                    metric3.metric(
                        "Average predicted risk",
                        f"{probabilities.mean():.1%}"
                    )

                    st.markdown("### Prediction results")
                    st.dataframe(
                        pd.concat(
                            [prediction_data.reset_index(drop=True), results],
                            axis=1
                        ),
                        use_container_width=True
                    )

                    st.download_button(
                        "Download prediction results",
                        data=pd.concat(
                            [prediction_data.reset_index(drop=True), results],
                            axis=1
                        ).to_csv(index=False).encode("utf-8"),
                        file_name="hr_attrition_predictions.csv",
                        mime="text/csv"
                    )

                    with st.expander("Model information"):
                        st.write("Model:", metadata["best_model_name"])
                        st.write("Expected features:", len(feature_names))
                        st.write("Training decision threshold:", threshold)
                        st.caption(
                            "Risk levels are descriptive probability bands, "
                            "not validated HR decision thresholds. Predictions "
                            "are only meaningful when the input preprocessing "
                            "matches training."
                        )

        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")

    st.markdown("---")
    st.markdown("### Required feature template")
    st.caption(
        "Download a blank template containing the exact feature names. "
        "The template is for structure only; it does not contain valid "
        "employee values."
    )

    template = pd.DataFrame(columns=feature_names)
    st.download_button(
        "Download 53-feature template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="hr_prediction_template.csv",
        mime="text/csv"
    )
