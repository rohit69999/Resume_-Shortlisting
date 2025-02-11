import streamlit as st
import pandas as pd
import tempfile
import os
from resume import ParallelResumeRanker
import time
from datetime import datetime
import io
import xlsxwriter

# Custom CSS styles
def load_css():
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Custom title styling */
        .custom-title {
            color: #1E88E5;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
            text-align: center;
            padding: 1rem;
            border-bottom: 3px solid #1E88E5;
        }
        
        /* Card-like containers */
        .stCard {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Section headers */
        .section-header {
            color: #333;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f0f0;
        }
        
        /* Status indicators */
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        
        /* Custom file uploader */
        .uploadedFile {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 0.5rem;
        }
        
        /* Custom metrics */
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
            margin: 0.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory and return the directory path"""
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return temp_dir

def display_metrics(results_df):
    """Display key metrics from the results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Candidates</div>
            </div>
        """.format(len(results_df)), unsafe_allow_html=True)
    
    with col2:
        avg_score = results_df['Match Score'].mean()
        st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Average Match Score</div>
            </div>
        """.format(avg_score), unsafe_allow_html=True)
    
    with col3:
        top_score = results_df['Match Score'].max()
        st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Top Match Score</div>
            </div>
        """.format(top_score), unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_css()
    
    # App title and description
    st.markdown('<h1 class="custom-title">📄 Smart Resume Ranker</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            Automatically rank and analyze resumes based on job requirements using AI
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for storing progress
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📝 Input", "📊 Results"])
    
    with tab1:
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
        
        # API key input with info
        with st.expander("🔑 API Configuration"):
            api_key = st.text_input(
                "Enter your API key:",
                type="password",
                help="Your API key is required for processing resumes"
            )
        
        # File upload section
        st.markdown('<div class="section-header">Upload Resumes</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drop your resumes here (PDF, DOC, DOCX)",
            type=["pdf", "doc", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files uploaded")
            
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    # Determine file type icon
                    if file.name.lower().endswith('.pdf'):
                        icon = "📄"
                    elif file.name.lower().endswith(('.doc', '.docx')):
                        icon = "📝"
                    else:
                        icon = "📎"
                    
                    st.markdown(f"""
                        <div style="
                            background-color: #f8f9fa;
                            padding: 10px 15px;
                            border-radius: 5px;
                            margin-bottom: 8px;
                            border: 1px solid #e9ecef;
                            display: flex;
                            align-items: center;
                            ">
                            <span style="
                                font-size: 20px;
                                margin-right: 10px;
                                ">{icon}</span>
                            <span style="
                                color: #333s;
                                font-weight: 500;
                                flex-grow: 1;
                                ">{file.name}</span>
                            <span style="
                                color: #060f17;
                                font-size: 0.9em;
                                ">{file.size/1024:.1f} KB</span>
                        </div>
                    """, unsafe_allow_html=True)
        # Job description input
        st.markdown('<div class="section-header">Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "Enter detailed job requirements:",
            height=200,
            placeholder="Example:\n• 5+ years of Python development experience\n• Strong background in machine learning\n• Experience with cloud platforms\n• Excellent communication skills"
        )
        
        # Process button
        if st.button("🚀 Start Processing", key="process_button", use_container_width=True):
            if not uploaded_files or not job_description or not api_key:
                if not uploaded_files:
                    st.warning("⚠️ Please upload at least one resume.")
                if not job_description:
                    st.warning("⚠️ Please enter a job description.")
                if not api_key:
                    st.warning("⚠️ Please enter your API key.")
            else:
                try:
                    with st.spinner("🔄 Processing resumes..."):
                        # Save uploaded files
                        temp_dir = save_uploaded_files(uploaded_files)
                        
                        # Initialize ranker
                        ranker = ParallelResumeRanker(api_key=api_key)
                        
                        # Process resumes with progress bar
                        progress_bar = st.progress(0)
                        results_df = ranker.process_resumes(temp_dir, job_description)
                        progress_bar.progress(100)
                        
                        if not results_df.empty:
                            st.session_state.results_df = results_df
                            st.session_state.processing_complete = True
                            st.success("✅ Processing complete! Switch to Results tab to view rankings.")
                        else:
                            st.error("❌ No results were generated. Please check the uploaded files and try again.")
                        
                        # Clean up
                        import shutil
                        shutil.rmtree(temp_dir)
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
    
    with tab2:
        if st.session_state.processing_complete and hasattr(st.session_state, 'results_df'):
            st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
            
            # Display metrics
            display_metrics(st.session_state.results_df)
            
            # Display results table
            st.markdown("### Ranked Candidates")
            st.dataframe(
                st.session_state.results_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = st.session_state.results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"resume_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # Create Excel file in memory
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    st.session_state.results_df.to_excel(writer, index=False)
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"resume_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.info("👈 Start by uploading resumes and entering job requirements in the Input tab")

if __name__ == "__main__":
    main()
