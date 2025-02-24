import streamlit as st
import pandas as pd
import tempfile
import os
from resume import ResumeRanker  # Assuming the updated ResumeRanker class is saved in resume.py
 
def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory and return the directory path."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        # Get the file path in the temporary directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir
 
def main():
    st.title("Resume Ranker")
    st.write("Upload resumes and job description to rank candidates.")
    
    # Add API key input
    # api_key = st.text_input("Enter your API key:", type="password")
    model_choice = st.selectbox(
        "Choose AI Model:",
        ["gpt-4o", "gpt-4o-mini","deepseek-r1-distill-llama-70b"],  # Add more models if needed
        index=0  # Default to the first option
    )
    # File uploader for multiple PDFs, DOCX, and DOC files
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOC, DOCX)",
        type=["pdf", "doc", "docx"],
        accept_multiple_files=True
    )
    
    # Job description input
    job_description = st.text_area(
        "Enter job description:",
        height=200,
        placeholder="Enter the job requirements and qualifications..."
    )
    
    # Configurable scoring weights
    st.sidebar.header("Scoring Configuration")
    st.sidebar.write("Adjust the weights for each scoring criterion (must sum to 100%).")
    
    skills_weight = st.sidebar.slider("Skills Match Weight (%)", 0, 100, 35)
    experience_weight = st.sidebar.slider("Experience Weight (%)", 0, 100, 25)
    education_weight = st.sidebar.slider("Education Weight (%)", 0, 100, 20)
    certifications_weight = st.sidebar.slider("Certifications Weight (%)", 0, 100, 10)
    location_weight = st.sidebar.slider("Location Weight (%)", 0, 100, 10)
    
    # Validate weights
    total_weight = skills_weight + experience_weight + education_weight + certifications_weight + location_weight
    if total_weight != 100:
        st.sidebar.error("Weights must sum to 100%. Please adjust the values.")
    
    # Tie-breaking priority
    st.sidebar.header("Tie-Breaking Priority")
    st.sidebar.write("Set the priority order for breaking ties between candidates with the same score.")
    
    priority_order = st.sidebar.multiselect(
        "Priority Order (drag to reorder):",
        options=["skills_match", "experience", "education", "certifications", "location"],
        default=["skills_match", "experience", "education", "certifications", "location"]
    )
    
    # Single button with validation logic
    if st.button("Rank Resumes", key="rank_button"):
        if not uploaded_files or not job_description or total_weight != 100:
            if not uploaded_files:
                st.warning("Please upload at least one resume.")
            if not job_description:
                st.warning("Please enter a job description.")
            if total_weight != 100:
                st.warning("Scoring weights must sum to 100%. Please adjust the weights.")
        else:
            try:
                with st.spinner("Processing resumes..."):
                    # Save uploaded files to temporary directory
                    temp_dir = save_uploaded_files(uploaded_files)
                    
                    # Convert weights to decimals
                    scoring_weights = {
                        "skills_match": skills_weight / 100,
                        "experience": experience_weight / 100,
                        "education": education_weight / 100,
                        "certifications": certifications_weight / 100,
                        "location": location_weight / 100
                    }
                    
                    # Initialize ranker with custom weights and priority
                    ranker = ResumeRanker(
                        # api_key=os.getenv("OPENAI_API_KEY"),
                        model=model_choice,
                        scoring_weights=scoring_weights,
                        ranking_priority=priority_order
                    )
                    
                    # Process resumes
                    results_df = ranker.process_resumes(temp_dir, job_description)
                    
                    if not results_df.empty:
                        # Display results
                        st.subheader("Rankings:")
                        st.dataframe(results_df)
                        
                        # Convert DataFrame to CSV
                        csv = results_df.to_csv(index=False)
                        
                        # Create download button
                        st.download_button(
                            label="Download Rankings as CSV",
                            data=csv,
                            file_name="resume_rankings.csv",
                            mime="text/csv",
                            key="download_button"
                        )
                    else:
                        st.error("No results were generated. Please check the uploaded files and try again.")
                    
                    # Clean up temporary directory
                    import shutil
                    shutil.rmtree(temp_dir)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
 
if __name__ == "__main__":
    main()
 
 
