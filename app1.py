import streamlit as st
import pandas as pd
import tempfile
import os
import shutil 
from resume import ResumeRanker
from rag import Chatbot

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory and return the directory path."""
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir

def main():
    st.title("Resume Ranker")
    st.write("Upload resumes and job description to rank candidates.")
    
    # Initialize session state variables
    if 'ranked' not in st.session_state:
        st.session_state.ranked = False
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'shortlisted_resumes' not in st.session_state:
        st.session_state.shortlisted_resumes = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'new_chat' not in st.session_state:
        st.session_state.new_chat = False
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""  # Initialize user_query
    
    model_choice = st.selectbox(
        "Choose AI Model:",
        ["gpt-4o", "gpt-4o-mini", "deepseek-r1-distill-llama-70b"],
        index=0
    )

    # Job description input
    job_description = st.text_area(
        "Enter job description:",
        height=200,
        placeholder="Enter the job requirements and qualifications..."
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOC, DOCX)",
        type=["pdf", "doc", "docx"],
        accept_multiple_files=True
    )
    
    # Sidebar configuration
    st.sidebar.header("Scoring Configuration")
    st.sidebar.write("Adjust the weights for each scoring criterion (must sum to 100%).")
    
    skills_weight = st.sidebar.slider("Skills Match Weight (%)", 0, 100, 35)
    experience_weight = st.sidebar.slider("Experience Weight (%)", 0, 100, 25)
    education_weight = st.sidebar.slider("Education Weight (%)", 0, 100, 20)
    certifications_weight = st.sidebar.slider("Certifications Weight (%)", 0, 100, 10)
    location_weight = st.sidebar.slider("Location Weight (%)", 0, 100, 10)
    
    total_weight = skills_weight + experience_weight + education_weight + certifications_weight + location_weight
    if total_weight != 100:
        st.sidebar.error("Weights must sum to 100%. Please adjust the values.")
    
    st.sidebar.header("Tie-Breaking Priority")
    st.sidebar.write("Set the priority order for breaking ties between candidates with the same score.")
    
    priority_order = st.sidebar.multiselect(
        "Priority Order (drag to reorder):",
        options=["skills_match", "experience", "education", "certifications", "location"],
        default=["skills_match", "experience", "education", "certifications", "location"]
    )
    
    # Function to handle resume ranking
    def rank_resumes():
        if not uploaded_files or not job_description or total_weight != 100:
            if not uploaded_files:
                st.warning("Please upload at least one resume.")
            if not job_description:
                st.warning("Please enter a job description.")
            if total_weight != 100:
                st.warning("Scoring weights must sum to 100%. Please adjust the weights.")
            return False
        
        try:
            with st.spinner("Processing resumes..."):
                # Clean up previous temp directory if it exists
                if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                    shutil.rmtree(st.session_state.temp_dir)
                
                # Save new files
                st.session_state.temp_dir = save_uploaded_files(uploaded_files)
                
                scoring_weights = {
                    "skills_match": skills_weight / 100,
                    "experience": experience_weight / 100,
                    "education": education_weight / 100,
                    "certifications": certifications_weight / 100,
                    "location": location_weight / 100
                }
                
                ranker = ResumeRanker(
                    model=model_choice,
                    scoring_weights=scoring_weights,
                    ranking_priority=priority_order
                )
                
                st.session_state.results_df = ranker.process_resumes(st.session_state.temp_dir, job_description)
                st.session_state.shortlisted_resumes = list(st.session_state.results_df[(st.session_state.results_df['total_score'] > 60)]["File"])
                
                # Create a chatbot with the shortlisted resumes
                st.session_state.chatbot = Chatbot(st.session_state.temp_dir, st.session_state.shortlisted_resumes)
                st.session_state.ranked = True
                return True
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return False
    
    # Handling the rank button click
    if st.button("Rank Resumes", key="rank_button"):
        rank_resumes()
    
    # Display results if ranking has been done
    if st.session_state.ranked and st.session_state.results_df is not None:
        st.subheader("Rankings:")
        st.dataframe(st.session_state.results_df)
        
        csv = st.session_state.results_df.to_csv(index=False)
        
        st.download_button(
            label="Download Rankings as CSV",
            data=csv,
            file_name="resume_rankings.csv",
            mime="text/csv",
            key="download_button"
        )
        
        # Chatbot section - only show if we have shortlisted resumes
        if st.session_state.chatbot and st.session_state.shortlisted_resumes:
            st.subheader("Ask the Chatbot")
            st.write(f"You can ask questions about {len(st.session_state.shortlisted_resumes)} shortlisted resumes.")
            
            # Reset chat history if new chat is requested
            if st.session_state.new_chat:
                st.session_state.chat_history = []
                st.session_state.new_chat = False
            
            # Display chat history
            if st.session_state.chat_history:
                st.write("### Chat History")
                for chat in st.session_state.chat_history:
                    st.markdown(f"**You:** {chat['user']}")
                    st.markdown(f"**Chatbot:** {chat['bot']}")
                    st.markdown("---")
            
            # Chatbot form
            with st.form(key="chatbot_form"):
                # Bind the text input to the session state variable
                user_query = st.text_input(
                    "Ask a question about the resumes:",
                    value="",  # Start with empty string
                    key=f"user_query_input_{len(st.session_state.chat_history)}"  # Dynamic key based on chat history length
                )
                
                submit_button = st.form_submit_button("Ask Question")
                
                if submit_button and user_query:
                    with st.spinner("Getting the answer..."):
                        # Ensure that the query method exists
                        if hasattr(st.session_state.chatbot, 'query'):
                            response = st.session_state.chatbot.query(user_query,st.session_state.chat_history)  # Get the query response

                            if response:  # Check if response is valid
                                # Add the conversation to chat history
                                st.session_state.chat_history.append({"user": user_query, "bot": response})

                                # Clear the input field by resetting the session state variable
                                st.session_state.user_query = ""  # Reset input

                            else:
                                st.write("Chatbot could not generate a response.")
                        else:
                            st.error("The Chatbot does not have a query method!")
                    
                    # Rerun the app to update the UI
                    st.rerun()
    
        # Add a button to start a new chat
        if st.button("Start New Chat"):
            st.session_state.new_chat = True
            st.rerun()  # Use st.rerun() to reset the chat

if __name__ == "__main__":
    main()
