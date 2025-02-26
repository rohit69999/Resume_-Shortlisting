import os
import glob
import json
import re
import PyPDF2
import docx2txt
import pandas as pd
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from datetime import datetime
import logging
import time
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
 
 
 
 
 
logging.basicConfig(level=logging.INFO)
 
class ResumeRanker:
    def __init__(self,model=str, scoring_weights: Dict[str, float] = None, ranking_priority: List[str] = None):
        """Initialize the resume ranker with API key and scoring configuration."""
        self.gpt_api_key = st.secrets ["OPENAI_API_KEY"]
        self.groq_api_key = st.secrets ["GROQ_API_KEY"]
        # self.gpt_api_key = os.getenv("OPENAI_API_KEY")
        # self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.gpt_api_key or not self.groq_api_key:
            raise ValueError("API key is missing. Add it to the .env file.")
        self.model = model
        if self.model in ["gpt-4o","gpt-4o-mini"]:
            self.llm = ChatOpenAI(
                model=model,
                api_key=self.gpt_api_key,
                temperature=0.1
            )
        else:
            self.llm = ChatGroq(
                model=model,
                api_key=self.groq_api_key
            )
        # Get the current month and year
        self.current_month_year = datetime.today().strftime("%B %Y")
        # Default scoring weights
        self.scoring_weights = scoring_weights or {
            "skills_match": 0.35,
            "experience": 0.25,
            "education": 0.20,
            "certifications": 0.10,
            "location": 0.10
        }
 
        # Default tie-breaking priority
        self.ranking_priority = ranking_priority or [
            "skills_match",
            "experience",
            "education",
            "certifications",
            "location"
        ]
        logging.info(f"Scoring weights: {self.scoring_weights}")
        logging.info(f"Ranking priority: {self.ranking_priority}")
        
        # Combined prompt template for both extraction and scoring
        self.unified_prompt_template = """You are an expert HR analyst tasked with evaluating resumes against a job description. Your task is to extract relevant information from the resume and evaluate how well it matches the job description. Please follow these instructions carefully:
 
### 1. **Extract Candidate Details:**
 
- **Full Name**: Extract the full name of the candidate.
- **Years of Experience**: Calculate the total years of relevant professional experience. Ensure the experience is directly aligned with the job description and key responsibilities (e.g., if applying for a Data Scientist role, ensure the experience is related to data analysis, machine learning, etc.).
    - Only include experience relevant to the job description (do not count unrelated or irrelevant jobs).
    - For current roles, calculate the duration up to the current month ({{current_month_year}}).
    - Round the result to one decimal place.
- **Skills List**: Extract the list of skills mentioned in the resume and compare them to the job description’s required skills.
    - Ensure that only **relevant skills** are considered for scoring (e.g., if the job requires Python and SQL, make sure these are mentioned in the resume).
- **Education Details**: List the degrees, certifications, and institutions attended, if applicable.
- **Certifications**: List any relevant certifications, especially those mentioned in the job description.
- **Location**: Extract the location of the candidate and check if they are within a reasonable distance from the job location.
- **Email**: Extract the candidate’s email address.
- **Phone Number**: Extract the candidate’s phone number.
 
### 2. **Evaluate the Resume Using the Following Criteria:**
 
Evaluate the candidate’s match against the job description using the following criteria and score each on a scale of 0-100:
 
#### Skills Match:
- Score based on how closely the **skills in the resume** match the skills required in the job description.
- Only consider **relevant skills** (e.g., don’t count generic or irrelevant skills).
- Score from 0 to 100.
 
#### Experience:
- Evaluate if the candidate’s **experience** matches the key responsibilities and requirements of the role.
- Only count **relevant experience** (e.g., if the job is for a Data Scientist, the candidate’s experience should include data analysis, machine learning, etc.).
- Score from 0 to 100.
 
#### Education:
- Evaluate if the candidate’s education matches the qualifications required for the job.
- Score from 0 to 100.
 
#### Certifications:
- Check if the candidate has certifications that are relevant to the job.
- Score from 0 to 100.
 
#### Location:
- Evaluate if the candidate’s location is relevant to the job location (commutable distance or open to relocation).
- Score from 0 to 100.
 
### 3. **Calculate the Total Score:**
 
After scoring the individual criteria, calculate the **total score** by summing the weighted scores for each criterion. Use the following criteria list:
{{criteria_list}}
 

 
### 4. **Cut-off Threshold (50 Points):**
- **Candidates with a total score below 50** will be **excluded** from the final list.
- Only candidates whose **total score** is **50 or above** should be shown in the results.
 
### 5. **Tie-Breaking Priority:**
If two candidates have the same total score, break the tie using the following priority order:
{{priority_order}}
 
### Job Requirements (Job Description):
 
{{job_desc}}
 
### Resume Content:
 
{{resume}}
 
Return the response in a JSON format with the following structure:
 
json
{
    "information": {
        "name": "Full Name",
        "experience_years": 5.5,
        "skills": ["Python", "SQL", "Data Analysis"],
        "education": ["BSc Computer Science", "MSc Data Science"],
        "certifications": ["AWS Certified Solutions Architect"],
        "location": "New York, NY",
        "email": "candidate@example.com",
        "phone": "+1-555-555-5555"
    },
    "evaluation": {
        "skills_match": 85,
        "experience": 90,
        "education": 80,
        "certifications": 75,
        "location": 100,
        "total_score": 85.5,
        "explanation": "The candidate has excellent skills in Python and SQL, with significant experience in data analysis. Their educational background is suitable, and their location is ideal for the job."
    }
}
 
    IMPORTANT: Be precise in experience calculation and provide detailed reasoning in the explanation field."""
 
    def update_scoring_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights and validate totals."""
        total = sum(new_weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError("Weights must sum to ~1.0")
        self.scoring_weights = new_weights
 
    def update_ranking_priority(self, new_priority: List[str]):
        """Update tie-breaking priority order."""
        if set(new_priority) != set(self.scoring_weights.keys()):
            raise ValueError("Priority list must contain all scoring criteria")
        self.ranking_priority = new_priority
 
    def _generate_criteria_list(self) -> str:
        """Generate formatted criteria list for prompt."""
        return "\n".join(
            [f"- {k.capitalize()}: {v * 100}%"
             for k, v in self.scoring_weights.items()]
        )
 
    def read_pdf(self, file_path: str) -> str:
        """Read PDF file and return text content."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return ""
 
    def read_docx(self, file_path: str) -> str:
        """Read DOCX file and return text content."""
        try:
            text = docx2txt.process(file_path)
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {str(e)}")
            return ""
 
    def load_resumes(self, directory: str) -> List[Dict]:
        """Load resumes from directory."""
        documents = []
        file_patterns = [
            os.path.join(directory, "*.pdf"),
            os.path.join(directory, "*.docx"),
            os.path.join(directory, "*.doc")
        ]
 
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(pattern))
 
        print(f"Found {len(all_files)} files in directory")
 
        for file_path in all_files:
            try:
                content = ""
                if file_path.lower().endswith('.pdf'):
                    content = self.read_pdf(file_path)
                elif file_path.lower().endswith(('.docx', '.doc')):
                    content = self.read_docx(file_path)
 
                if content:
                    documents.append({
                        "content": content,
                        "metadata": {"source": file_path}
                    })
                    print(f"Successfully loaded: {os.path.basename(file_path)}")
                else:
                    print(f"No content extracted from: {os.path.basename(file_path)}")
 
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
 
        return documents
 
    def clean_llm_output(self, result: str) -> str:
        """Extract valid JSON from the LLM result."""
        try:
            result = result.replace("```json", "").replace("```", "").strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json_match.group(0)
            else:
                print(f"Error: No valid JSON found in result: {result}")
                return "{}"
        except Exception as e:
            print(f"Error cleaning LLM output: {str(e)}")
            return "{}"
   
    def analyze_resume(self, resume_text: str, job_description: str) -> Dict:
        """Analyze resume with combined extraction and scoring in a single LLM call."""
        start_time = time.time()  # Start timing
        
        # Check if resume text is empty or too short to be valid
        if not resume_text or len(resume_text.strip()) < 50:  # Arbitrary minimum length
            return {
                "extracted_info": {
                    "name": "FILE_NOT_SUPPORTED",
                    "experience_years": 0,
                    "skills": [],
                    "education": [],
                    "certifications": [],
                    "location": "FILE_NOT_SUPPORTED",
                    "email": "FILE_NOT_SUPPORTED",
                    "phone": "FILE_NOT_SUPPORTED"
                },
                "evaluation": {
                    "skills_match": 0,
                    "experience": 0,
                    "education": 0,
                    "certifications": 0,
                    "location": 0,
                    "total_score": 0,
                    "explanation": "File could not be parsed or is not supported."
                },
                "processing_time": time.time() - start_time,
                "error": "FILE_NOT_SUPPORTED"
            }
        
        default_info = {
            "name": "FILE_NOT_SUPPORTED",
            "experience_years": 0,
            "skills": [],
            "education": [],
            "certifications": [],
            "location": "FILE_NOT_SUPPORTED",
            "email": "FILE_NOT_SUPPORTED",
            "phone": "FILE_NOT_SUPPORTED"
        }

        default_scores = {
            "skills_match": 0,
            "experience": 0,
            "education": 0,
            "certifications": 0,
            "location": 0,
            "total_score": 0,
            "explanation": "Error occurred while analyzing."
        }
        
        try:
            prompt = PromptTemplate(
                template=self.unified_prompt_template,
                input_variables=["job_desc", "resume", "criteria_list", "priority_order", "current_month_year"]
            )
            logging.info("LLM response received")

            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "job_desc": job_description,
                "resume": resume_text,
                "criteria_list": self._generate_criteria_list(),
                "priority_order": ", ".join(self.ranking_priority),
                "current_month_year": self.current_month_year
            })
            logging.info(f"LLM response: {result}")

            cleaned_result = self.clean_llm_output(result)
            analysis = json.loads(cleaned_result)

            # Check if the analysis has valid data
            if "information" in analysis and analysis["information"].get("name") == "Not found":
                return {
                    "extracted_info": default_info,
                    "evaluation": default_scores,
                    "processing_time": time.time() - start_time,
                    "error": "FILE_NOT_SUPPORTED"
                }

            # Map the correct keys (information to extracted_info if needed)
            extracted_info = analysis.get("information", analysis.get("extracted_info", {}))
            if not extracted_info:
                extracted_info = default_info
                
            evaluation = analysis.get("evaluation", {})
            if not evaluation:
                evaluation = default_scores

            # Recalculate total score
            total = sum(
                evaluation.get(criterion, 0) * self.scoring_weights[criterion]
                for criterion in self.scoring_weights
            )
            evaluation["total_score"] = round(total, 2)

            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                "extracted_info": extracted_info,
                "evaluation": evaluation,
                "processing_time": processing_time
            }

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error analyzing resume: {str(e)}")
            return {
                "extracted_info": default_info,
                "evaluation": default_scores,
                "processing_time": processing_time,
                "error": "FILE_NOT_SUPPORTED"
            }

    def process_resumes(self, resume_dir: str, job_description: str) -> pd.DataFrame:
        """Process resumes with combined extraction and scoring."""
        overall_start_time = time.time()  # Start timing overall process
        print("Loading resumes...")
        documents = self.load_resumes(resume_dir)

        if not documents:
            print("No resumes found in the specified directory")
            return pd.DataFrame()

        print(f"\nProcessing {len(documents)} resumes...")

        results = []
        unsupported_files = []
        total_processing_time = 0
        
        for i, doc in enumerate(documents, 1):
            try:
                print(f"\nProcessing resume {i}/{len(documents)}")
                
                # Single LLM call for both extraction and scoring
                analysis = self.analyze_resume(doc["content"], job_description)
                
                # Check if file is not supported
                if analysis.get("error") == "FILE_NOT_SUPPORTED":
                    unsupported_files.append(os.path.basename(doc["metadata"]["source"]))
                    print(f"File not supported: {os.path.basename(doc['metadata']['source'])}")
                    continue
                    
                info = analysis["extracted_info"]
                scores = analysis["evaluation"]
                processing_time = analysis["processing_time"]
                total_processing_time += processing_time

                total_score = scores.get('total_score', 0)
                
                # Apply threshold: Only process resumes with a score > 50
                if total_score > 50:
                    # Convert experience_years to float if possible
                    try:
                        experience_years = float(info.get('experience_years', 0))
                    except (ValueError, TypeError):
                        experience_years = 0

                    # Combine results
                    result = {
                        'name': info.get('name', 'Not found'),
                        'total_score': total_score,
                        'skills_match': scores.get('skills_match', 0),
                        'experience': scores.get('experience', 0),
                        'education': scores.get('education', 0),
                        'certifications': scores.get('certifications', 0),
                        'location': scores.get('location', 0),
                        'experience_years': experience_years,
                        'phone': info.get('phone', 'Not found'),
                        'email': info.get('email', 'Not found'),
                        'skills': ", ".join(info.get('skills', [])) if isinstance(info.get('skills', []), list) else info.get('skills', ""),
                        'location_info': info.get('location', 'Not found'),
                        'File': os.path.basename(doc["metadata"]["source"]),
                        'processing_time': round(processing_time, 2)
                    }
                    results.append(result)
                    print(f"Match Score: {total_score}")
                    print(f"Processing Time: {processing_time:.2f} seconds")
                    logging.info(f"Processed {doc['metadata']['source']}: Score {total_score}, Time: {processing_time:.2f}s")
                else:
                    print(f"Skipping resume {i} due to low score ({total_score})")
            except Exception as e:
                print(f"Error processing resume {i}: {str(e)}")
                unsupported_files.append(os.path.basename(doc["metadata"]["source"]))
                continue

        overall_time = time.time() - overall_start_time
        print(f"\nOverall Processing Statistics:")
        print(f"Total time taken: {overall_time:.2f} seconds")
        print(f"Average time per resume: {(total_processing_time/max(1, len(documents))):.2f} seconds")
        print(f"Total LLM processing time: {total_processing_time:.2f} seconds")
        
        # Report unsupported files
        if unsupported_files:
            print(f"\nUnsupported files ({len(unsupported_files)}):")
            for file in unsupported_files:
                print(f"- {file}")

        if not results:
            print("No results to process")
            return pd.DataFrame()

        # Create DataFrame with all columns
        df = pd.DataFrame(results)
        
        # Sort by total score
        df = df.sort_values('total_score', ascending=False).reset_index(drop=True)
        
        # Add rank column and set as index
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        # Reorder columns for better readability
        columns = [
            'Rank', 'name', 'total_score', 'skills', 'experience_years', 'phone', 'email',
            'location_info', 'File', 'processing_time'
        ]
        df = df[columns]
        df.set_index('Rank', inplace=True)
        return df
