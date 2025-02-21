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

load_dotenv()



logging.basicConfig(level=logging.INFO)

class ResumeRanker:
    def __init__(self,model=str, scoring_weights: Dict[str, float] = None, ranking_priority: List[str] = None):
        """Initialize the resume ranker with API key and scoring configuration."""
        self.api_key = "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" 
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Add it to the .env file.")
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=0.1
        )
        # self.api_key = api_key
        # self.llm = ChatGroq(
        #     model="deepseek-r1-distill-llama-70b",
        #     api_key=api_key
        # )
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
        self.unified_prompt_template = """You are an expert HR analyst. Your task is to extract information from the resume and evaluate it based on the job requirements.

    1. Extract the following candidate details:
    - Full name
    - Years of experience: Calculate this by:
      * Summing the duration of all professional roles (excluding internships)
      * For current roles, calculate duration up to the present month ({current_month_year})
      * Avoid double-counting overlapping positions
      * Convert all experience to decimal years (e.g., 3 years 6 months = 3.5 years)
      * Include only relevant professional experience
      * Exclude education periods unless they involved professional work
      * Round to one decimal place
    - Skills list
    - Education details
    - Certifications
    - Location (city, state)
    - Email
    - Phone number

    2. Evaluate the resume using the following criteria and weights:
    {criteria_list}

    Additional Instructions:
    1. Score each criterion from 0 to 100 based on job requirements.
    2. Calculate the weighted total score.
    3. For tied totals, prioritize in this order: {priority_order}.

    Job Requirements:
    {job_desc}

    Resume Content:
    {resume}

    Return the response as a valid JSON object with the following structure:
    {{
        "extracted_info": {{
        "name": "full name",
        "experience_years": float,
        "skills": ["skill1", "skill2", ...],
        "education": ["degree1", "degree2", ...],
        "certifications": ["cert1", "cert2", ...],
        "location": "city, state",
        "email": "email address",
        "phone": "phone number"
        }},
        "evaluation": {{
        "skills_match": int (0-100),
        "experience": int (0-100),
        "education": int (0-100),
        "certifications": int (0-100),
        "location": int (0-100),
        "total_score": float,
        "explanation": "brief reasoning for scores, including experience calculation details"
        }}
    }}

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
        try:
            prompt = PromptTemplate(
                template=self.unified_prompt_template,
                input_variables=["job_desc", "resume", "criteria_list", "priority_order"]
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

            cleaned_result = self.clean_llm_output(result)
            analysis = json.loads(cleaned_result)
            logging.info(f"Extracted Info: {analysis.get('extracted_info', {})}")
            logging.info(f"Evaluation: {analysis.get('evaluation', {})}")


            # Default structures
            default_info = {
                "name": "Not found",
                "experience_years": 0,
                "skills": [],
                "education": [],
                "certifications": [],
                "location": "Not found",
                "email": "Not found",
                "phone": "Not found"
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

            # Merge with defaults for missing fields
            extracted_info = {**default_info, **analysis.get("extracted_info", {})}
            evaluation = {**default_scores, **analysis.get("evaluation", {})}

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
                "processing_time": processing_time
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
        total_processing_time = 0
        
        for i, doc in enumerate(documents, 1):
            try:
                print(f"\nProcessing resume {i}/{len(documents)}")
                
                # Single LLM call for both extraction and scoring
                analysis = self.analyze_resume(doc["content"], job_description)
                info = analysis["extracted_info"]
                scores = analysis["evaluation"]
                processing_time = analysis["processing_time"]
                total_processing_time += processing_time

                # Convert experience_years to float if possible
                try:
                    experience_years = float(info.get('experience_years', 0))
                except (ValueError, TypeError):
                    experience_years = 0

                # Combine results
                result = {
                    'name': info.get('name', 'Not found'),
                    'total_score': scores.get('total_score', 0),
                    'skills_match': scores.get('skills_match', 0),
                    'experience': scores.get('experience', 0),
                    'education': scores.get('education', 0),
                    'certifications': scores.get('certifications', 0),
                    'location': scores.get('location', 0),
                    'experience_years': experience_years,
                    'phone': info.get('phone', 'Not found'),
                    'email': info.get('email', 'Not found'),
                    'skills': ", ".join(info.get('skills', [])),
                    'location_info': info.get('location', 'Not found'),
                    'File': os.path.basename(doc["metadata"]["source"]),
                    'processing_time': round(processing_time, 2)  # Add processing time to results
                }
                results.append(result)
                print(f"Match Score: {scores['total_score']}")
                print(f"Processing Time: {processing_time:.2f} seconds")
                logging.info(f"Processed {doc['metadata']['source']}: Score {analysis.get('evaluation', {}).get('total_score', 0)}, Time: {processing_time:.2f}s")
            except Exception as e:
                print(f"Error processing resume {i}: {str(e)}")
                continue

        overall_time = time.time() - overall_start_time
        print(f"\nOverall Processing Statistics:")
        print(f"Total time taken: {overall_time:.2f} seconds")
        print(f"Average time per resume: {(total_processing_time/len(documents)):.2f} seconds")
        print(f"Total LLM processing time: {total_processing_time:.2f} seconds")

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
            'location_info', 'File', 'processing_time'  # Added processing_time to columns
        ]
        df = df[columns]
        df.set_index('Rank', inplace=True)
        return df
