import os
import glob
import json
import re
import PyPDF2
import docx2txt
import pandas as pd
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

class ParallelResumeRanker:
    def __init__(self, api_key: str, max_workers: int = 4):
        """Initialize the resume ranker with API key and number of worker threads"""
        self.api_key = api_key
        self.max_workers = max_workers
        genai.configure(api_key=api_key)
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=api_key
        )

    def read_pdf(self, file_path: str) -> str:
        """Read PDF file and return text content"""
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
        """Read DOCX file and return text content"""
        try:
            text = docx2txt.process(file_path)
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {str(e)}")
            return ""

    def process_single_file(self, file_path: str) -> Dict:
        """Process a single resume file"""
        try:
            content = ""
            if file_path.lower().endswith('.pdf'):
                content = self.read_pdf(file_path)
            elif file_path.lower().endswith(('.docx', '.doc')):
                content = self.read_docx(file_path)

            if content:
                return {
                    "content": content,
                    "metadata": {"source": file_path}
                }
            print(f"No content extracted from: {os.path.basename(file_path)}")
            return None
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def load_resumes(self, directory: str) -> List[Dict]:
        """Load resumes from directory using parallel processing"""
        file_patterns = [
            os.path.join(directory, "*.pdf"),
            os.path.join(directory, "*.docx"),
            os.path.join(directory, "*.doc")
        ]

        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(pattern))

        print(f"Found {len(all_files)} files in directory")
        
        documents = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_file, file_path): file_path 
                            for file_path in all_files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        documents.append(result)
                        print(f"Successfully loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

        return documents

    def clean_llm_output(self, result: str) -> str:
        """Extract valid JSON from the LLM result."""
        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
        if json_match:
            return json_match.group(1)
        else:
            print(f"Error: No valid JSON found in result: {result}")
            return ""

    def process_single_resume(self, doc: Dict, job_description: str) -> Dict:
        """Process a single resume including extraction and scoring"""
        try:
            print(f"Processing resume: {os.path.basename(doc['metadata']['source'])}")
            
            # Extract candidate information
            info = self.extract_candidate_info(doc["content"])
            
            # Score resume
            score = self.score_resume(doc["content"], job_description)
            
            return {
                "Name": info["name"],
                "Experience (Years)": info["experience_years"],
                "Location": info["location"],
                "Email": info["email"],
                "Phone": info["phone_number"],
                "Match Score": score,
                "File": os.path.basename(doc["metadata"]["source"])
            }
        except Exception as e:
            print(f"Error processing resume {doc['metadata']['source']}: {str(e)}")
            return None

    def extract_candidate_info(self, resume_text: str) -> Dict:
        """Extract candidate information using LLM"""
        try:
            prompt = PromptTemplate(
                template="""Analyze the following resume text and extract these details:
                1. Full Name
                2. Total years of experience (just the number)
                3. Current location/city
                4. Email address
                5. Phone number

                Format the response as a JSON with these exact keys:
                {{
                    "name": "extracted name",
                    "experience_years": "number only",
                    "location": "extracted location",
                    "email": "extracted email",
                    "phone_number": "extracted phone"
                }}

                Resume text: {text}

                JSON response:""",
                input_variables=["text"]
            )

            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": resume_text})
            cleaned_result = self.clean_llm_output(result)

            if cleaned_result:
                return json.loads(cleaned_result)
            else:
                raise ValueError("Failed to extract valid JSON from the LLM response.")
        except Exception as e:
            print(f"Error extracting info: {str(e)}")
            return {
                "name": "Not found",
                "experience_years": "0",
                "location": "Not found",
                "email": "Not found",
                "phone_number": "Not found"
            }

    def score_resume(self, resume_text: str, job_description: str) -> float:
        """Score resume against job description"""
        try:
            prompt = PromptTemplate(
                template="""You are an expert HR professional. Review this resume against the job requirements and provide a score from 0-100.
                Return ONLY the numerical score, no other text.

                Job Requirements:
                {job_desc}

                Resume:
                {resume}

                Score (0-100):""",
                input_variables=["job_desc", "resume"]
            )

            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "job_desc": job_description,
                "resume": resume_text
            }).strip()

            score = float(''.join(filter(lambda x: x.isdigit() or x == '.', result)))
            return min(max(score, 0), 100)
        except Exception as e:
            print(f"Error scoring resume: {str(e)}")
            return 0.0

    def process_resumes(self, resume_dir: str, job_description: str) -> pd.DataFrame:
        """Process all resumes in parallel and return results as DataFrame"""
        print("Loading resumes...")
        documents = self.load_resumes(resume_dir)

        if not documents:
            print("No resumes found in the specified directory")
            return pd.DataFrame()

        print(f"\nProcessing {len(documents)} resumes in parallel...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures for each document
            futures = []
            for doc in documents:
                future = executor.submit(self.process_single_resume, doc, job_description)
                futures.append(future)
            
            # Process completed futures
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"Processed resume {len(results)}/{len(documents)}")
                except Exception as e:
                    print(f"Error processing resume: {str(e)}")
                    continue

        if not results:
            print("No results to process")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by Match Score in descending order
        if "Match Score" in df.columns:
            df = df.sort_values("Match Score", ascending=False)
        
        # Add S.No as first column after sorting
        df.insert(0, "S.No", range(1, len(df) + 1))
        
        # Define the desired column order
        columns_order = [
            "S.No",
            "Name",
            "Experience (Years)",
            "Location",
            "Email",
            "Phone",
            "Match Score",
            "File"
        ]
        
        # Reorder columns and ensure all expected columns exist
        existing_columns = df.columns.tolist()
        final_columns = []
        for col in columns_order:
            if col in existing_columns:
                final_columns.append(col)
        
        # Add any additional columns that weren't in our ordered list
        for col in existing_columns:
            if col not in final_columns:
                final_columns.append(col)
        
        # Reorder columns
        df = df[final_columns]
        
        # Round Match Score to 2 decimal places
        if "Match Score" in df.columns:
            df["Match Score"] = df["Match Score"].round(2)
        
        # Format Experience Years as integers if possible
        if "Experience (Years)" in df.columns:
            df["Experience (Years)"] = pd.to_numeric(df["Experience (Years)"], errors='ignore')
            
        print("Resume processing complete!")
        return df
def main():
    job_description = """
    We are looking for a Senior Software Engineer with:
    - 5+ years of experience in Python development
    - Strong background in machine learning and AI
    - Experience with cloud platforms (AWS/GCP/Azure)
    - Good communication skills
    - Bachelor's degree in Computer Science or related field
    """

    try:
        # Initialize ranker with 4 worker threads (adjust based on your CPU cores)
        ranker = ParallelResumeRanker(
            api_key="YOUR_API_KEY",
            max_workers=4
        )

        # Process resumes
        results_df = ranker.process_resumes("/content/resumes", job_description)

        if results_df.empty:
            print("No results to display")
            return None

        # Display results
        print("\nTop Ranked Candidates:")
        print(results_df.to_string())

        # Save results
        excel_path = "resume_rankings.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nResults saved to {excel_path}")

        return results_df

    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None

if __name__ == "__main__":
    df = main()