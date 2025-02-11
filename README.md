# AI-Powered Resume Ranker

An intelligent resume ranking system that automatically analyzes and scores resumes based on job requirements using AI. Built with Streamlit and powered by LLM for accurate candidate matching.

![Resume Ranker Demo](path_to_demo_image.gif)

## 🌟 Features

- **Automated Resume Analysis**: Process multiple resumes (PDF, DOC, DOCX) simultaneously
- **AI-Powered Matching**: Advanced matching against job requirements using LLM
- **Parallel Processing**: Fast processing with multi-threading support
- **Interactive UI**: Clean, modern interface built with Streamlit
- **Detailed Analytics**: 
  - Match scoring
  - Experience analysis
  - Location mapping
  - Contact information extraction
- **Export Options**: Download results in CSV or Excel format

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Groq API key (for LLM access)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-ranker.git
cd resume-ranker
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_api_key_here
```

## 💻 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Follow these steps in the UI:
   - Enter your API key
   - Upload resume files (PDF, DOC, DOCX supported)
   - Enter the job description
   - Click "Start Processing"
   - View results and download reports

## 📁 Project Structure

```
resume-ranker/
├── app.py              # Streamlit interface
├── resume.py           # Core resume processing logic
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

## 📦 Dependencies

- `streamlit`: Web interface
- `pandas`: Data processing
- `PyPDF2`: PDF file processing
- `python-docx`: Word document processing
- `langchain`: LLM integration
- `groq`: AI model access

## 🛠️ Development

### Add New Features

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```
3. Make changes and commit:
```bash
git commit -m "Add your feature"
```
4. Push and create a pull request

### Run Tests

```bash
python -m pytest tests/
```

## 🔍 Sample Output

The system generates a DataFrame with the following columns:
- S.No
- Name
- Experience (Years)
- Location
- Email
- Phone
- Match Score
- File

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


