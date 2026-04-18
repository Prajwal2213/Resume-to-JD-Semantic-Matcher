# Resume-to-JD-Semantic-Matcher (ResumeIQ)

An intelligent, hybrid AI Applicant Tracking System (ATS) that evaluates candidate resumes against Job Descriptions (JDs) using a combination of **Custom Named Entity Recognition (NER)** and **Advanced Semantic Reasoning**.

## 🚀 Features

- **Custom spaCy NER Model**: Trained on real resume datasets to specifically extract critical entities including:
  - Candidate Name
  - Educational Degree & College
  - Technical Skills & Tools
  - CGPA extraction (via rules and regex)
- **Deep Semantic Matching**: Integrates with a local LLM through [Ollama](https://ollama.com/) (using `deepseek-r1:8b` or your chosen model) to logically compare the candidate's extracted profile with the provided Job Description.
- **ATS Match Score & Verdict**: Generates a percentage match score, a concise verdict of candidate fit, and explicitly flags missing required skills.
- **Interactive UI**: A clean, accessible frontend built with Streamlit to easily upload PDF resumes, paste JDs, and visualize the AI reasoning step-by-step.
- **Privacy First (Local Execution)**: Both the spaCy NER extraction and the semantic reasoning logic run locally on your hardware.

## 🛠 Prerequisites

1. **Python 3.8+**
2. **Ollama**: You must have Ollama installed and running on your system to run the semantic engine. You can download it from [ollama.com](https://ollama.com/).
3. **DeepSeek (or another LLM)**: Ensure the target model is pulled in Ollama. For example:
   ```bash
   ollama run deepseek-r1:8b
   ```

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prajwal2213/Resume-to-JD-Semantic-Matcher.git
   cd Resume-to-JD-Semantic-Matcher
   ```

2. **Install Python dependencies:**
   Make sure you install the base requirements as well as Streamlit and Ollama's Python client:
   ```bash
   pip install -r requirements.txt
   pip install streamlit ollama
   ```

## 🧠 Training & Testing the Custom Model

If you wish to train the NER model yourself or test the extraction pipeline without the UI:

1. **Format the Dataset**: Run `convert_data.py` to convert the `kaggle_resume_dataset.json` to a `.spacy` binary format (`train_data.spacy`).
2. **Train the Model**: Run `train_custom_ai.py`. This reads your configuration (`config.cfg`) and outputs the models to `my_custom_resume_model/`.
3. **Test the Extraction**: Place a sample PDF in the `sample_resumes/` folder (named `test.pdf`) and run:
   ```bash
   python test_model.py
   ```

## 🎯 Running the Web App

Start the interactive Streamlit interface to run the full Hybrid Matcher:

```bash
streamlit run app.py
```

### Usage Instructions in the UI
1. **Target Role:** Paste the specific Job Description in the left sidebar.
2. **Upload Resume:** Upload a candidate's resume as a PDF file in the main panel.
3. The app will automatically run the NER extraction and query the semantic engine.
4. **Insights:** View the ATS Match Analysis score, the parsed candidate profile, and a transparent dropdown showing the AI's "Thought Process" + spaCy NER highlighing.

## 📁 Repository Map

- `app.py` - Main Streamlit UI and semantic reasoning logic.
- `train_custom_ai.py` - Automated training loop to build the spaCy NER model.
- `test_model.py` - CLI script to quickly test NER extraction accuracy on a PDF.
- `convert_data.py` - Utility to convert JSON training data into `.spacy` binaries.
- `config.cfg` - Custom training configurations for spaCy.
- `my_custom_resume_model/` - The compiled/trained spaCy NLP model used by the application.
- `sample_resumes/` - Folder designated for testing PDF files.
