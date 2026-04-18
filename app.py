import streamlit as st
import spacy
import pdfplumber
import ollama
import re
import json
from spacy import displacy

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hybrid AI Resume Intelligence", page_icon="🎯", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_spacy():
    # Loading your custom model (83.13% accuracy)
    return spacy.load("my_custom_resume_model/model-best")

nlp = load_spacy()

# --- HELPER FUNCTIONS ---
def extract_cgpa(text):
    """Explicitly find CGPA using Regex to ensure it doesn't get lost"""
    pattern = r'(\d+\.\d+)\s*/\s*10|CGPA:\s*(\d+\.\d+)'
    match = re.search(pattern, text)
    if match:
        return match.group(1) or match.group(2)
    return "Not Found"

def format_skills_block(skill_text):
    """Takes a massive paragraph of skills and formats it into clean bullet points"""
    headers = [
        "Languages :", "Languages:", "Frameworks :", "Frameworks:", 
        "Developer Tools :", "Developer Tools:", "Databases :", "Databases:", 
        "AI Tools :", "AI Tools:", "Soft Skills :", "Soft Skills:", 
        "Languages Known :", "Languages Known", "Tech Stack:"
    ]
    
    formatted_text = skill_text
    for h in headers:
        formatted_text = formatted_text.replace(h, f"\n* **{h.replace(':', '').strip()}**: ")
        
    return formatted_text.strip()

def get_semantic_analysis(resume_text, jd_text):
    """Interacts with the local reasoning engine silently"""
    prompt = f"""
    You are an expert HR Manager. Analyze the following resume against the job description.
    
    JOB DESCRIPTION: {jd_text}
    RESUME TEXT: {resume_text}
    
    Provide the following in strictly valid JSON format:
    1. "score": An integer (0-100)
    2. "missing_skills": A list of key skills from the JD missing in the resume.
    3. "verdict": A 1-sentence summary of the candidate's fit.
    """
    try:
        # The backend still uses DeepSeek, but the UI will never show it
        response = ollama.chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': prompt}])
        full_content = response['message']['content']
        json_match = re.search(r'\{.*\}', full_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group()), full_content
        return None, full_content
    except Exception as e:
        return None, str(e)

# --- SIDEBAR: JOB DESCRIPTION ---
with st.sidebar:
    st.header("🎯 Target Role")
    jd_input = st.text_area("Paste the Job Description here:", height=300)
    st.divider()
    # Changed the caption to hide the model name
    st.caption("Powered by: Custom spaCy NER & Advanced Semantic Engine")

# --- MAIN UI ---
st.title("🤖 Hybrid AI Resume Parser & Scorer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file and jd_input:
    # Changed the spinner text
    with st.spinner("Semantic Engine is performing deep reasoning analysis..."):
        # Extract Text
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()

        # Process Models
        doc = nlp(text)
        analysis_json, raw_thought = get_semantic_analysis(text, jd_input)
        
        # Group NER Entities for cleaner UI
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]: 
                entities[ent.label_].append(ent.text)

        # --- LAYOUT ---
        col1, col2 = st.columns([1.2, 1]) 

        with col1:
            st.subheader("📊 ATS Match Analysis")
            if analysis_json:
                score = analysis_json.get('score', 0)
                
                if score >= 75: st.success(f"**Match Score: {score}%**")
                elif score >= 50: st.warning(f"**Match Score: {score}%**")
                else: st.error(f"**Match Score: {score}%**")
                
                st.progress(score / 100)
                st.info(f"**Verdict:** {analysis_json.get('verdict', 'N/A')}")
                
                st.markdown("**Missing Requirements:**")
                for skill in analysis_json.get('missing_skills', []):
                    st.markdown(f"- ❌ {skill}")
            else:
                st.warning("Could not parse Semantic Score. Check reasoning below.")

        with col2:
            st.subheader("👤 Candidate Profile")
            
            name = entities.get('Name', ["Not Found"])[0]
            st.markdown(f"**Name:** {name}")
            
            raw_degree = entities.get('Degree', ["Not Found"])[0]
            clean_degree = raw_degree.replace("(CGPA", "").replace(":", "").strip()
            
            st.markdown(f"**Degree:** {clean_degree}")
            st.markdown(f"**CGPA:** {extract_cgpa(text)}")
            
            st.divider()
            
            st.subheader("🛠 Technical Skills")
            if 'Skills' in entities:
                for skill_chunk in entities['Skills']:
                    clean_chunk = format_skills_block(skill_chunk)
                    st.markdown(clean_chunk)
            else:
                st.markdown("*No specific skills detected.*")

        st.divider()

        # --- NER VISUALIZER ---
        # Changed the headers to hide "DeepSeek"
        with st.expander("🔍 Debugger: Custom NER & Engine Reasoning"):
            st.markdown("### Semantic Engine Internal Reasoning")
            st.text_area("Thought Process:", value=raw_thought, height=200)
            st.markdown("### spaCy NER Highlights")
            html = displacy.render(doc, style="ent")
            st.html(html)
else:
    st.info("Please upload a PDF and provide a Job Description to start the analysis.")