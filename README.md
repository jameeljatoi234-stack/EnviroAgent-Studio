# 🌿 EnviroAgent Studio
### Automated Multi-Agent Environmental Impact Assessment (EIA) Workflow

EnviroAgent Studio is an end-to-end multi-agent AI system designed for Civil and Environmental Engineers. It automates baseline environmental risk assessments and drafts regulatory-compliant Environmental Management Plans (EMP) in minutes.

---

## 📌 Problem Statement
Conducting initial Environmental Impact Assessments (EIAs) for infrastructure developments is slow, siloed, and labor-intensive. Consulting teams spend weeks manually consolidating hydrological risks, soil erosion indices, and biodiversity impacts into standardized matrices, delaying project design phases.

## 💡 Solution Overview
EnviroAgent Studio coordinates 4 specialized AI agents using **CrewAI** and **Llama 3.3 (via Groq)**:
1. **Hydrological Engineer Agent:** Evaluates runoff alterations, stream siltation, and groundwater risks.
2. **Geotechnical & Air Quality Agent:** Evaluates slope stability, excavation dust, and machinery emissions.
3. **Ecology & Social Safeguards Agent:** Evaluates wildlife impact, tree clearance, and decibel exposure on settlements.
4. **Lead Compliance Synthesizer:** Consolidates findings into an executive matrix and formal Environmental Management Plan (EMP).

---

## 🚀 Live Streamlit Deployment

### 1. Push to GitHub
Extract this repository folder and push directly to GitHub:
```bash
git init
git add .
git commit -m "Initial commit: EnviroAgent Studio"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **New app**.
3. Select your repository, set the branch to `main`, and the main file path to `app.py`.
4. Under **Advanced Settings > Secrets**, paste your API key:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```
5. Click **Deploy!**

---

## 🛠️ Local Setup (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
