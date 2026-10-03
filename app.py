import streamlit as st
import os
from crew import run_eia_crew

st.set_page_config(
    page_title="EnviroAgent Studio | AI EIA Generator",
    page_icon="🌿",
    layout="wide"
)

# Header Section
st.title("🌿 EnviroAgent Studio")
st.markdown("### Autonomous Multi-Agent System for Environmental Impact Assessment (EIA)")
st.caption("Powered by Generative AI, CrewAI Multi-Agent Workflows, and High-Speed LLM Orchestration.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Enter Groq API Key:", type="password", help="Get a free key from console.groq.com")
    
    # Priority: Streamlit Secrets > User Input > System Env
    groq_api_key = api_key_input or st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    
    st.divider()
    st.markdown("#### 🤖 Agent Roster")
    st.markdown("""
    - **Hydrology Specialist Agent**
    - **Geotechnical & Air Analyst**
    - **Ecology & Community Safeguard**
    - **Lead Compliance Synthesizer**
    """)
    st.info("The agents run in sequence, passing domain risk evaluations to the Lead Synthesizer.")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    project_name = st.text_input("Project Title:", value="Regional Highway Widening & Realignment Project")
    project_type = st.selectbox(
        "Infrastructure Category:",
        ["Highway / Roadway Infrastructure", "Hydroelectric Dam & Reservoir", "Bridge & Flyover Structure", "Commercial / Industrial Township", "Water Supply & Canal Network"]
    )
    location = st.text_input("Geographical Location:", value="District Khuzdar, Balochistan, Pakistan")

with col2:
    terrain = st.selectbox(
        "Terrain & Soil Classification:",
        ["Arid Mountainous / Calcareous Gravelly Soil", "Alluvial Plain / Silt-Clay Mixture", "Coastal Sandy / High Salinity", "Hilly / High Landslide Susceptibility"]
    )
    earthwork = st.text_input("Estimated Cut & Fill Volume:", value="65,000 m³ excavation with 30,000 m³ embankment fill")
    water_proximity = st.text_input("Proximity to Natural Water Bodies:", value="Traverses 2 seasonal stream channels (Wadis), 400m from local reservoir")

st.divider()

# Execution Button
if st.button("🚀 Run Multi-Agent EIA Workflow", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("⚠️ Please provide a valid Groq API Key in the sidebar or in Streamlit Secrets to run the multi-agent system.")
    else:
        status_box = st.status("🔄 AI Agent Crew Activated...", expanded=True)
        try:
            status_box.write("🌊 [Agent 1/4] Senior Hydrologist analyzing drainage & siltation patterns...")
            status_box.write("⛰️ [Agent 2/4] Geotech & Air Quality Officer computing earthwork hazards...")
            status_box.write("🐾 [Agent 3/4] Ecological Officer evaluating habitat & local noise thresholds...")
            status_box.write("📋 [Agent 4/4] Lead Compliance Synthesizer generating Environmental Management Plan...")
            
            # Execute Crew
            final_report = run_eia_crew(
                project_name=project_name,
                project_type=project_type,
                location=location,
                terrain=terrain,
                earthwork=earthwork,
                water_proximity=water_proximity,
                groq_api_key=groq_api_key
            )
            
            status_box.update(label="✅ EIA Assessment Complete!", state="complete", expanded=False)
            
            # Output Display
            st.success("Draft Environmental Impact Assessment Generated Successfully!")
            st.markdown(final_report)
            
            # Download Capability
            st.download_button(
                label="📥 Download EIA Draft Report (.md)",
                data=final_report,
                file_name=f"{project_name.replace(' ', '_')}_EIA_Report.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            status_box.update(label="❌ Workflow Execution Error", state="error", expanded=True)
            st.error(f"Error during execution: {str(e)}")
