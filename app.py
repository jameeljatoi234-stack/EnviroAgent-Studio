import streamlit as st
import os
from crew import run_eia_crew

st.set_page_config(
    page_title="EnviroAgent Studio | AI EIA Generator",
    page_icon="🌱",
    layout="centered"
)

# Fetch GROQ API key automatically from secrets or environment
groq_api_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")

# Header
st.title("🌱 EnviroAgent Studio")
st.markdown("#### Automated Environmental Impact Assessment (EIA) for Any Project")
st.write(
    "Quickly generate an Environmental Impact Assessment (EIA) and Management Plan. "
    "Designed for environmental scientists, engineers, planners, and sustainability students."
)

st.divider()

# Sidebar: Simple Overview
with st.sidebar:
    st.header("🌿 How It Works")
    st.markdown("""
    EnviroAgent runs **4 specialized AI agents** simultaneously:
    1. 💧 **Water & Drainage Agent**
    2. 🏔️ **Soil, Land & Air Agent**
    3. 🦋 **Ecology & Wildlife Agent**
    4. 📋 **Lead Environmental Auditor**
    """)
    st.divider()
    st.caption("Powered by Multi-Agent AI & Groq (openai/gpt-oss-20b)")

# Optional 1-Click Quick Examples
st.subheader("1. Pick an Example or Enter Your Own")
sample_choice = st.selectbox(
    "Choose a preset scenario to auto-fill:",
    [
        "Custom (Type your own details)",
        "☀️ 100 MW Solar Power Farm (Renewable Energy)",
        "🛣️ Regional Highway Widening Project (Infrastructure)",
        "🏭 Solid Waste & Recycling Facility (Urban Management)",
        "🏞️ Eco-Tourism Resort & Riverfront Park (Conservation/Tourism)"
    ]
)

# Preset autofill logic
if sample_choice == "☀️ 100 MW Solar Power Farm (Renewable Energy)":
    default_title = "100 MW Desert Solar Power Park"
    default_type = "Renewable Energy (Solar / Wind / Hydro)"
    default_loc = "Thar Desert, Sindh, Pakistan"
    default_land = "Arid desert grassland with low vegetation"
    default_nature = "2 km from an seasonal wetland; minimal tree cover"
    default_desc = "Installation of photovoltaic solar arrays, substations, and access tracks covering 400 acres."

elif sample_choice == "🛣️ Regional Highway Widening Project (Infrastructure)":
    default_title = "Inter-City Road Widening & Realignment"
    default_type = "Roadway / Highway / Bridge"
    default_loc = "Khuzdar District, Balochistan, Pakistan"
    default_land = "Dry hilly terrain with rocky, gravelly soil"
    default_nature = "Crosses 2 seasonal mountain streams; near native scrubland"
    default_desc = "Widening 20 km of two-lane road into four lanes with culvert installations and hill cutting."

elif sample_choice == "🏭 Solid Waste & Recycling Facility (Urban Management)":
    default_title = "Integrated Municipal Waste & Composting Center"
    default_type = "Waste Management & Recycling Plant"
    default_loc = "Outskirts of Lahore, Punjab, Pakistan"
    default_land = "Flat agricultural land with clayey silt soil"
    default_nature = "1.5 km from an irrigation canal; 800m from nearest village"
    default_desc = "Development of sorting bays, composting pads, leachate collection tanks, and biogas units."

elif sample_choice == "🏞️ Eco-Tourism Resort & Riverfront Park (Conservation/Tourism)":
    default_title = "River Valley Eco-Tourism Retreat"
    default_type = "Tourism & Community Development"
    default_loc = "Swat Valley, Khyber Pakhtunkhwa, Pakistan"
    default_land = "Mountainous valley with rich organic topsoil"
    default_nature = "Directly adjacent to a fresh river; surrounded by pine forests"
    default_desc = "Building eco-cottages, hiking pathways, and sustainable solar water pumps on 15 acres."

else:
    default_title = ""
    default_type = "General Development / Other"
    default_loc = ""
    default_land = ""
    default_nature = ""
    default_desc = ""

st.subheader("2. Project Information")

project_name = st.text_input("Project Name:", value=default_title, placeholder="e.g., Solar Energy Park Phase 1")

project_type = st.selectbox(
    "Project Category:",
    [
        "Renewable Energy (Solar / Wind / Hydro)",
        "Roadway / Highway / Bridge",
        "Urban Housing / Commercial Development",
        "Waste Management & Recycling Plant",
        "Agriculture & Irrigation Project",
        "Tourism & Community Development",
        "General Development / Other"
    ],
    index=0 if not default_type else [
        "Renewable Energy (Solar / Wind / Hydro)",
        "Roadway / Highway / Bridge",
        "Urban Housing / Commercial Development",
        "Waste Management & Recycling Plant",
        "Agriculture & Irrigation Project",
        "Tourism & Community Development",
        "General Development / Other"
    ].index(default_type) if default_type in [
        "Renewable Energy (Solar / Wind / Hydro)",
        "Roadway / Highway / Bridge",
        "Urban Housing / Commercial Development",
        "Waste Management & Recycling Plant",
        "Agriculture & Irrigation Project",
        "Tourism & Community Development",
        "General Development / Other"
    ] else 0
)

location = st.text_input("Project Location:", value=default_loc, placeholder="e.g., District / City, Province, Country")

terrain = st.text_input(
    "Land & Surroundings (Simple words):",
    value=default_land,
    placeholder="e.g., Flat farmland, hilly scrubland, desert sand, coastal shoreline"
)

water_proximity = st.text_input(
    "Nearby Water, Forests, or Protected Areas:",
    value=default_nature,
    placeholder="e.g., 500m from a seasonal stream, 1 km from national park"
)

project_desc = st.text_area(
    "Brief Project Description (What is being built or done?):",
    value=default_desc,
    placeholder="Describe in 1-2 simple sentences what activities will happen during construction and operation."
)

st.divider()

# Action Button
if st.button("🚀 Generate Full EIA Report & Management Plan", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("GROQ_API_KEY is missing. Please add it to your Streamlit App Settings -> Secrets.")
    elif not project_name.strip() or not location.strip():
        st.warning("Please provide at least a Project Name and Location.")
    else:
        status_box = st.status("🔄 Environmental Agents Collaborating...", expanded=True)
        try:
            status_box.write("💧 [1/4] Water Specialist evaluating runoff, streams, and water contamination...")
            status_box.write("🏔️ [2/4] Land & Air Analyst examining soil erosion, dust, and emission risks...")
            status_box.write("🦋 [3/4] Ecologist assessing local wildlife, flora loss, and community impact...")
            status_box.write("📋 [4/4] Lead Environmental Auditor drafting final EIA and mitigation plan...")

            final_report = run_eia_crew(
                project_name=project_name,
                project_type=project_type,
                location=location,
                terrain=terrain,
                earthwork=project_desc,
                water_proximity=water_proximity,
                groq_api_key=groq_api_key
            )

            status_box.update(label="✅ Environmental Assessment Complete!", state="complete", expanded=False)
            st.success("Draft Environmental Impact Assessment Generated Successfully!")
            st.markdown(final_report)

            st.download_button(
                label="📥 Download EIA Report (.md)",
                data=final_report,
                file_name=f"{project_name.replace(' ', '_')}_EIA_Report.md",
                mime="text/markdown"
            )
        except Exception as e:
            status_box.update(label="❌ Assessment Error", state="error", expanded=True)
            st.error(f"Error during execution: {str(e)}")
