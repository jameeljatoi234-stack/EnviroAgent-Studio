import os
from crewai import Agent, Crew, Process, Task, LLM

def run_eia_crew(project_name, project_type, location, terrain, earthwork, water_proximity, groq_api_key):
    """
    Initializes and executes the EnviroAgent Studio multi-agent workflow.
    """
    # 1. Connect to Groq using its OpenAI-compatible endpoint
    llm = LLM(
        model="openai/llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        temperature=0.2
    )

    # 2. Agent Definitions (Multi-Agent System)
    hydrologist = Agent(
        role="Senior Hydrology & Water Resources Specialist",
        goal="Identify surface drainage disruption, groundwater table risks, and erosion runoff potential for infrastructure projects.",
        backstory=(
            "You are a licensed Civil and Hydrological Engineer with 15+ years of experience analyzing "
            "catchment areas, flash-flood risks, runoff coefficients, and culvert/drainage adequacy."
        ),
        llm=llm,
        verbose=False
    )

    geotech_air = Agent(
        role="Geotechnical & Environmental Emissions Analyst",
        goal="Quantify soil instability, cut-and-fill erosion, fugitive dust generation, and heavy machinery carbon emissions.",
        backstory=(
            "You are a Geotechnical Engineer and Air Quality Consultant. You evaluate slope stability, "
            "topsoil preservation, borrow pit impacts, and particulate matter (PM10/PM2.5) during mass excavation."
        ),
        llm=llm,
        verbose=False
    )

    ecologist = Agent(
        role="Ecological Conservation & Community Impact Officer",
        goal="Assess biodiversity loss, wildlife habitat corridor severance, noise pollution, and nearby community welfare.",
        backstory=(
            "You are an environmental biologist and social safeguard expert specializing in sensitive eco-zones, "
            "vegetation clearing, noise threshold audits, and public safety mitigation."
        ),
        llm=llm,
        verbose=False
    )

    lead_synthesizer = Agent(
        role="Lead Environmental Compliance Director & EIA Reporter",
        goal="Synthesize specialist inputs into a regulatory-grade Environmental Impact Assessment (EIA) and actionable Environmental Management Plan (EMP).",
        backstory=(
            "You are a veteran Environmental Consultant with global accreditation. You transform multi-disciplinary technical "
            "risk assessments into structured, audit-ready reports featuring prioritized mitigation matrices."
        ),
        llm=llm,
        verbose=False
    )

    # 3. Contextual Data Inputs
    project_context = f"""
    Project Name: {project_name}
    Infrastructure Type: {project_type}
    Geographical Location: {location}
    Terrain & Soil Type: {terrain}
    Earthwork Scale: {earthwork}
    Proximity to Water Bodies: {water_proximity}
    """

    # 4. Sequential Tasks
    task_hydro = Task(
        description=f"Evaluate hydrological risks using this project data:\n{project_context}\n"
                    "Focus on natural drainage blockages, runoff volume changes, potential stream siltation, "
                    "and groundwater contamination. Provide concrete risk metrics and engineering precautions.",
        expected_output="A structured report covering: 1. Catchment & Runoff Alterations, 2. Siltation Risk, 3. Proposed Hydraulic Interventions.",
        agent=hydrologist
    )

    task_geotech = Task(
        description=f"Assess geotechnical and atmospheric impacts using this project data:\n{project_context}\n"
                    "Analyze slope failure risks from cut-and-fill, borrow pit degradations, and fugitive dust (PM2.5/PM10) "
                    "from machinery operations.",
        expected_output="A structured report covering: 1. Slope & Soil Degradation, 2. Dust/Particulate Emissions, 3. Earthwork Management Measures.",
        agent=geotech_air
    )

    task_ecology = Task(
        description=f"Examine ecological disruption and social factors using this project data:\n{project_context}\n"
                    "Evaluate flora/fauna habitat destruction, tree cutting impact, equipment decibel levels on neighboring settlements, "
                    "and community safety.",
        expected_output="A structured report covering: 1. Ecological & Habitat Impact, 2. Noise & Air Quality on Settlements, 3. Social Protection Measures.",
        agent=ecologist
    )

    task_final_eia = Task(
        description="Synthesize the hydrological, geotechnical, and ecological findings into a formal Environmental Impact Assessment (EIA). "
                    "Include: 1. Executive Project Summary, 2. Environmental Impact Matrix (Risk, Severity, Probability), "
                    "3. Comprehensive Environmental Management Plan (EMP) with mitigation techniques, frequency of monitoring, and responsible entities.",
        expected_output="A professional, comprehensive, Markdown-formatted EIA report ready for client and regulatory submission.",
        agent=lead_synthesizer
    )

    # 5. Assemble Workflow Crew
    crew = Crew(
        agents=[hydrologist, geotech_air, ecologist, lead_synthesizer],
        tasks=[task_hydro, task_geotech, task_ecology, task_final_eia],
        process=Process.sequential,
        verbose=False
    )

    # 6. Kickoff Execution
    result = crew.kickoff()
    return str(result)
