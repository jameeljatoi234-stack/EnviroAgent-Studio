import os
from crewai import Agent, Crew, Process, Task, LLM

def run_eia_crew(project_name, project_type, location, terrain, earthwork, water_proximity, openai_api_key):
    # 1. Native OpenAI LLM
    llm = LLM(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.2
    )

    # 2. Domain Agents
    hydrologist = Agent(
        role="Senior Hydrology & Water Resources Specialist",
        goal="Identify surface drainage disruption, groundwater risks, and runoff alterations.",
        backstory="You are a licensed Civil and Hydrological Engineer evaluating catchment basins and culvert design.",
        llm=llm,
        verbose=False
    )

    geotech_air = Agent(
        role="Geotechnical & Environmental Emissions Analyst",
        goal="Quantify soil instability, excavation erosion, and machinery dust emissions.",
        backstory="You are a Geotechnical Engineer assessing slope stability and particulate matter emissions.",
        llm=llm,
        verbose=False
    )

    ecologist = Agent(
        role="Ecological Conservation & Community Impact Officer",
        goal="Assess biodiversity loss, flora clearance, and noise impact on local communities.",
        backstory="You are an environmental biologist evaluating wildlife habitats and public safeguard protocols.",
        llm=llm,
        verbose=False
    )

    lead_synthesizer = Agent(
        role="Lead Environmental Compliance Director & EIA Reporter",
        goal="Synthesize specialist inputs into a regulatory EIA report and Environmental Management Plan (EMP).",
        backstory="You are an accredited Environmental Consultant who transforms risk assessments into audit-ready reports.",
        llm=llm,
        verbose=False
    )

    # 3. Context Data
    project_context = f"""
    Project Name: {project_name}
    Infrastructure Type: {project_type}
    Geographical Location: {location}
    Terrain & Soil Type: {terrain}
    Earthwork Scale: {earthwork}
    Proximity to Water Bodies: {water_proximity}
    """

    # 4. Tasks
    task_hydro = Task(
        description=f"Analyze hydrological and water risks based on this site data:\n{project_context}",
        expected_output="Bullet list of drainage risks, runoff changes, and siltation hazards.",
        agent=hydrologist
    )

    task_geotech = Task(
        description=f"Analyze geotechnical, soil erosion, and dust risks based on this site data:\n{project_context}",
        expected_output="Bullet list of slope stability hazards, earthwork volume impacts, and dust emissions.",
        agent=geotech_air
    )

    task_ecology = Task(
        description=f"Analyze wildlife habitat, tree loss, and local settlement noise based on this site data:\n{project_context}",
        expected_output="Bullet list of ecological impacts and community safeguards.",
        agent=ecologist
    )

    task_final_eia = Task(
        description="Synthesize the hydrological, geotechnical, and ecological findings into a structured Environmental Impact Assessment (EIA). "
                    "Include: 1. Executive Summary, 2. Environmental Impact Matrix, 3. Complete Environmental Management Plan (EMP) with mitigation measures.",
        expected_output="A full, professional Markdown-formatted EIA report.",
        agent=lead_synthesizer
    )

    # 5. Execute Crew
    crew = Crew(
        agents=[hydrologist, geotech_air, ecologist, lead_synthesizer],
        tasks=[task_hydro, task_geotech, task_ecology, task_final_eia],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    return str(result)
