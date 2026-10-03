import os
import litellm
from crewai import Agent, Crew, Process, Task, LLM

# -------------------------------------------------------------------------
# INTERCEPTOR: Strips 'cache_breakpoint' so Groq never throws BadRequestError
# -------------------------------------------------------------------------
litellm.drop_params = True

_orig_completion = litellm.completion
def _clean_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return _orig_completion(*args, **kwargs)
litellm.completion = _clean_completion

_orig_acompletion = litellm.acompletion
async def _clean_acompletion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return await _orig_acompletion(*args, **kwargs)
litellm.acompletion = _clean_acompletion
# -------------------------------------------------------------------------

def run_eia_crew(project_name, project_type, location, terrain, earthwork, water_proximity, groq_api_key):
    """
    Initializes and executes the General Environmental Impact Assessment (EIA) workflow
    using openai/gpt-oss-20b on Groq.
    """
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Initialize openai/gpt-oss-20b on Groq
    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_api_key,
        temperature=0.2
    )

    # 1. Generalized Multi-Disciplinary Environmental Agents
    hydrologist = Agent(
        role="Water Resources & Hydrology Specialist",
        goal="Evaluate impacts on natural water flow, drainage changes, potential water pollution, and aquatic systems.",
        backstory=(
            "You are an environmental hydrologist who evaluates the impact of development projects on streams, "
            "rivers, wetlands, drainage basins, and groundwater reserves."
        ),
        llm=llm,
        verbose=False
    )

    land_air_agent = Agent(
        role="Land, Soil & Air Quality Analyst",
        goal="Assess land degradation, soil erosion, dust, chemical runoff, and air emissions from project activities.",
        backstory=(
            "You are an environmental scientist specializing in soil quality, terrain stability, and atmospheric "
            "emissions (particulate matter, exhaust fumes, and greenhouse gases)."
        ),
        llm=llm,
        verbose=False
    )

    ecologist = Agent(
        role="Ecology, Biodiversity & Community Safeguard Officer",
        goal="Identify threats to local flora, wildlife habitats, green cover, and public health or noise concerns.",
        backstory=(
            "You are a conservation ecologist and social safeguard auditor who ensures that vulnerable ecosystems, "
            "native vegetation, and nearby residential communities remain protected."
        ),
        llm=llm,
        verbose=False
    )

    lead_synthesizer = Agent(
        role="Lead Environmental Auditor & EIA Coordinator",
        goal="Synthesize individual specialist evaluations into a professional Environmental Impact Assessment (EIA) report and Environmental Management Plan (EMP).",
        backstory=(
            "You are an experienced international environmental consultant certified in regulatory EIA guidelines. "
            "You turn multi-domain risk findings into clear, structured, actionable reports with mitigation matrices."
        ),
        llm=llm,
        verbose=False
    )

    # 2. Contextual Project Data
    project_context = f"""
    Project Title: {project_name}
    Project Category: {project_type}
    Location: {location}
    Surrounding Landscape: {terrain}
    Key Activities / Description: {earthwork}
    Proximity to Water Bodies / Forest: {water_proximity}
    """

    # 3. Collaborative Tasks
    task_hydro = Task(
        description=f"Evaluate water and hydrological risks based on this project data:\n{project_context}\n"
                    "Identify risks such as runoff disruption, surface or groundwater pollution, and water consumption impact. "
                    "Provide clear precautions.",
        expected_output="A bulleted summary of water-related risks, impacts on natural drainage, and recommended hydrological safeguards.",
        agent=hydrologist
    )

    task_land = Task(
        description=f"Assess land, soil, and air quality impacts based on this project data:\n{project_context}\n"
                    "Evaluate soil disturbance, potential land contamination, dust generation, and atmospheric emissions during construction and operation.",
        expected_output="A bulleted summary of land and air quality impacts with practical emission and soil management measures.",
        agent=land_air_agent
    )

    task_ecology = Task(
        description=f"Examine ecological disruption and community safeguards based on this project data:\n{project_context}\n"
                    "Evaluate plant clearing, wildlife disturbance, noise levels, and impacts on adjacent settlements.",
        expected_output="A bulleted summary of biodiversity risks, wildlife impacts, noise concerns, and community protection measures.",
        agent=ecologist
    )

    task_final_eia = Task(
        description="Synthesize the water, land/air, and ecological findings into a structured Environmental Impact Assessment (EIA) report. "
                    "Structure the report with the following clear sections:\n"
                    "1. Project Executive Summary\n"
                    "2. Baseline Environmental Setting\n"
                    "3. Key Environmental Impacts (Water, Land, Air, Biodiversity, Community)\n"
                    "4. Environmental Management Plan (EMP) in a clear Markdown Table with columns: "
                    "[Environmental Aspect | Potential Impact | Proposed Mitigation Measure | Monitoring Frequency | Responsible Entity]\n"
                    "5. Final Recommendation & Conclusion.",
        expected_output="A comprehensive, professionally formatted EIA and EMP report in clean Markdown.",
        agent=lead_synthesizer
    )

    # 4. Assemble and Run
    crew = Crew(
        agents=[hydrologist, land_air_agent, ecologist, lead_synthesizer],
        tasks=[task_hydro, task_land, task_ecology, task_final_eia],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    return str(result)
