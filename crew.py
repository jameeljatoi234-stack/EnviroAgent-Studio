import os
import time
import re
import litellm
from crewai import Agent, Crew, Process, Task, LLM

# -------------------------------------------------------------------------
# SMART RATE LIMIT INTERCEPTOR (HANDLES GROQ TPM PAUSES AUTOMATICALLY)
# -------------------------------------------------------------------------
litellm.drop_params = True

_orig_completion = litellm.completion
def _smart_completion(*args, **kwargs):
    # 1. Clean cache_breakpoint parameter
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                
    # 2. Resilient retry loop respecting Groq's cooldown time
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            return _orig_completion(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "RateLimitError" in err_str or "rate_limit_exceeded" in err_str:
                # Extract exact seconds: "Please try again in 32.85s"
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                wait_seconds = float(match.group(1)) + 2.0 if match else 25.0
                
                if attempt < max_attempts - 1:
                    time.sleep(wait_seconds)
                    continue
            raise e

litellm.completion = _smart_completion

_orig_acompletion = litellm.acompletion
async def _smart_acompletion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return await _orig_acompletion(*args, **kwargs)
litellm.acompletion = _smart_acompletion
# -------------------------------------------------------------------------

def run_eia_crew(project_name, project_type, location, terrain, earthwork, water_proximity, groq_api_key):
    """
    Executes the EIA workflow on openai/gpt-oss-20b via Groq with comprehensive technical depth.
    """
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Generous token limit allowing comprehensive reports
    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=2500
    )

    # 1. Domain Specialist Agents with Rich Context
    hydrologist = Agent(
        role="Senior Hydrology and Drainage Specialist",
        goal="Conduct a thorough assessment of catchment hydrology, runoff coefficients, flood risk, and water quality safeguards.",
        backstory="A licensed water resources engineer with extensive experience evaluating drainage channels, seasonal streams (wadis), and groundwater protection.",
        llm=llm,
        verbose=False
    )

    land_air_agent = Agent(
        role="Senior Geotechnical and Atmospheric Quality Analyst",
        goal="Evaluate cut-and-fill slope stability, topsoil conservation, erosion vulnerability, and PM10/PM2.5 particulate emissions.",
        backstory="A geotechnical engineer and atmospheric consultant expert in excavation impacts, soil mechanics, and industrial dust abatement.",
        llm=llm,
        verbose=False
    )

    ecologist = Agent(
        role="Biodiversity and Community Safeguards Officer",
        goal="Audit native flora destruction, wildlife corridor fragmentation, construction noise thresholds, and local community welfare.",
        backstory="A conservation biologist and social safeguards specialist dedicated to sustainable infrastructure and zero net habitat loss.",
        llm=llm,
        verbose=False
    )

    lead_synthesizer = Agent(
        role="Lead Environmental Compliance Auditor",
        goal="Synthesize all specialist findings into a comprehensive, publication-ready Environmental Impact Assessment and full EMP table.",
        backstory="Chief Environmental Consultant accredited in international EIA standards (EPA, IFC, World Bank) with expertise in cross-disciplinary arbitration.",
        llm=llm,
        verbose=False
    )

    # 2. Project Context
    project_context = (
        f"Project Name: {project_name}\n"
        f"Infrastructure Type: {project_type}\n"
        f"Geographic Location: {location}\n"
        f"Terrain & Topography: {terrain}\n"
        f"Earthwork & Construction Scope: {earthwork}\n"
        f"Proximity to Sensitive Water Bodies/Habitats: {water_proximity}"
    )

    # 3. Tasks Requesting Detailed Technical Analysis
    task_hydro = Task(
        description=f"Conduct a detailed hydrological impact evaluation for:\n{project_context}\n"
                    "Analyze: surface runoff patterns, catchment alteration, flood hazards, and siltation risks. "
                    "Recommend specific drainage and mitigation infrastructure.",
        expected_output="A comprehensive technical analysis of hydrological risks with specific engineering safeguards.",
        agent=hydrologist
    )

    task_land = Task(
        description=f"Conduct a detailed geotechnical and atmospheric impact assessment for:\n{project_context}\n"
                    "Analyze: slope stability in cut-and-fill zones, erosion hazards, topsoil preservation, "
                    "and air quality degradation from heavy machinery and fugitive dust (PM10/PM2.5). "
                    "Recommend specific stabilization and dust control measures.",
        expected_output="A thorough geotechnical and air quality report with concrete engineering mitigation protocols.",
        agent=land_air_agent
    )

    task_ecology = Task(
        description=f"Conduct a detailed ecological and community safeguards assessment for:\n{project_context}\n"
                    "Analyze: indigenous vegetation clearance, wildlife migratory corridor disruption, endangered species habitat loss, "
                    "and community impacts including construction noise (dB levels) and traffic disruption. "
                    "Recommend buffer zones, ecological protection, and social welfare protocols.",
        expected_output="A thorough biodiversity and social safeguards evaluation with clear preservation guidelines.",
        agent=ecologist
    )

    task_final_eia = Task(
        description="Synthesize the hydrological, geotechnical, and ecological findings into an exhaustive, professional Environmental Impact Assessment (EIA) report.\n\n"
                    "Your report MUST include the following detailed sections:\n"
                    "1. ## Executive Summary: A detailed multi-paragraph synthesis of the project and its primary environmental dimensions.\n"
                    "2. ## Hydrological & Water Resource Analysis: Detailed breakdown of runoff, flood risks, and drainage measures.\n"
                    "3. ## Geotechnical, Soil & Air Quality Analysis: In-depth evaluation of slope stability, earthwork, and dust suppression.\n"
                    "4. ## Biodiversity, Habitat & Community Welfare: Detailed review of wildlife corridors, flora, noise, and community protections.\n"
                    "5. ## Environmental Management Plan (EMP): A complete, multi-row Markdown table with the exact columns:\n"
                    "   | Environmental Aspect | Potential Site Impact | Proposed Mitigation Measure | Monitoring Frequency | Responsible Entity |\n"
                    "   (Ensure all key domains—water, soil, air, noise, wildlife, and community—are thoroughly addressed in the table).\n"
                    "6. ## Compliance Determination & Final Engineering Recommendations: Concluding verdict and site clearances.",
        expected_output="An exhaustive, professional EIA report complete with all detailed narrative sections and the comprehensive 5-column EMP Markdown table.",
        agent=lead_synthesizer
    )

    # 4. Form and Kickoff Crew
    crew = Crew(
        agents=[hydrologist, land_air_agent, ecologist, lead_synthesizer],
        tasks=[task_hydro, task_land, task_ecology, task_final_eia],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    return str(result)
