import os
import time
import re
import litellm
from crewai import Agent, Crew, Process, Task, LLM

# -------------------------------------------------------------------------
# SMART RATE LIMIT INTERCEPTOR
# -------------------------------------------------------------------------
litellm.drop_params = True

_orig_completion = litellm.completion
def _smart_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            return _orig_completion(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "RateLimitError" in err_str or "rate_limit_exceeded" in err_str:
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                wait_seconds = float(match.group(1)) + 1.5 if match else 20.0
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
    Executes EIA workflow with fast throughput and professional depth.
    """
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Balanced token limit: prevents long 30s rate-limit timeouts
    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=1500
    )

    # 1. Domain Specialist Agents
    hydrologist = Agent(
        role="Senior Hydrology and Drainage Specialist",
        goal="Assess catchment hydrology, surface runoff alteration, and water contamination risks.",
        backstory="Water resources engineer expert in drainage networks, wadi dynamics, and flood paths.",
        llm=llm,
        verbose=False
    )

    land_air_agent = Agent(
        role="Geotechnical and Atmospheric Quality Analyst",
        goal="Quantify cut-and-fill slope stability, erosion vulnerability, and PM10/PM2.5 dust.",
        backstory="Geotechnical consultant auditing excavation safety, soil mechanics, and fugitive emissions.",
        llm=llm,
        verbose=False
    )

    ecologist = Agent(
        role="Biodiversity and Community Safeguards Officer",
        goal="Evaluate flora loss, wildlife corridor severance, and community noise limits.",
        backstory="Conservation biologist auditing habitat protection and settlement welfare.",
        llm=llm,
        verbose=False
    )

    lead_synthesizer = Agent(
        role="Lead Environmental Compliance Auditor",
        goal="Synthesize specialist inputs into a comprehensive EIA report with a full EMP table.",
        backstory="Chief Environmental Consultant accredited in international EIA compliance standards.",
        llm=llm,
        verbose=False
    )

    # 2. Project Parameters
    project_context = (
        f"Project: {project_name} | Type: {project_type} | Location: {location} | "
        f"Terrain: {terrain} | Scope: {earthwork} | Proximity: {water_proximity}"
    )

    # 3. Tasks: Concise specialist briefs + Detailed master report
    task_hydro = Task(
        description=f"Evaluate hydrological risks for: {project_context}. "
                    "Detail surface runoff, flood vulnerability, and siltation risks with recommended structures.",
        expected_output="A structured technical summary of water and drainage findings with specific engineering safeguards.",
        agent=hydrologist
    )

    task_land = Task(
        description=f"Evaluate geotechnical and atmospheric risks for: {project_context}. "
                    "Detail cut-and-fill slope stability, erosion hazards, and PM10/PM2.5 dust suppression.",
        expected_output="A structured technical summary of soil stability and air quality with suppression measures.",
        agent=land_air_agent
    )

    task_ecology = Task(
        description=f"Evaluate biodiversity and social safeguards for: {project_context}. "
                    "Detail native vegetation loss, wildlife corridors, and community noise limits.",
        expected_output="A structured technical summary of ecological and community welfare safeguards.",
        agent=ecologist
    )

    task_final_eia = Task(
        description="Synthesize the hydrological, geotechnical, and ecological findings into a full Environmental Impact Assessment report.\n\n"
                    "Structure the report with these exact sections:\n"
                    "1. ## Executive Summary\n"
                    "2. ## Hydrological & Water Resource Analysis\n"
                    "3. ## Geotechnical, Soil & Air Quality Analysis\n"
                    "4. ## Biodiversity, Habitat & Community Welfare\n"
                    "5. ## Environmental Management Plan (EMP) Matrix:\n"
                    "   Create a clean, multi-row Markdown table with columns:\n"
                    "   | Environmental Aspect | Potential Site Impact | Proposed Mitigation Measure | Monitoring Frequency | Responsible Entity |\n"
                    "   (Address Water, Soil, Air, Noise, and Ecology in detail).\n"
                    "6. ## Compliance Verdict & Final Engineering Recommendations",
        expected_output="A full, professional EIA report with detailed narrative sections and the complete 5-column EMP Markdown table.",
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
