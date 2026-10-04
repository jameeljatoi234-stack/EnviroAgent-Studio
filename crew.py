import os
import time
import litellm
from crewai import Agent, Crew, Process, Task, LLM

# -------------------------------------------------------------------------
# INTERCEPTOR & RATE LIMIT HANDLER
# -------------------------------------------------------------------------
litellm.drop_params = True
# Automatically retry on rate limits (HTTP 429) up to 5 times
litellm.num_retries = 5

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
    Executes the EIA workflow on openai/gpt-oss-20b via Groq with rate-limit protection.
    """
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Initialize openai/gpt-oss-20b with max_tokens cap to stay under 8,000 TPM
    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=1000
    )

    # 1. Specialized Domain Agents (concise descriptions to save input tokens)
    hydrologist = Agent(
        role="Water Resources Specialist",
        goal="Identify surface drainage, runoff, and aquatic siltation risks.",
        backstory="Licensed Hydrological Engineer assessing flood paths and stream disruption.",
        llm=llm,
        verbose=False
    )

    land_air_agent = Agent(
        role="Land and Air Quality Analyst",
        goal="Quantify soil erosion, slope stability, and particulate dust emissions.",
        backstory="Geotechnical and environmental engineer auditing earthwork and atmospheric dust.",
        llm=llm,
        verbose=False
    )

    ecologist = Agent(
        role="Ecological Safeguards Officer",
        goal="Evaluate flora loss, wildlife corridor severance, and community noise.",
        backstory="Conservation biologist auditing habitat fragmentation and settlement welfare.",
        llm=llm,
        verbose=False
    )

    lead_synthesizer = Agent(
        role="Lead Environmental Auditor",
        goal="Synthesize specialist inputs into a regulatory EIA and 5-column EMP table.",
        backstory="Senior Environmental Consultant accredited in international EIA compliance.",
        llm=llm,
        verbose=False
    )

    # 2. Compact Project Context (conserves token budget)
    project_context = (
        f"Project: {project_name} | Type: {project_type} | Location: {location} | "
        f"Terrain: {terrain} | Scope: {earthwork} | Proximity: {water_proximity}"
    )

    # 3. Tasks with explicit token-efficient output guidelines
    task_hydro = Task(
        description=f"Analyze hydrological risks for: {project_context}. Summarize drainage, runoff, and water risks in 3-4 bullet points.",
        expected_output="3-4 concise bullet points of water and drainage findings with recommended safeguards.",
        agent=hydrologist
    )

    task_land = Task(
        description=f"Analyze land and air risks for: {project_context}. Summarize slope stability, erosion, and dust in 3-4 bullet points.",
        expected_output="3-4 concise bullet points of soil and air findings with practical suppression measures.",
        agent=land_air_agent
    )

    task_ecology = Task(
        description=f"Analyze ecological risks for: {project_context}. Summarize biodiversity, habitat, and community noise in 3-4 bullet points.",
        expected_output="3-4 concise bullet points of ecological and community findings with safeguards.",
        agent=ecologist
    )

    task_final_eia = Task(
        description="Synthesize the hydrological, geotechnical, and ecological findings into a structured EIA report. "
                    "Include:\n"
                    "1. Executive Summary\n"
                    "2. Key Environmental Findings\n"
                    "3. Environmental Management Plan (EMP) as a clean Markdown table with columns: "
                    "[Environmental Aspect | Potential Impact | Proposed Mitigation Measure | Monitoring Frequency | Responsible Entity]\n"
                    "4. Final Recommendation.",
        expected_output="A structured EIA report with the complete 5-column EMP Markdown table.",
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
