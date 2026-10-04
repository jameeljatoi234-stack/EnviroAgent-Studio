import os
import time
import re
import litellm
from crewai import Agent, Crew, Process, Task, LLM

# -------------------------------------------------------------------------
# SMART RATE LIMIT INTERCEPTOR (HANDLES 30s+ GROQ TPM PAUSES)
# -------------------------------------------------------------------------
litellm.drop_params = True

_orig_completion = litellm.completion
def _smart_completion(*args, **kwargs):
    # 1. Clean cache_breakpoint parameter
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                
    # 2. Resilient retry loop that respects Groq's exact cooldown time
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            return _orig_completion(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "RateLimitError" in err_str or "rate_limit_exceeded" in err_str:
                # Extract exact seconds from: "Please try again in 32.85s"
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                wait_seconds = float(match.group(1)) + 2.0 if match else 30.0
                
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
    Executes the EIA workflow on openai/gpt-oss-20b via Groq with auto-cooldown protection.
    """
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Cap max output tokens strictly to 800 to prevent context bloat
    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=800
    )

    # 1. Specialized Domain Agents
    hydrologist = Agent(
        role="Water Specialist",
        goal="Identify surface drainage and aquatic siltation risks.",
        backstory="Hydrological Engineer assessing flood paths and wadi disruption.",
        llm=llm,
        verbose=False
    )

    land_air_agent = Agent(
        role="Land and Air Analyst",
        goal="Quantify soil stability and dust emissions.",
        backstory="Geotechnical engineer auditing excavation and atmospheric dust.",
        llm=llm,
        verbose=False
    )

    ecologist = Agent(
        role="Ecological Safeguards Officer",
        goal="Evaluate flora loss and community noise limits.",
        backstory="Conservation officer auditing habitat fragmentation.",
        llm=llm,
        verbose=False
    )

    lead_synthesizer = Agent(
        role="Lead Environmental Auditor",
        goal="Synthesize findings into an audit-ready EMP table.",
        backstory="Senior Environmental Auditor compiling regulatory matrices.",
        llm=llm,
        verbose=False
    )

    # 2. Compact Project Parameters
    project_context = (
        f"Project: {project_name} | Type: {project_type} | Location: {location} | "
        f"Terrain: {terrain} | Scope: {earthwork} | Proximity: {water_proximity}"
    )

    # 3. Tasks with strict output bounds
    task_hydro = Task(
        description=f"Analyze hydrological risks for: {project_context}. Provide exactly 3 short bullet points.",
        expected_output="3 short bullet points on water, runoff, and drainage.",
        agent=hydrologist
    )

    task_land = Task(
        description=f"Analyze land and air risks for: {project_context}. Provide exactly 3 short bullet points.",
        expected_output="3 short bullet points on soil stability and dust.",
        agent=land_air_agent
    )

    task_ecology = Task(
        description=f"Analyze ecological risks for: {project_context}. Provide exactly 3 short bullet points.",
        expected_output="3 short bullet points on wildlife and community safety.",
        agent=ecologist
    )

    task_final_eia = Task(
        description="Synthesize the findings into a concise EIA report with:\n"
                    "1. Executive Summary (1 short paragraph)\n"
                    "2. Environmental Management Plan (EMP) as a clean Markdown table with columns: "
                    "[Environmental Aspect | Potential Impact | Proposed Mitigation Measure | Monitoring Frequency | Responsible Entity]\n"
                    "3. Final Decision (2 sentences).",
        expected_output="A concise EIA report with the complete 5-column EMP Markdown table.",
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
