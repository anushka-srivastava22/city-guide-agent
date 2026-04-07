import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

# --- Save user prompt/state ---

def add_prompt_to_state(tool_context: ToolContext, prompt: str) -> dict[str, str]:
    """Stores the user's travel query."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] PROMPT: {prompt}")
    return {"status": "success"}

# --- Wikipedia Tool (for city knowledge) ---
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# --- 1. City Researcher Agent ---
city_researcher = Agent(
    name="city_researcher",
    model=model_name,
    description="Researches cities, attractions, culture, and travel info.",
    instruction="""
    You are a travel research assistant.

    Your job is to fully answer the user's PROMPT about a city or destination.

    You can use:
    - Wikipedia tool for general city knowledge (history, culture, landmarks, food, geography).

    Instructions:
    - Understand the user's intent (e.g., itinerary, food, attractions, history).
    - Use the Wikipedia tool when needed.
    - Extract useful travel insights such as:
        * Top attractions
        * Local food
        * Cultural highlights
        * Travel tips
    - Combine all findings into structured research notes.

    PROMPT:
    { PROMPT }
    """,
    tools=[wikipedia_tool],
    output_key="research_data"
)

# --- 2. Travel Response Formatter ---
city_response_formatter = Agent(
    name="city_response_formatter",
    model=model_name,
    description="Formats city travel info into an engaging guide.",
    instruction="""
    You are a friendly and knowledgeable travel guide.

    Using the RESEARCH_DATA, create a helpful and engaging response.

    Guidelines:
    - Start with a quick overview of the city.
    - Suggest top attractions and experiences.
    - Include food recommendations or local specialties.
    - Add helpful travel tips if relevant.
    - Keep it conversational and inspiring.

    If some data is missing, still provide the best possible answer.

    RESEARCH_DATA:
    { research_data }
    """
)

# --- Workflow ---
city_guide_workflow = SequentialAgent(
    name="city_guide_workflow",
    description="Handles travel-related queries about cities.",
    sub_agents=[
        city_researcher,
        city_response_formatter,
    ]
)

# --- Root Agent ---
root_agent = Agent(
    name="city_guide_greeter",
    model=model_name,
    description="Entry point for the City Guide AI.",
    instruction="""
    - Greet the user as a travel assistant.
    - Ask what city or destination they are interested in.
    - When the user responds:
        * Use 'add_prompt_to_state' tool to save their query.
        * Then transfer control to 'city_guide_workflow'.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[city_guide_workflow]
)
