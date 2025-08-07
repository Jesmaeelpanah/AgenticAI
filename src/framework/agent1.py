import asyncio
import contextlib
import signal
import sys
import uuid

from src import Agent, function_tool, OpenAIChatCompletionsModel
from src.utils.embeddings_cloudflare import get_embedding
from src.utils.gemini_llm import ask_gemini
from src.utils.weaviate_kb import fetch_similar_issues, close_client
from src.prompts import REACT_INSTRUCTIONS
#from langfuse import Langfuse, Trace
from pydantic import BaseModel
from openai import AsyncOpenAI


class PlannerInput(BaseModel):
    issue_id: str
    issue_type: str
    issue_description: str
    resolution: str


class PlannerOutput(BaseModel):
    issue_id: str
    issue_type: str
    remediation_plan: str
    status: str


# langfuse = Langfuse()
async_openai_client = AsyncOpenAI()


# Graceful shutdown handlers (keep as is)
async def _cleanup_clients() -> None:
    close_client()


def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_sigint)


async def recommend_solutions(issue: PlannerInput) -> str:
    # trace = Trace(
    #     name="Agent1_Recommendation",
    #     input=issue.model_dump(),
    #     id=str(uuid.uuid4()),
    # )
    #trace.start()
    try:
        issue_dict = issue.model_dump()
        issue_id = issue_dict.get("issue_id", "unknown")
        issue_description = issue_dict.get("issue_description", "")
        combined_info = "\n".join(f"{k}: {v}" for k, v in issue_dict.items() if v)

        if not issue_description:
            return f"Issue {issue_id} has no description provided."

        # Summarize with Gemini
        # Note: Span management removed since your Langfuse version might not support it
        prompt = f"""
            Summarize the following banking issue into a single, dense paragraph.
            Focus on the core problem, its root cause, and the actions taken for the final resolution.
            ISSUE DATA:
            {combined_info}
        """
        summary = ask_gemini(prompt)

        # Embed
        query_vector = get_embedding(summary)

        # Retrieve
        similar_issues = fetch_similar_issues(query_vector)

        if not similar_issues:
            return f"No similar resolved issues found for Issue {issue_id}."

        output = [
            f"Issue ID: {i['issue_id']} | Type: {i['issue_type']}\nSuggested Remediation:\n{i['remediation_plan']}"
            for i in similar_issues
        ]

        result = "\n---\n".join(output)
        #trace.set_output(result)
        return result

    except Exception as e:
        #trace.set_error(str(e))
        raise

    finally:
        #trace.end()
        print("enmf")


worker_agent = Agent(
    name="IssueWorkerAgent",
    instructions=(
        "Analyze a banking issue dictionary: summarize it, embed it, "
        "and find similar resolved issues to recommend solutions."
    ),
    tools=[recommend_solutions],
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=async_openai_client
    ),
)

planner_agent = Agent(
    name="IssuePlannerAgent",
    instructions=REACT_INSTRUCTIONS,
    tools=[
        worker_agent.as_tool(
            tool_name="analyze_issue",
            tool_description="Analyze a banking issue dictionary and recommend remediation suggestions."
        )
    ],
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=async_openai_client
    ),
)


@function_tool
async def invoke_planner(input: PlannerInput) -> PlannerOutput:
    remediation_text = await recommend_solutions(input)
    return PlannerOutput(
        issue_id=input.issue_id,
        issue_type=input.issue_type,
        remediation_plan=remediation_text,
        status="Resolved"
    )
