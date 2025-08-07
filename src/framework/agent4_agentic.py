import asyncio
import contextlib
import logging
import signal
import sys

import agents
import gradio as gr
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from typing import Optional
from datetime import date
import json

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_items_to_gradio_messages,
    pretty_print,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client

load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO)

AGENT_LLM_NAME = "gemini-2.5-flash"
WEAVIATE_COLLECTION_NAME="td_1_banking_issue_kb"

# === AGENT INSTRUCTIONS ===
PLANNER_INSTRUCTIONS = """\
You are a research planner. \
Given a user's query, produce a list of search terms that can be used to retrieve
relevant information from a knowledge base to answer the question. \
As you are not able to clarify from the user what they are looking for, \
your search terms should be broad and cover various aspects of the query. \
Output between 5 to 10 search terms to query the knowledge base. \
Note that the knowledge base is a Wikipedia dump and cuts off at May 2025.
"""

RESEARCHER_INSTRUCTIONS = """\
You are a research assistant with access to a knowledge base. \
Given a potentially broad search term, use the search tool to \
retrieve relevant information from the knowledge base and produce a short
summary of at most 300 words.
"""

WRITER_INSTRUCTIONS = """\
You are an expert at synthesizing information and writing coherent reports. \
Given a user's query and a set of search summaries, synthesize these into a \
coherent report (at least a few paragraphs long) that answers the user's question. \
Do not make up any information outside of the search summaries.
"""

# === DATA MODELS ===
class IssueInput(BaseModel):
    issue_id: str
    issue_type: str
    department: str
    root_cause: str
    system: str
    severity: str
    status: str
    creation_date: date
    remediation_completion_date: Optional[date]
    due_date: date
    issue_description: str
    remediation_plan: str
    comments_log: str
    contradiction_flag: str
    contradiction_type: Optional[str]

class IssueSummary(BaseModel):
    executive_summary: str
    detailed_summary: str

# === Summary Models ===
class SearchItem(BaseModel):
    search_term: str
    reasoning: str

class SearchPlan(BaseModel):
    search_steps: list[SearchItem]

    def __str__(self) -> str:
        return "\n".join(
            f"Search Term: {step.search_term}\nReasoning: {step.reasoning}\n"
            for step in self.search_steps
        )

class ResearchReport(BaseModel):
    summary: str
    full_report: str


# === AGENT EXECUTION FUNCTIONS ===
async def _create_search_plan(planner_agent: agents.Agent, query: str) -> SearchPlan:
    with langfuse_client.start_as_current_span("create_search_plan", input=query) as span:
        response = await agents.Runner.run(planner_agent, input=query)
        search_plan = response.final_output_as(SearchPlan)
        span.update(output=search_plan)
    return search_plan


async def _generate_final_report(writer_agent: agents.Agent, search_results: list[str], query: str) -> agents.RunResult:
    input_data = f"Original question: {query}\n"
    input_data += "Search summaries:\n" + "\n".join(f"{i + 1}. {result}" for i, result in enumerate(search_results))

    with langfuse_client.start_as_current_span("generate_final_report", input=input_data) as span:
        response = await agents.Runner.run(writer_agent, input=input_data)
        span.update(output=response.final_output)
    return response


async def _cleanup_clients() -> None:
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# === MAIN PIPELINE ===
async def _main(issue_id: str, ui_messages: list[str]):
    
    #Planner Agent
    planner_agent = agents.Agent(
        name="Planner Agent",
        instructions=PLANNER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client
        ),
        output_type=SearchPlan,
    )

    # Research Agent
    research_agent = agents.Agent(
        name="Research Agent",
        instructions=RESEARCHER_INSTRUCTIONS,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
        model_settings=agents.ModelSettings(tool_choice="required"),
    )

    # Writer Agent
    writer_agent = agents.Agent(
        name="Writer Agent",
        instructions=WRITER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client
        ),
        output_type=ResearchReport,
    )

    ui_messages.append(f"USER: Summarize issue with ID {issue_id}")
    yield ui_messages

    with langfuse_client.start_as_current_span("Multi-Agent-Trace", input=issue_id) as span:
        issue = await async_knowledgebase.get_issue_by_id(issue_id)
        search_plan = await _create_search_plan(planner_agent, issue.issue_description)
        ui_messages.append(f"ASSISTANT: Search Plan: {search_plan}")
        pretty_print(ui_messages)
        yield ui_messages

        search_results = []
        for step in search_plan.search_steps:
            with langfuse_client.start_as_current_span("execute_search_step", input=step.search_term) as step_span:
                response = await agents.Runner.run(research_agent, input=step.search_term)
                search_result = response.final_output
                step_span.update(output=search_result)
            search_results.append(search_result)
            ui_messages += oai_agent_items_to_gradio_messages(response.new_items)
            yield ui_messages

        writer_response = await _generate_final_report(writer_agent, search_results, issue.issue_description)
        span.update(output=writer_response.final_output)
        report = writer_response.final_output_as(ResearchReport)

        ui_messages.append(f"ASSISTANT:\nSummary:\n{report.summary}\n Full Report:\n{report.full_report}")
        pretty_print(ui_messages)
        yield ui_messages

# === BOOTSTRAP ===
if __name__ == "__main__":
    configs = Configs.from_env_var()
    async_weaviate_client = get_weaviate_async_client(
        http_host=configs.weaviate_http_host,
        http_port=configs.weaviate_http_port,
        http_secure=configs.weaviate_http_secure,
        grpc_host=configs.weaviate_grpc_host,
        grpc_port=configs.weaviate_grpc_port,
        grpc_secure=configs.weaviate_grpc_secure,
        api_key=configs.weaviate_api_key,
    )
    async_knowledgebase = AsyncWeaviateKnowledgeBase(
        async_weaviate_client,
        collection_name=WEAVIATE_COLLECTION_NAME,
    )

    async_openai_client = AsyncOpenAI()
    setup_langfuse_tracer()

    with gr.Blocks(title="Banking Issue Summarizer") as app:
                chatbot = gr.Chatbot(type="messages", label="Agent", height=600)
                chat_message = gr.Textbox(lines=1, label="Enter Issue ID")
                print(chat_message, chatbot)
                chat_message.submit(_main, [chat_message, chatbot], [chatbot])
                '''
                def handle_submit(issue_id):
                    try:
                        async def run_main():
                            messages = []
                            async for step in _main(issue_id, []):
                                messages = step  # overwrite each yield with latest state
                            return messages

                        return asyncio.run(run_main())
                    except Exception as e:
                        return [f"ERROR: {str(e)}"]
                '''
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        app.launch(server_name="0.0.0.0", share=True)
    finally:
        asyncio.run(_cleanup_clients())