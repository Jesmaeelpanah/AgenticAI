import asyncio
import contextlib
import logging
import signal
import sys

import agents
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional
from datetime import date


from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
    setup_langfuse_tracer,
)
#from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO)

AGENT_LLM_NAME = "gemini-2.5-flash"
WEAVIATE_COLLECTION_NAME="td_1_banking_issue_kb"

PLANNER_INSTRUCTIONS = """\
You are a research planner. \
Given a user's query, produce a list of search terms that can be used to retrieve \
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

class Issue(BaseModel):
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


async def summarize_issue(issue: Issue, client: AsyncOpenAI) -> IssueSummary:
    prompt = f"""
You are an expert IT auditor.

Here is a detailed issue record from an internal report:

---
**Issue ID**: {issue.issue_id}
**Type**: {issue.issue_type}
**Department**: {issue.department}
**System**: {issue.system}
**Root Cause**: {issue.root_cause}
**Severity**: {issue.severity}
**Status**: {issue.status}
**Created On**: {issue.creation_date}
**Due Date**: {issue.due_date}
**Resolved On**: {issue.remediation_completion_date}

**Issue Description**:
{issue.issue_description}

**Remediation Plan**:
{issue.remediation_plan}

**Comments Log**:
{issue.comments_log}

**Contradiction Flag**: {issue.contradiction_flag}
**Contradiction Type**: {issue.contradiction_type}
---

1. Write a short **Executive Summary** in 3–5 bullet points.
2. Write a 2–3 paragraph **Detailed Summary** useful for compliance or internal audit teams.

Do not invent facts. Use only the provided information.
"""

    response = await client.chat.completions.create(
        model=AGENT_LLM_NAME,  # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content.strip()

    # Rough splitting: customize if formatting differs
    if "Detailed Summary" in content:
        exec_sum, detailed_sum = content.split("Detailed Summary", 1)
        return IssueSummary(
            executive_summary=exec_sum.replace("Executive Summary", "").strip(),
            detailed_summary=detailed_sum.strip()
        )
    else:
        return IssueSummary(executive_summary="N/A", detailed_summary=content)


if __name__ == "__main__":
    sample_input = {
        "issue_id": "ISSUE-0002",
        "issue_type": "Security Vulnerability",
        "department": "IT",
        "root_cause": "Data Corruption",
        "system": "Reporting Database",
        "severity": "Low",
        "status": "Open",
        "creation_date": "2025-05-01",
        "remediation_completion_date": None,
        "due_date": "2025-10-11",
        "issue_description": "**Issue Title:** Security Vulnerability Due to Misconfigured Firewall Rules in IT Network Infrastructure\n\n**Issue Description:**\n\nOn November 20, 2023...",
        "remediation_plan": "**Issue Description:**\n\nIn September 2023, several customers reported discrepancies in their account balances and transaction histories...",
        "comments_log": "**Issue ID: ISSUE-0002**\n\n**Comment Log:**\n\n1. **Date: 2023-10-15 09:30 AM**...",
        "contradiction_flag": "No",
        "contradiction_type": ""
    }

    issue = Issue(**sample_input)
    async_openai_client = AsyncOpenAI()
    setup_langfuse_tracer()

    summary = asyncio.run(summarize_issue(issue, async_openai_client))
    print("\n--- Executive Summary ---\n", summary.executive_summary)
    print("\n--- Detailed Summary ---\n", summary.detailed_summary)