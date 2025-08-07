import requests
import os
import json
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

CLOUDFLARE_EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
CLOUDFLARE_API_TOKEN = os.getenv("EMBEDDING_API_KEY")
CLOUDFLARE_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME") 
CLOUDFLARE_EMBEDDING_URL = f"{CLOUDFLARE_EMBEDDING_BASE_URL}/embeddings"
# Google Gemini configuration
GEMINI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_URL = os.getenv("OPENAI_BASE_URL")

GEMINI_API_URL = f"{GEMINI_API_URL}chat/completions"

# --- Helper Function for API Calls with Exponential Backoff ---
def make_api_call(url, headers, payload, max_retries=1, initial_delay=1):
    """
    Makes an API call with exponential backoff for retries.
    """
    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API call failed (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                delay = initial_delay * (2 ** i)
                print(f"Retrying in {delay} seconds...")
                import time
                time.sleep(delay)
            else:
                print("Max retries exceeded. Exiting.")
                raise

# --- 1. Test Cloudflare Workers AI (bge-3) Embedding Model ---
def get_embedding(text_to_embed):
    """
    Sends text to Cloudflare Workers AI (bge-3) and retrieves embeddings
    using the OpenAI-compatible /v1/embeddings endpoint.
    """
    #print(f"\n--- Testing Cloudflare Workers AI ({CLOUDFLARE_MODEL_NAME}) Embeddings ---")
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    # For OpenAI-compatible /embeddings endpoint, the payload structure is different.
    # It expects 'model' and 'input' (which can be a string or list of strings).
    payload = {
        "model": CLOUDFLARE_MODEL_NAME,
        "input": [text_to_embed] # Input should be a list, even for a single text
    }

    try:
        response_data = make_api_call(CLOUDFLARE_EMBEDDING_URL, headers, payload)
        # OpenAI-compatible embedding responses have a 'data' field containing embeddings.
        embeddings_data = response_data.get("data")
        if embeddings_data and len(embeddings_data) > 0:
            embeddings = embeddings_data[0].get("embedding")
            if embeddings:
                #print(f"Successfully generated embeddings for: '{text_to_embed}'")
                #print(f"Embedding dimensions: {len(embeddings)}")
                #print(f"First 5 embedding values: {embeddings[:5]}...")
                return embeddings
            else:
                #print("Failed to retrieve 'embedding' from Cloudflare response data.")
                #print(f"Response: {response_data}")
                return None
        else:
            #print("Failed to retrieve embeddings from Cloudflare. 'data' field missing or empty.")
            #print(f"Response: {response_data}")
            return None
    except Exception as e:
        print(f"An error occurred while testing Cloudflare embeddings: {e}")
        return None
# --- 2. Test Google Gemini LLM ---
def get_gemini_response(prompt_text, model_name="gemini-2.5-flash"):
    """
    Sends a prompt to Google Gemini and retrieves a text response
    using the OpenAI-compatible /v1beta/openai/chat/completions endpoint.
    """
    #print(f"\n--- Testing Google Gemini LLM ({model_name}) ---")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}" # API key in Authorization header for OpenAI-compatible endpoint
    }
    # For OpenAI-compatible chat completions, the payload uses 'messages'
    # with 'role' and 'content', and the model is specified in the payload.
    payload = {
        "model": model_name, # Model name passed in the payload
        "messages": [
            {"role": "user", "content": prompt_text}
        ]
    }

    try:
        # No API key in URL query parameter for OpenAI-compatible endpoint
        response_data = make_api_call(GEMINI_API_URL, headers, payload)
        
        # OpenAI-compatible chat completions response parsing
        if response_data and response_data.get("choices"):
            generated_text = response_data["choices"][0]["message"]["content"]
            #print(f"Prompt: '{prompt_text}'")
            #print(f"Generated Response:\n{generated_text}")
            return generated_text
        else:
            print("Failed to retrieve response from Gemini. 'choices' field missing or empty.")
            print(f"Response: {response_data}")
            return None
    except Exception as e:
        print(f"An error occurred while testing Google Gemini: {e}")
        return None

# --- Data Structure---
banking_issues_data = [
    {
        "issue_id":"ISSUE-0001", "issue_type":"Service Outage", "department":"Operations",
        "root_cause":"Third-Party Vendor Issue", "system":"Core Banking System", "severity":"Medium",
        "status":"Open", "creation_date":"2025-06-25", "remediation_completion_date":None,
        "due_date":"2025-09-22",
        "issue_description":"**New Incident Description:**\n\nTitle: \"Automated Payment Processing Halted Due to API Timeout\"\n\nIncident Summary: On November 3, 2023, at approximately 11:45 AM, the Operations Department identified a service outage affecting the automated payment processing system...",
        "remediation_plan":"**Issue Description:**\n\nOn October 15, 2023, customers of UrbanBank began experiencing delays in processing online bill payments...",
        "comments_log":"**Issue ID: ISSUE-0001**\n\n1. **Date: 2023-11-10, 09:15 AM**  \n   **Comment:** Initial customer report received concerning intermittent issues with online banking login...",
        "contradiction_flag":"No", "contradiction_type":""
    },
    {
        "issue_id":"ISSUE-0002", "issue_type":"Security Vulnerability", "department":"IT",
        "root_cause":"Data Corruption", "system":"Reporting Database", "severity":"Low",
        "status":"Open", "creation_date":"2025-05-01", "remediation_completion_date":None,
        "due_date":"2025-10-11",
        "issue_description":"**Issue Title:** Security Vulnerability Due to Misconfigured Firewall Rules in IT Network Infrastructure\n\n**Issue Description:**\n\nOn November 20, 2023...",
        "remediation_plan":"**Issue Description:**\n\nIn September 2023, several customers reported discrepancies in their account balances and transaction histories. Initial investigations revealed that the issue stemmed from data corruption...",
        "comments_log":"**Issue ID: ISSUE-0002**\n\n**Comment Log:**\n\n1. **Date: 2023-10-15 09:30 AM**  \n   **Commenter: John Smith, Banking Operations Analyst**  \n   *Initial identification of issue:*  \n   The issue was raised following a customer report of being unable to initiate international wire transfers...",
        "contradiction_flag":"No", "contradiction_type":""
    },
    {
        "issue_id":"ISSUE-0003", "issue_type":"System Bug", "department":"Fraud Prevention",
        "root_cause":"Human Error", "system":"ATM Network", "severity":"High",
        "status":"On Hold", "creation_date":"2025-07-07", "remediation_completion_date":None,
        "due_date":"2025-08-16",
        "issue_description":"**Issue Title:** System Bug in Fraud Detection Algorithm Due to Human Error\n\n**Severity Level:** Critical\n\n**Department:** Fraud Prevention\n\n**Affected System:** Fraud Detection Engine\n\n**Issue Description:**\n\nOn November 3, 2023...",
        "remediation_plan":"**Issue Description:**\nOn August 15, 2023, a processing error occurred in the wire transfer operations department, resulting in the duplication of approximately 500 wire transfer transactions...",
        "comments_log":"**Issue ID:** ISSUE-0003  \n**Status:** On Hold  \n\n**Comment Log:**\n\n1. **Date:** 2023-09-12  \n   **Comment:** Initial investigation into ISSUE-0003 began following reports of intermittent connectivity issues with the mobile banking application...",
        "contradiction_flag":"No", "contradiction_type":""
    },
    {
        "issue_id":"ISSUE-0004", "issue_type":"Software Glitch", "department":"IT",
        "root_cause":"Software Defect", "system":"Payment Gateway", "severity":"Critical",
        "status":"Closed", # <-- NOTE: Changed to 'Closed' for Agent 1 demo
        "creation_date":"2025-04-24", "remediation_completion_date":"2025-05-10",
        "due_date":"2025-10-21",
        "issue_description":"**Incident Title:** Mobile Banking App Transaction Freeze\n\n**Incident Description:**\n\nOn November 14, 2023...",
        "remediation_plan":"**Issue Description:**\n\nA software defect was identified in the bank's online transaction processing system, causing intermittent failures...",
        "comments_log": None, "contradiction_flag":"No", "contradiction_type":""
    }
    ]


# FOR AGENT 1
def calculate_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def agent_1_recommend_solutions(new_issue_description: str, resolved_issues_data: list):
    """
    Agent 1: Simulates a vector DB search in memory to find similar resolved issues.
    """
    #print("\n--- Agent 1: Recommending Solutions (In-Memory Simulation) ---")

    # 1. Create in-memory "vector database" from resolved issues
    print("Step 1: Creating in-memory vector store from resolved issues...")
    resolved_issues_with_embeddings = []
    for issue in resolved_issues_data:
        if issue.get("status") == "Closed":
            #print(f"  - Processing resolved issue for embedding: {issue['issue_id']}")
            summary_prompt = (
                f"Summarize the following issue description and remediation plan into a single, dense paragraph. "
                f"Focus on the root cause, actions taken, and final outcome.\n\n"
                f"Description: {issue['issue_description']}\n\n"
                f"Remediation: {issue['remediation_plan']}"
            )
            summary = get_gemini_response(summary_prompt)
            if summary:
                embedding = get_embedding(summary)
                if embedding:
                    resolved_issues_with_embeddings.append({"issue": issue, "embedding": embedding})

    if not resolved_issues_with_embeddings:
        print("No resolved issues could be processed for the recommendation knowledge base.")
        return

    # 2. Prepare the new issue for searching
    #print("\nStep 2: Preparing new issue for similarity search...")
    query_summary_prompt = (
        f"Summarize the following new issue description into a single, dense paragraph "
        f"focusing on the core problem.\n\nDescription: {new_issue_description}"
    )
    query_summary = get_gemini_response(query_summary_prompt)
    if not query_summary:
        #print("Could not generate summary for new issue. Aborting.")
        return
    query_vector = get_embedding(query_summary)
    if not query_vector:
        #print("Could not generate embedding for new issue. Aborting.")
        return

    # 3. Calculate similarity scores and find the best matches
    #print("\nStep 3: Calculating similarity scores...")
    results = []
    for item in resolved_issues_with_embeddings:
        similarity = calculate_cosine_similarity(query_vector, item['embedding'])
        results.append({"issue": item['issue'], "similarity": similarity})

    # Sort results by similarity score in descending order
    sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    # 4. Display the top recommendations
    #print("\n--- Top 3 Similar Resolved Issues ---")
    for result in sorted_results[:1]:
        issue = result['issue']
        print(f"  - Issue ID: {issue['issue_id']} (Similarity Score: {result['similarity']:.4f})")
        print(f"    Type: {issue['issue_type']}")
        print(f"    Suggested Remediation Plan:\n{issue['remediation_plan']}\n")

class Inconsistency(BaseModel):
    """Data model for a single inconsistency found in an issue."""
    contradiction: str = Field(description="A concise description of the contradiction.")
    justification: str = Field(description="The reasoning or evidence from the text that supports this finding.")
    involved_fields: List[str] = Field(description="List of fields where the contradiction was found (e.g., ['issue_description', 'comments_log']).")

class ContradictionAnalysis(BaseModel):
    """Data model for the complete contradiction analysis of an issue."""
    issue_id: str = Field(description="The ID of the issue being analyzed.")
    inconsistencies_found: bool = Field(description="True if any inconsistencies were detected, otherwise False.")
    findings: Optional[List[Inconsistency]] = Field(default=None, description="A list of detected inconsistencies.")

def agent_2_detect_contradictions(issue: dict):
    """
    Agent 2: Chunks text and uses an LLM to find inconsistencies.
    """
    #print(f"\n--- Agent 2: Detecting Contradictions for {issue['issue_id']} ---")
    context_text = (
        f"Issue ID: {issue.get('issue_id')}\n\n"
        f"--- Issue Description ---\n{issue.get('issue_description')}\n\n"
        f"--- Comments Log ---\n{issue.get('comments_log')}\n\n"
        f"--- Remediation Plan ---\n{issue.get('remediation_plan')}\n\n"
    )

    prompt = (
        f"Analyze the following banking issue data for any contradictions or inconsistencies "
        f"between the 'Issue Description', 'Comments Log', and 'Remediation Plan'. Check for conflicting dates, timelines, "
        f"root causes, or described problems.\n\n"
        f"Format your response as a JSON object that strictly follows this Pydantic schema:\n"
        f"```json\n{json.dumps(ContradictionAnalysis.model_json_schema(), indent=2)}\n```\n\n"
        f"If no contradictions are found, set 'inconsistencies_found' to false and 'findings' to null.\n\n"
        f"--- DATA TO ANALYZE ---\n"
        f"{context_text}"
    )
    llm_output_str = get_gemini_response(prompt)

    # Clean the output in case the LLM wraps it in markdown
    if llm_output_str and "```json" in llm_output_str:
        llm_output_str = llm_output_str.split("```json\n")[1].split("\n```")[0]

    try:
        analysis_result = ContradictionAnalysis.model_validate_json(llm_output_str)
        if analysis_result.inconsistencies_found:
            print("Contradictions Found:")
            for finding in analysis_result.findings:
                print(f"  - Contradiction: {finding.contradiction}")
                print(f"    Justification: {finding.justification}")
                print(f"    Involved Fields: {finding.involved_fields}")
        else:
            print("No contradictions were detected.")
    except Exception as e:
        print(f"Failed to parse LLM output for contradiction detection: {e}")
        print(f"Raw Output:\n{llm_output_str}")

def agent_3_monitor_due_dates(all_issues_data: list):
    """
    Agent 3 (Stretch): Monitors issues nearing their due date.
    """
    print("\n--- Agent 3 (Stretch): Monitoring Due Dates ---")
    today = datetime.now() # Current time is August 4, 2025
    thirty_days_from_now = today + timedelta(days=30)

    issues_nearing_due_date = []
    for issue in all_issues_data:
        if issue.get("status") != "Closed" and issue.get("due_date"):
            due_date = datetime.strptime(issue["due_date"], "%Y-%m-%d")
            if today <= due_date <= thirty_days_from_now:
                issues_nearing_due_date.append(issue)

    if not issues_nearing_due_date:
        print("No open issues are due within the next 30 days.")
        return

    print(f"Found {len(issues_nearing_due_date)} issues nearing their due date:")
    for issue in issues_nearing_due_date:
        due_date = datetime.strptime(issue["due_date"], "%Y-%m-%d")
        days_left = (due_date - today).days
        print(f"  - ALERT: Issue {issue['issue_id']} is due on {issue['due_date']} ({days_left} days remaining).")
        # Email sending logic could be triggered here if SMTP is configured
        # _send_email(...)

# FOR AGENT 4
class GeneratedSummaries(BaseModel):
    executive_summary: str = Field(description="A one-sentence summary focusing on the core problem and its most significant impact.")
    detailed_summary: str = Field(description="A detailed paragraph outlining the problem, business impact, and timelines.")    

def agent_4b_summarize_issue_single_call(issue: dict):
    """
    Agent 4 (Alternative): Generates both summaries in a single, structured API call.
    """
    #print(f"\n--- Agent 4 (Alternative): Generating Summaries for {issue['issue_id']} in a Single Call ---")
    combined_info = "\n".join([f"{k}: {v}" for k, v in issue.items() if v])

    # A single, more complex prompt that asks for a JSON object
    prompt = (
        f"Analyze the following banking issue and generate two summaries based on its content. "
        f"Format your response as a single, valid JSON object that strictly follows this Pydantic schema:\n"
        f"```json\n{json.dumps(GeneratedSummaries.model_json_schema(), indent=2)}\n```\n\n"
        f"--- ISSUE DATA ---\n"
        f"{combined_info}"
    )

    # Get the response from the LLM
    llm_output_str = get_gemini_response(prompt)
    if not llm_output_str:
        print("Failed to get response from LLM.")
        return None, None

    # Clean the output in case the LLM wraps it in markdown ```json ... ```
    if "```json" in llm_output_str:
        llm_output_str = llm_output_str.split("```json\n")[1].split("\n```")[0]

    # Parse the JSON string into the Pydantic model
    try:
        summaries = GeneratedSummaries.model_validate_json(llm_output_str)
        print(f"\nExecutive Summary:\n{summaries.executive_summary}")
        print(f"\nDetailed Summary:\n{summaries.detailed_summary}")
        return summaries.executive_summary, summaries.detailed_summary
    except Exception as e:
        print(f"Failed to parse the LLM's JSON output: {e}")
        print(f"Raw Output from LLM:\n{llm_output_str}")
        return None, None

# --- Example Usage ---
if __name__ == "__main__":
    # # Example text for embedding
    # sample_text = "Banking operational issue: unauthorized transaction."
    # embeddings = get_embedding(sample_text)

    # # Example prompt for Gemini
    # gemini_prompt = "Explain the primary function of a banking operational issue management system in a concise paragraph."
    # gemini_response = get_gemini_response(gemini_prompt)

    # print("\n--- Individual Model Checks Complete ---")
    # print("Before proceeding with the agentic AI framework, ensure:")
    # print("1. Cloudflare embeddings are generated correctly and capture semantic meaning.")
    # print("2. Google Gemini provides accurate, relevant, and coherent responses for your banking-specific prompts.")
    # print("You will use these models as building blocks for your RAG pipeline and integrate them with Weaviate.")
    
    # --- Example for Agent 1: recommend_solutions---
    new_issue = "Our customers are complaining that the mobile app freezes when they try to move money. Bill payments are getting stuck in a pending state and some people are getting double charged. We need a solution fast."
    agent_1_recommend_solutions(new_issue, banking_issues_data)

    print('***************************************************************************************')

    # --- Run Agent 2 Demo ---
    # NOTE: ISSUE-0002 has massive contradictions. The description is about a firewall,
    # the remediation is about data corruption, and the log is about wire transfers.
    # This is a perfect test case for the agent.
    issue_to_check_for_contradictions = banking_issues_data[1]
    agent_2_detect_contradictions(issue_to_check_for_contradictions)
    print('***************************************************************************************')
    # --- Run Agent 3 Demo ---
    # NOTE: As of today (Aug 4, 2025), ISSUE-0003 (due Aug 16, 2025) should be flagged.
    agent_3_monitor_due_dates(banking_issues_data)
    print('***************************************************************************************')
    # --- Run Stretch Goal: Agent 3 & 4 ---
    # poc.agent_3_monitor_due_dates()

    # --- Example for Agent 4: Structured Issue Summarizer ---
    issue_to_summarize = banking_issues_data[0]
    agent_4b_summarize_issue_single_call(issue_to_summarize)
    print('***************************************************************************************')

