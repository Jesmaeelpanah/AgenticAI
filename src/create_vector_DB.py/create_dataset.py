import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import os
import requests
from dotenv import load_dotenv
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()

GEMINI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_URL = os.getenv("OPENAI_BASE_URL") + "chat/completions"

def ask_gemini(prompt_text, model_name="gemini-2.5-flash"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
# Step 1: Load JSON
with open("/home/coder/AgenticAI/agentic_framework/create_vector_DB.py/banking_historical_issues.json", "r") as f:
    issues = json.load(f)

# rows = []

# for issue in tqdm(issues, desc="Summarizing issues"):
#     issue_id = issue.get("issue_id", "unknown")
#     issue_description = issue.get("issue_description", "")
#     combined_info = "\n".join([f"{k}: {v}" for k, v in issue.items() if v])
#     if not issue_description.strip():
#         summary = f"Issue {issue_id} has no description provided."
#     else:
#         summary_prompt = f"""
#         Summarize the following banking issue into a single, dense paragraph. Focus on the core problem, its root cause, and the actions taken for the final resolution. This summary will be used to find similar documents, so it must be rich with specific keywords and concepts.

#         ISSUE DATA:
#         {combined_info}
#         """
#     try:
#             summary = ask_gemini(summary_prompt.strip())
#     except Exception as e:
#             summary = f"[Error summarizing issue {issue_id}]: {e}"    

#     # ✅ This is the only thing that will be embedded
#     rows.append({
#         "text": summary,  # used for embedding
#         "issue_id": issue.get("issue_id"),
#         "issue_type": issue.get("issue_type"),
#         "severity": issue.get("severity"),
#         "status": issue.get("status"),
#         "system": issue.get("system"),
#         "department": issue.get("department"),
#         "root_cause": issue.get("root_cause"),
#         "creation_date": issue.get("creation_date"),
#         "remediation_completion_date": issue.get("remediation_completion_date"),
#         "contradiction_flag": issue.get("contradiction_flag"),
#         "contradiction_type": issue.get("contradiction_type"),
#         "issue_description": issue.get("issue_description"),
#         "remediation_plan": issue.get("remediation_plan"),
#         "comments_log": issue.get("comments_log")
#     })
    
# # Convert to HuggingFace dataset
# df = pd.DataFrame(rows)
# df.to_csv('df.csv')
# dataset = Dataset.from_pandas(df)  


# Parallelized summary function
def summarize_issue(issue):
    issue_id = issue.get("issue_id", "unknown")
    issue_description = issue.get("issue_description", "")
    combined_info = "\n".join([f"{k}: {v}" for k, v in issue.items() if v])

    if not issue_description.strip():
        summary = f"Issue {issue_id} has no description provided."
    else:
        summary_prompt = f"""
        You are an expert analyst. Read the banking issue below and write a clear, concise paragraph that summarizes the problem, its root cause, the system affected, and the actions taken to resolve it.

        Avoid using bullet points, issue IDs, or formatting symbols like asterisks. Write in plain language suitable for use in a search engine or knowledge base but clear and detailed.

        BANKING ISSUE:

        {combined_info}
        """
        try:
            summary = ask_gemini(summary_prompt.strip())
        except Exception as e:
            summary = f"[Error summarizing issue {issue_id}]: {e}"

    return {
        "text": summary,
        "issue_id": issue.get("issue_id"),
        "issue_type": issue.get("issue_type"),
        "severity": issue.get("severity"),
        "status": issue.get("status"),
        "system": issue.get("system"),
        "department": issue.get("department"),
        "root_cause": issue.get("root_cause"),
        "creation_date": issue.get("creation_date"),
        "remediation_completion_date": issue.get("remediation_completion_date"),
        "contradiction_flag": issue.get("contradiction_flag"),
        "contradiction_type": issue.get("contradiction_type"),
        "issue_description": issue.get("issue_description"),
        "remediation_plan": issue.get("remediation_plan"),
        "comments_log": issue.get("comments_log"),
        "issue":issue
    }

# Run parallelized summarization
rows = []
with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
    futures = [executor.submit(summarize_issue, issue) for issue in issues]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing in parallel"):
        rows.append(future.result())

# Save or push
df = pd.DataFrame(rows)
dataset = Dataset.from_pandas(df)
df.to_csv('df.csv')
dataset.push_to_hub("Manisha007/banking-issue-kb", private=False)

