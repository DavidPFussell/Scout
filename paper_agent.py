import arxiv
import datetime
import os
import requests
import time
from openai import OpenAI

# --- Configuration ---
# Keywords for ArXiv
ARXIV_KEYWORDS = '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI)'
# Keywords for GitHub Search
GITHUB_QUERY = 'topic:machine-learning OR topic:llm OR topic:ai'
LLM_MODEL = "gpt-4.1-mini"

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Source 1: ArXiv ---
def get_arxiv_papers():
    print("Fetching ArXiv...")
    client = arxiv.Client(page_size=15, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(query=ARXIV_KEYWORDS, max_results=15, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    today = datetime.datetime.now(datetime.timezone.utc).date()
    papers = []
    try:
        for result in client.results(search):
            if result.published.date() >= (today - datetime.timedelta(days=1)):
                papers.append({
                    "source": "ArXiv",
                    "title": result.title,
                    "desc": result.summary[:300] + "...", # Snippet for LLM
                    "url": result.entry_id
                })
    except Exception as e:
        print(f"ArXiv Error: {e}")
    return papers

# --- Source 2: Hugging Face Daily Papers ---
def get_hf_papers():
    print("Fetching Hugging Face...")
    try:
        response = requests.get("https://huggingface.co/api/papers", timeout=10)
        data = response.json()
        hf_papers = []
        # Take top 10 trending from HF
        for entry in data[:10]:
            hf_papers.append({
                "source": "Hugging Face",
                "title": entry['paper']['title'],
                "desc": "Trending paper on HF community.",
                "url": f"https://huggingface.co/papers/{entry['paper']['id']}"
            })
        return hf_papers
    except Exception as e:
        print(f"HF Error: {e}")
        return []

# --- Source 3: GitHub Trending (AI Search) ---
def get_github_trending():
    print("Fetching GitHub Trending...")
    # Finds repos updated in the last 24h with high stars
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://api.github.com/search/repositories?q={GITHUB_QUERY}+pushed:>{yesterday}&sort=stars&order=desc"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        repos = []
        for item in data.get('items', [])[:8]:
            repos.append({
                "source": "GitHub",
                "title": item['full_name'],
                "desc": item['description'] or "No description provided.",
                "url": item['html_url']
            })
        return repos
    except Exception as e:
        print(f"GitHub Error: {e}")
        return []

# --- THE BRAIN: Summarize and Rank ---
def summarize_and_rank(all_items):
    if not all_items:
        return None

    print(f"Brain is analyzing {len(all_items)} items...")
    
    # Format the list for the LLM
    input_text = ""
    for i, item in enumerate(all_items):
        input_text += f"ID: {i} | Source: {item['source']} | Title: {item['title']} | Desc: {item['desc']}\n\n"

    prompt = f"""
    You are an elite AI Research Scout. Below is a list of new papers and repos from ArXiv, Hugging Face, and GitHub.
    
    TASKS:
    1. Select the TOP 5 most significant items for a professional AI engineer.
    2. For each selected item, write a 2-sentence punchy summary.
    3. Categorize it (e.g., 'New Architecture', 'Tooling', 'Fine-tuning').

    LIST:
    {input_text}

    OUTPUT FORMAT:
    Return only a JSON-like list of the 5 IDs you chose, with your summary and category.
    Example: [{{"id": 0, "summary": "...", "cat": "..."}}, ...]
    """

    try:
        response = client_ai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        # Parse the LLM's selection
        import json
        raw_output = json.loads(response.choices[0].message.content)
        # Note: Depending on LLM response, might need adjustment
        selections = raw_output.get('selections', raw_output.get('top_items', list(raw_output.values())[0]))

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "🚀 Research Scout: Daily Top 5"}},
            {"type": "divider"}
        ]

        for sel in selections:
            item = all_items[int(sel['id'])]
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{item['source']}* | `{sel['cat']}`\n*<{item['url']}|{item['title']}>*\n{sel['summary']}"
                }
            })
        
        return blocks
    except Exception as e:
        print(f"Summarize/Rank Error: {e}")
        return None

def send_to_slack(blocks):
    webhook_url = os.getenv("SLACK_WEBHOOK")
    if not webhook_url: return
    requests.post(webhook_url, json={"blocks": blocks})

if __name__ == "__main__":
    # Collect
    findings = []
    findings.extend(get_arxiv_papers())
    findings.extend(get_hf_papers())
    findings.extend(get_github_trending())

    # Process
    if findings:
        final_report = summarize_and_rank(findings)
        if final_report:
            send_to_slack(final_report)
            print("Done!")
    else:
        print("Nothing found today.")
