import arxiv
import datetime
import os
import requests
import json
from openai import OpenAI

# --- Configuration ---
ARXIV_KEYWORDS = '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.MA)'
GITHUB_QUERY = 'topic:machine-learning OR topic:llm OR topic:ai'
LLM_MODEL = "gpt-4.1-mini"

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_arxiv_papers():
    print("Fetching ArXiv...")
    client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(query=ARXIV_KEYWORDS, max_results=20, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    today = datetime.datetime.now(datetime.timezone.utc).date()
    papers = []
    try:
        for result in client.results(search):
            if result.published.date() >= (today - datetime.timedelta(days=2)): # Look back 2 days to be safe
                papers.append({
                    "source": "ArXiv",
                    "title": result.title,
                    "desc": result.summary[:400], 
                    "url": result.entry_id
                })
    except Exception as e:
        print(f"ArXiv Error: {e}")
    return papers

def get_hf_papers():
    print("Fetching Hugging Face...")
    try:
        # Correcting the API mapping here
        response = requests.get("https://huggingface.co/api/papers", timeout=10)
        response.raise_for_status()
        data = response.json()
        hf_papers = []
        for entry in data[:10]:
            hf_papers.append({
                "source": "Hugging Face",
                "title": entry.get('title', 'Untitled'),
                "desc": "Trending paper on Hugging Face community.",
                "url": f"https://huggingface.co/papers/{entry.get('id')}"
            })
        return hf_papers
    except Exception as e:
        print(f"HF Error: {e}")
        return []

def get_github_trending():
    print("Fetching GitHub Trending...")
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
    url = f"https://api.github.com/search/repositories?q={GITHUB_QUERY}+pushed:>{yesterday}&sort=stars&order=desc"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        repos = []
        for item in data.get('items', [])[:10]:
            repos.append({
                "source": "GitHub",
                "title": item['full_name'],
                "desc": item.get('description') or "AI Repository",
                "url": item['html_url']
            })
        return repos
    except Exception as e:
        print(f"GitHub Error: {e}")
        return []

def summarize_and_rank(all_items):
    if not all_items:
        return None

    print(f"Brain is analyzing {len(all_items)} items...")
    
    input_list = []
    for i, item in enumerate(all_items):
        input_list.append({
            "id": i,
            "source": item['source'],
            "title": item['title'],
            "description": item['desc'][:300]
        })

    prompt = f"""
    You are a Research Scout. I have {len(all_items)} items from ArXiv, Hugging Face, and GitHub.
    1. Select the top 5 most interesting items.
    2. Provide a 1-2 sentence summary for each.
    3. Categorize each (e.g., 'Model', 'Dataset', 'Tool', 'Research').
    
    Data: {json.dumps(input_list)}
    
    Return a JSON object with a key 'selections' containing a list of objects with keys: id, summary, category.
    """

    try:
        response = client_ai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "You only output valid JSON."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        raw_data = json.loads(response.choices[0].message.content)
        selections = raw_data.get('selections', [])

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "🚀 Daily AI Scout Report"}},
            {"type": "divider"}
        ]

        for sel in selections:
            idx = int(sel['id'])
            original = all_items[idx]
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{original['source']}* | `{sel['category']}`\n*<{original['url']}|{original['title']}>*\n{sel['summary']}"
                }
            })
        
        return blocks
    except Exception as e:
        print(f"Brain Error: {e}")
        return None

def send_to_slack(blocks):
    webhook_url = os.getenv("SLACK_WEBHOOK")
    if not webhook_url:
        print("No Slack Webhook found.")
        return
    
    response = requests.post(webhook_url, json={"blocks": blocks})
    if response.status_code == 200:
        print("✅ Sent to Slack!")
    else:
        print(f"❌ Slack Error: {response.status_code} {response.text}")

if __name__ == "__main__":
    findings = []
    findings.extend(get_arxiv_papers())
    findings.extend(get_hf_papers())
    findings.extend(get_github_trending())

    if findings:
        report_blocks = summarize_and_rank(findings)
        if report_blocks:
            send_to_slack(report_blocks)
    else:
        print("No news found today.")
