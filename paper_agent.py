import arxiv
import datetime
import os
import requests
import json
import feedparser
import re
from openai import OpenAI

# --- Configuration ---
ARXIV_KEYWORDS = '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI)'
GITHUB_QUERY = 'topic:machine-learning OR topic:llm OR topic:ai'
NEWS_RSS = "https://news.google.com/rss/search?q=Artificial+Intelligence+when:24h&hl=en-US&gl=US&ceid=US:en"
LLM_MODEL = "gpt-4.1-mini"

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper: Code Detector ---
def detect_code_link(text):
    match = re.search(r'github\.com/[\w\-/]+', text)
    return f"https://{match.group(0)}" if match else None

# --- Source Fetchers ---
def get_arxiv_papers():
    print("Fetching ArXiv...")
    client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(query=ARXIV_KEYWORDS, max_results=20, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "desc": result.summary[:500],
                "url": result.entry_id,
                "code_url": detect_code_link(result.summary)
            })
    except Exception as e: print(f"ArXiv Error: {e}")
    return papers

def get_hf_papers():
    print("Fetching Hugging Face...")
    try:
        response = requests.get("https://huggingface.co/api/papers", timeout=10)
        return [{"title": x.get('title'), "desc": "Trending on HF.", "url": f"https://huggingface.co/papers/{x.get('id')}", "code_url": None} for x in response.json()[:15]]
    except Exception as e: print(f"HF Error: {e}"); return []

def get_github_trending():
    print("Fetching GitHub...")
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
    url = f"https://api.github.com/search/repositories?q={GITHUB_QUERY}+pushed:>{yesterday}&sort=stars&order=desc"
    try:
        response = requests.get(url, timeout=10)
        return [{"title": x['full_name'], "desc": x.get('description', ''), "url": x['html_url'], "code_url": x['html_url']} for x in response.json().get('items', [])[:15]]
    except Exception as e: print(f"GitHub Error: {e}"); return []

def get_ai_news():
    print("Fetching AI News...")
    try:
        feed = feedparser.parse(NEWS_RSS)
        return [{"title": x.title, "desc": "Latest news update.", "url": x.link, "code_url": None} for x in feed.entries[:15]]
    except Exception as e: print(f"News Error: {e}"); return []

# --- THE BRAIN: Process each source individually ---
def process_source(source_name, items):
    if not items: return []

    print(f"Brain is ranking top 5 for {source_name}...")
    
    input_data = [{"id": i, "title": item['title'], "desc": item['desc'][:300]} for i, item in enumerate(items)]

    prompt = f"""
    Pick the top 5 most important items from this {source_name} list.
    Return JSON only: {{"selections": [{{"id": 0, "summary": "1-sentence", "hype": 1-10, "cat": "tag"}}, ...]}}
    """

    try:
        response = client_ai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "You are a senior AI researcher. JSON only."},
                      {"role": "user", "content": f"{prompt}\n\nData: {json.dumps(input_data)}"}],
            response_format={ "type": "json_object" }
        )
        
        raw_data = json.loads(response.choices[0].message.content)
        selections = raw_data.get('selections', [])

        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*--- TOP 5 {source_name.upper()} ---*"}}
        ]

        for sel in selections:
            item = items[int(sel['id'])]
            score = int(sel['hype'])
            hype_emoji = "🚀" if score >= 8 else "📈" if score >= 5 else "☕"
            code_text = f" | 💻 <{item['code_url']}|*Code*>" if item['code_url'] else ""
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"`{sel['cat']}` {hype_emoji} *Hype: {score}/10*{code_text}\n*<{item['url']}|{item['title']}>*\n{sel['summary']}"
                }
            })
        return blocks
    except Exception as e: 
        print(f"Error processing {source_name}: {e}")
        return []

def send_to_slack(all_blocks):
    webhook_url = os.getenv("SLACK_WEBHOOK")
    if not webhook_url: return
    
    # Slack has a block limit per message (50), so we send in chunks if needed
    for i in range(0, len(all_blocks), 40):
        chunk = all_blocks[i:i+40]
        requests.post(webhook_url, json={"blocks": chunk})

if __name__ == "__main__":
    final_blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "🔥 Elite AI Scout: Daily 20"}}
    ]

    # Process each source
    sources = {
        "ArXiv Papers": get_arxiv_papers(),
        "Hugging Face": get_hf_papers(),
        "GitHub Repos": get_github_trending(),
        "Industry News": get_ai_news()
    }

    for name, data in sources.items():
        source_blocks = process_source(name, data)
        if source_blocks:
            final_blocks.extend(source_blocks)
            final_blocks.append({"type": "divider"})

    if len(final_blocks) > 1:
        send_to_slack(final_blocks)
        print("Report sent!")
    else:
        print("Nothing found.")
