import arxiv
import datetime
import os
import requests
import json
import feedparser
import re
from openai import OpenAI

# --- Configuration ---
NEG = " -vision -image -video -diffusion -cv"
ARXIV_QUERY = f'(cat:cs.CL OR cat:cs.LG OR cat:cs.AI){NEG}'
# Use keywords instead of topics for better reach
GITHUB_QUERY = f'llm OR "large language model" OR agents OR rag OR nlp{NEG}'
NEWS_RSS = f"https://news.google.com/rss/search?q=AI+LLM+OR+Agents+OR+NLP+-vision+-image+-video+-diffusion&hl=en-US&gl=US&ceid=US:en"
LLM_MODEL = "gpt-4.1-mini"

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_code_link(text):
    match = re.search(r'github\.com/[\w\-/]+', text)
    return f"https://{match.group(0)}" if match else None

# --- Source Fetchers ---
def get_arxiv_papers():
    print("Fetching ArXiv...")
    client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(query=ARXIV_QUERY, max_results=20, sort_by=arxiv.SortCriterion.SubmittedDate)
    return [{"title": r.title, "desc": r.summary[:500], "url": r.entry_id, "code_url": detect_code_link(r.summary)} for r in client.results(search)]

def get_hf_papers():
    print("Fetching Hugging Face...")
    try:
        response = requests.get("https://huggingface.co/api/papers", timeout=10)
        ignore = ['vision', 'image', 'video', 'diffusion', 'depth', 'segmentation']
        return [{"title": x['title'], "desc": "Trending on HF.", "url": f"https://huggingface.co/papers/{x['id']}", "code_url": None} 
                for x in response.json() if not any(word in x['title'].lower() for word in ignore)][:15]
    except Exception as e: print(f"HF Error: {e}"); return []

def get_github_trending():
    print("Fetching GitHub...")
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"} if token else {}
    
    # Check repos updated in the last 7 days (broader than 2 days to ensure results)
    since_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"https://api.github.com/search/repositories?q={GITHUB_QUERY}+pushed:>{since_date}&sort=stars&order=desc"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        print(f"GitHub Status Code: {response.status_code}")
        
        data = response.json()
        items = data.get('items', [])
        print(f"GitHub Found {len(items)} total items.")
        
        results = []
        for x in items[:15]:
            results.append({
                "title": x['full_name'],
                "desc": x.get('description', '') or "AI Project",
                "url": x['html_url'],
                "code_url": x['html_url']
            })
        return results
    except Exception as e: 
        print(f"GitHub Error: {e}")
        return []

def get_ai_news():
    print("Fetching News...")
    try:
        feed = feedparser.parse(NEWS_RSS)
        return [{"title": x.title, "desc": "NLP/LLM News.", "url": x.link, "code_url": None} for x in feed.entries[:15]]
    except Exception as e: print(f"News Error: {e}"); return []

# --- THE BRAIN ---
def process_source(source_name, items):
    if not items:
        print(f"Skipping {source_name} - No items found.")
        return []
        
    print(f"Brain is filtering {len(items)} items from {source_name}...")
    input_data = [{"id": i, "title": item['title'], "desc": item['desc'][:300]} for i, item in enumerate(items)]

    prompt = f"""
    Focus EXCLUSIVELY on NLP, LLMs, Agents, and RAG. 
    Pick the TOP 5 items.
    
    Data: {json.dumps(input_data)}
    
    Return JSON only: {{"selections": [{{"id": 0, "summary": "...", "hype": 1-10, "cat": "tag"}}, ...]}}
    """

    try:
        response = client_ai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "Professional NLP researcher. Output JSON."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        raw_data = json.loads(response.choices[0].message.content)
        selections = raw_data.get('selections', [])
        
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"*--- TOP 5 {source_name.upper()} ---*"}}]
        for sel in selections:
            item = items[int(sel['id'])]
            score = int(sel['hype'])
            hype_emoji = "🚀" if score >= 8 else "📈"
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
        print(f"Brain Error for {source_name}: {e}")
        return []

def send_to_slack(all_blocks):
    url = os.getenv("SLACK_WEBHOOK")
    if url: 
        for i in range(0, len(all_blocks), 30):
            requests.post(url, json={"blocks": all_blocks[i:i+30]})

if __name__ == "__main__":
    final_blocks = [{"type": "header", "text": {"type": "plain_text", "text": "🧠 NLP & LLM Scout: Daily Report"}}]
    
    sources = {
        "ArXiv Papers": get_arxiv_papers(),
        "Hugging Face": get_hf_papers(),
        "GitHub Repos": get_github_trending(),
        "Industry News": get_ai_news()
    }
    
    for name, data in sources.items():
        res = process_source(name, data)
        if res:
            final_blocks.extend(res)
            final_blocks.append({"type": "divider"})
            
    if len(final_blocks) > 1:
        send_to_slack(final_blocks)
        print("Scout mission success.")
