import arxiv
import datetime
import os
import requests
import time
from openai import OpenAI

# Configuration
KEYWORDS = '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.MA)'
LLM_MODEL = "gpt-4.1-mini"

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_latest_papers():
    # 1. Use the new Client object which handles rate limits better
    client = arxiv.Client(
        page_size = 20,
        delay_seconds = 3.0, # Wait 3 seconds between pages to avoid 429
        num_retries = 5      # Retry 5 times if 429 occurs
    )

    search = arxiv.Search(
        query=KEYWORDS,
        max_results=20, # Reduced from 100 to avoid heavy API calls
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    today = datetime.datetime.now(datetime.timezone.utc).date()
    papers = []
    
    # 2. Use client.results() instead of search.results() to fix deprecation
    try:
        results = client.results(search)
        for result in results:
            # Look for papers from today or yesterday
            if result.published.date() >= (today - datetime.timedelta(days=1)):
                papers.append({
                    "title": result.title,
                    "summary": result.summary,
                    "url": result.entry_id,
                    "author": result.authors[0].name if result.authors else "Unknown"
                })
    except Exception as e:
        print(f"Error fetching from ArXiv: {e}")
        return []

    return papers

def summarize_papers(papers):
    if not papers:
        return None

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🤖 Daily Research Scout Report"}
        },
        {"type": "divider"}
    ]
    
    # Only process top 5 papers to save LLM tokens and keep Slack clean
    for paper in papers[:5]:
        prompt = f"Summarize this AI research paper abstract in two sentences. Focus on why it matters. \n\nAbstract: {paper['summary']}"
        
        try:
            response = client_ai.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.choices[0].message.content
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<{paper['url']}|{paper['title']}>*\n_By: {paper['author']}_\n{summary}"
                }
            })
        except Exception as e:
            print(f"Error summarizing: {e}")
    
    return blocks

def send_to_slack(blocks):
    webhook_url = os.getenv("SLACK_WEBHOOK")
    if webhook_url and blocks:
        payload = {"blocks": blocks}
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 200:
            print(f"Error sending to Slack: {response.text}")

if __name__ == "__main__":
    print("Starting scout...")
    new_papers = get_latest_papers()
    print(f"Found {len(new_papers)} recent papers.")
    
    report_blocks = summarize_papers(new_papers)
    if report_blocks:
        send_to_slack(report_blocks)
        print("Report sent to Slack!")
    else:
        print("No new papers found or error occurred.")
