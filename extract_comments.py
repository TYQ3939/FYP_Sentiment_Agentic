import json
import csv
import os
from datetime import datetime

# Configuration
POSTS_FILE = 'data/raw_data/r_gaminglaptops1_posts.jsonl'
COMMENTS_FILE = 'data/raw_data/r_gaminglaptops1_comments.jsonl'
OUTPUT_CSV = 'data/filtered_data/filtered_gaminglaptops_comments.csv'
KEYWORD = ["acer nitro 5", "Acer Nitro 5", "acer nitro5", "acernitro5"]

def extract_to_csv():
    # Mapping of post 'name' (link_id) -> {'title': text, 'selftext': text}
    post_data_map = {}
    keyword_lower = [keyword.lower() for keyword in KEYWORD]

    print(f"Step 1: Mapping posts with keyword '{KEYWORD}'...")
    try:
        with open(POSTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                post = json.loads(line)
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                search_text = f"{title} {selftext}".lower()
                
                if any(keyword in search_text for keyword in keyword_lower):
                    fullname = post.get('name')
                    if fullname:
                        # Store both title and selftext in our dictionary map
                        post_data_map[fullname] = {
                            'title': title if title else '[No Title]',
                            'selftext': selftext if selftext else '[No Body Text]'
                        }
    except FileNotFoundError:
        print(f"Error: Could not find the source file at {POSTS_FILE}")
        return

    if not post_data_map:
        print("No matching posts found.")
        return

    print(f"Found {len(post_data_map)} posts. Matching comments...")

    # Automatically create 'data/processed_data/' folder structure if missing
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ADJUSTMENT: Added 'post_selftext' into the columns we want in the CSV
    fields = ['date_utc', 'author', 'post_title', 'post_selftext', 'comment_body', 'score', 'link_id', 'permalink']

    comment_count = 0
    
    try:
        with open(COMMENTS_FILE, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f_out:
            
            writer = csv.DictWriter(f_out, fieldnames=fields)
            writer.writeheader()
            
            for line in f_in:
                comment = json.loads(line)
                link_id = comment.get('link_id')
                
                if link_id in post_data_map:
                    utc_ts = comment.get('created_utc')
                    readable_date = datetime.fromtimestamp(utc_ts).strftime('%Y-%m-%d %H:%M:%S') if utc_ts else "N/A"
                    
                    # ADJUSTMENT: Build the row extracting from post_data_map structure
                    row = {
                        'date_utc': readable_date,
                        'author': comment.get('author'),
                        'post_title': post_data_map[link_id]['title'],
                        'post_selftext': post_data_map[link_id]['selftext'], # New Column Data
                        'comment_body': comment.get('body'),
                        'score': comment.get('score'),
                        'link_id': link_id,
                        'permalink': f"https://reddit.com{comment.get('permalink', '')}"
                    }
                    
                    writer.writerow(row)
                    comment_count += 1
                    
        print(f"Success! Saved {comment_count} comments to {OUTPUT_CSV}.")
        
    except PermissionError:
        print(f"\n❌ Error: Cannot write to {OUTPUT_CSV}.")
        print("Please close the CSV file in Microsoft Excel or any other program, then run this script again.")

if __name__ == "__main__":
    extract_to_csv()
