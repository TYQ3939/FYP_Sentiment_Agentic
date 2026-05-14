# Windows asyncio fix at ABSOLUTE module level - BEFORE anything else
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# NOW import everything else
import os
import json
import time
from datetime import datetime, timedelta
from playwright.async_api import async_playwright

# ========== LLM-BASED SUBREDDIT MAPPER ==========

def infer_subreddits_from_topic(topic: str, llm) -> list:
    """Uses the LLM to infer relevant subreddits based on the user's input topic."""
    
    prompt = f"""
    Given the topic: "{topic}"
    
    Generate a list of the TOP 3 most relevant Reddit subreddits where people discuss this topic.
    
    Return ONLY a JSON array of subreddit names (without the 'r/' prefix), like this:
    ["subreddit1", "subreddit2", "subreddit3"]
    
    Make sure the subreddits exist and are active. Do NOT include inactive or made-up subreddits.
    Focus on popular, relevant communities.
    """
    
    try:
        print(f"LLM analyzing topic: '{topic}' to find relevant subreddits...")
        response = llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            subreddits = json.loads(json_str)
            print(f"✅ LLM suggested subreddits: {subreddits}")
            return subreddits
        else:
            print("⚠️ Could not parse LLM response, using fallback subreddits")
            return ["iphone", "apple", "technology"]
    
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ LLM error: {error_msg[:150]}...")
        print("Using fallback subreddits: ['iphone', 'apple', 'technology']")
        return ["iphone", "apple", "technology"]


# ========== DATA PARSING AND FILTERING ==========

def parse_and_filter_data(subreddit: str, topic: str, posts_file: str, comments_file: str) -> dict:
    """Parses posts and comments JSONL files, filters by topic keyword, and matches comments to posts."""
    
    filtered_posts = []
    filtered_comments = []
    
    try:
        print(f"Parsing posts file: {posts_file}")
        if os.path.exists(posts_file):
            with open(posts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            post = json.loads(line)
                            title = post.get("title", "").lower()
                            topic_lower = topic.lower()
                            
                            if topic_lower in title or _is_topic_related(title, topic_lower):
                                filtered_posts.append({
                                    "link_id": post.get("id", ""),
                                    "title": post.get("title", ""),
                                    "text": post.get("selftext", ""),
                                    "score": post.get("score", 0),
                                    "created_at": post.get("created_utc", ""),
                                    "author": post.get("author", ""),
                                    "url": post.get("url", "")
                                })
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"⚠️ Posts file not found: {posts_file}")
        
        print(f"✅ Filtered posts by topic: {len(filtered_posts)} posts found")
        
        link_ids = set(post["link_id"] for post in filtered_posts)
        print(f"Extracted link_ids: {len(link_ids)} unique post IDs")
        
        print(f"Parsing comments file: {comments_file}")
        if os.path.exists(comments_file):
            with open(comments_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            comment = json.loads(line)
                            comment_link_id = comment.get("link_id", "").replace("t3_", "")
                            
                            if comment_link_id in link_ids:
                                filtered_comments.append({
                                    "link_id": comment_link_id,
                                    "text": comment.get("body", ""),
                                    "score": comment.get("score", 0),
                                    "created_at": comment.get("created_utc", ""),
                                    "author": comment.get("author", "")
                                })
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"⚠️ Comments file not found: {comments_file}")
        
        print(f"✅ Filtered comments by link_id: {len(filtered_comments)} comments found")
        
        return {
            "subreddit": subreddit,
            "topic": topic,
            "posts": filtered_posts,
            "comments": filtered_comments,
            "posts_count": len(filtered_posts),
            "comments_count": len(filtered_comments),
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error parsing and filtering data: {str(e)}")
        return {
            "subreddit": subreddit,
            "topic": topic,
            "posts": [],
            "comments": [],
            "posts_count": 0,
            "comments_count": 0,
            "status": "error",
            "error_message": str(e)
        }


def _is_topic_related(title: str, topic: str) -> bool:
    """Helper function to check if title is related to topic."""
    keywords = topic.split()
    matching_keywords = sum(1 for keyword in keywords if keyword in title)
    return matching_keywords >= len(keywords) * 0.5 if keywords else False


# ========== ARCTIC SHIFT (REDDIT) SCRAPER ==========

async def automate_arctic_shift_download(subreddit: str, start_date: str, end_date: str, data_dir: str = "./data/raw_data") -> tuple:
    """
    Automates the Arctic Shift website to download Reddit posts and comments data.
    """
    
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Arctic Shift Download Process")
    print(f"{'='*70}")
    print(f"Subreddit: {subreddit}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    browser = None
    context = None
    page = None
    
    try:
        print("📱 Launching browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            print("✅ Browser and page created\n")
            
            # Step 1: Navigate to Arctic Shift
            print(f"📡 Step 1: Navigating to Arctic Shift download tool...")
            try:
                await page.goto("https://arctic-shift.photon-reddit.com/download-tool", wait_until="networkidle", timeout=60000)
                print("✅ Page loaded successfully\n")
            except Exception as e:
                print(f"❌ Failed to load page: {str(e)}")
                raise
            
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(2000)
            
            # Step 2: Ensure "r/" toggle is selected
            print(f"🔘 Step 2: Ensuring subreddit toggle (r/) is selected...")
            try:
                toggle_buttons = await page.query_selector_all(".option-selector button.option")
                if len(toggle_buttons) >= 1:
                    await toggle_buttons[0].click()
                    print("  ✅ Subreddit toggle (r/) selected\n")
            except Exception as e:
                print(f"  ⚠️ Could not ensure toggle selection: {str(e)}, proceeding\n")
            
            # Step 3: Fill subreddit name
            print(f"📝 Step 3: Filling form fields...")
            try:
                subreddit_input = await page.query_selector("input[placeholder='Subreddit name']")
                if subreddit_input:
                    await subreddit_input.fill("")
                    await subreddit_input.fill(subreddit)
                    print(f"  ✅ Subreddit field filled: {subreddit}")
                else:
                    raise Exception("Could not find subreddit input field")
            except Exception as e:
                print(f"  ❌ Failed to fill subreddit: {str(e)}")
                raise
            
            # Step 4: Fill start date
            try:
                text_inputs = await page.query_selector_all("input[type='text'].text-input")
                if len(text_inputs) >= 2:
                    await text_inputs[1].fill(start_date)
                    print(f"  ✅ Start date filled: {start_date}")
                else:
                    raise Exception("Could not find start date input field")
            except Exception as e:
                print(f"  ❌ Failed to fill start date: {str(e)}")
                raise
            
            # Step 5: Fill end date
            try:
                text_inputs = await page.query_selector_all("input[type='text'].text-input")
                if len(text_inputs) >= 3:
                    await text_inputs[2].fill(end_date)
                    print(f"  ✅ End date filled: {end_date}\n")
                else:
                    raise Exception("Could not find end date input field")
            except Exception as e:
                print(f"  ❌ Failed to fill end date: {str(e)}")
                raise
            
            # Step 6: Check "Download posts" checkbox
            print(f"☑️  Step 4: Checking download options...")
            try:
                posts_checkbox = await page.query_selector("#download-posts")
                if posts_checkbox:
                    is_checked = await posts_checkbox.is_checked()
                    if not is_checked:
                        await posts_checkbox.check()
                        print("  ✅ 'Download posts' checkbox checked")
                    else:
                        print("  ✅ 'Download posts' checkbox already checked")
                else:
                    raise Exception("Could not find 'Download posts' checkbox")
            except Exception as e:
                print(f"  ❌ Failed to check posts checkbox: {str(e)}")
                raise
            
            # Step 7: Check "Download comments" checkbox
            try:
                comments_checkbox = await page.query_selector("#download-comments")
                if comments_checkbox:
                    is_checked = await comments_checkbox.is_checked()
                    if not is_checked:
                        await comments_checkbox.check()
                        print("  ✅ 'Download comments' checkbox checked\n")
                    else:
                        print("  ✅ 'Download comments' checkbox already checked\n")
                else:
                    raise Exception("Could not find 'Download comments' checkbox")
            except Exception as e:
                print(f"  ❌ Failed to check comments checkbox: {str(e)}")
                raise
            
            # Step 8: Click "Start" button
            print(f"🎯 Step 5: Clicking 'Start' button...")
            try:
                start_button = await page.query_selector("button.main-action.primary")
                if start_button:
                    button_text = await start_button.text_content()
                    print(f"  Found button with text: '{button_text.strip()}'")
                    await start_button.click()
                    print("  ✅ 'Start' button clicked")
                else:
                    raise Exception("Could not find 'Start' button")
                
                await page.wait_for_timeout(2000)
            except Exception as e:
                print(f"  ❌ Failed to click Start button: {str(e)}")
                raise
            
            # Step 9: Handle file save dialogs
            print(f"\n💾 Step 6: Handling file save dialogs...")
            
            posts_file = None
            comments_file = None
            
            try:
                print("  ⏳ Waiting for posts file download...")
                posts_download = await page.wait_for_download(timeout=120000)
                
                posts_filename = f"{subreddit}_posts_{start_date}_to_{end_date}.jsonl"
                posts_file = os.path.join(data_dir, posts_filename)
                
                await posts_download.save_as(posts_file)
                print(f"  ✅ Posts file saved: {posts_file}")
                
                await page.wait_for_timeout(1000)
                
                print("  ⏳ Waiting for comments file download...")
                comments_download = await page.wait_for_download(timeout=120000)
                
                comments_filename = f"{subreddit}_comments_{start_date}_to_{end_date}.jsonl"
                comments_file = os.path.join(data_dir, comments_filename)
                
                await comments_download.save_as(comments_file)
                print(f"  ✅ Comments file saved: {comments_file}\n")
            
            except asyncio.TimeoutError:
                print("  ❌ File download timeout (exceeded 2 minutes)")
                raise TimeoutError("File download timed out")
            
            except Exception as e:
                print(f"  ❌ Failed to save files: {str(e)}")
                raise
            
            # Verify files exist
            print(f"🔍 Step 7: Verifying downloaded files...")
            
            if not os.path.exists(posts_file):
                raise Exception(f"Posts file was not saved: {posts_file}")
            
            if not os.path.exists(comments_file):
                raise Exception(f"Comments file was not saved: {comments_file}")
            
            posts_size = os.path.getsize(posts_file)
            comments_size = os.path.getsize(comments_file)
            
            print(f"  ✅ Posts file verified: {posts_size:,} bytes")
            print(f"  ✅ Comments file verified: {comments_size:,} bytes\n")
            
            print(f"{'='*70}")
            print(f"✅ Download completed successfully!")
            print(f"{'='*70}\n")
            
            return posts_file, comments_file
    
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ Download failed: {str(e)}")
        print(f"{'='*70}\n")
        raise
    
    finally:
        if context:
            try:
                await context.close()
            except:
                pass
        if browser:
            try:
                await browser.close()
            except:
                pass


# ========== RETRY LOGIC ==========

def scrape_with_retry(subreddit: str, topic: str, min_comments: int = 500, max_iterations: int = 2) -> dict:
    """
    Scrapes data with retry logic for expanded date ranges.
    
    ⚠️ CRITICAL: Uses asyncio.run() to create fresh event loop with correct policy
    """
    
    all_posts = []
    all_comments = []
    iteration = 0
    base_wait_time = 2
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    while iteration < max_iterations:
        iteration += 1
        wait_time = base_wait_time * (2 ** (iteration - 1))
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{max_iterations}")
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        try:
            if iteration > 1:
                print(f"⏳ Waiting {wait_time}s before next request...")
                time.sleep(wait_time)
            
            # ⚠️ CRITICAL: Use asyncio.run() to create fresh event loop with correct policy
            posts_file, comments_file = asyncio.run(
                automate_arctic_shift_download(
                    subreddit, 
                    start_date.strftime("%Y-%m-%d"), 
                    end_date.strftime("%Y-%m-%d")
                )
            )
            
            # Parse and filter the downloaded data
            if posts_file and comments_file:
                filtered_data = parse_and_filter_data(subreddit, topic, posts_file, comments_file)
                all_posts.extend(filtered_data["posts"])
                all_comments.extend(filtered_data["comments"])
                
                print(f"\nIteration {iteration} summary:")
                print(f"  Posts: {filtered_data['posts_count']}")
                print(f"  Comments: {filtered_data['comments_count']}")
                print(f"  Total comments: {len(all_comments)}")
                
                if len(all_comments) >= min_comments:
                    print(f"\n✅ SUCCESS: Reached {len(all_comments)} comments (target: {min_comments})")
                    break
        
        except Exception as e:
            print(f"⚠️ Iteration {iteration} error: {str(e)[:80]}...")
            if iteration >= max_iterations:
                print(f"❌ Max iterations reached ({max_iterations})")
                break
        
        # Prepare next iteration with expanded date range
        if iteration < max_iterations and len(all_comments) < min_comments:
            start_date = start_date - timedelta(days=2)
            print(f"Expanding date range for next iteration...")
    
    # Remove duplicate comments
    unique_comments = {c["text"]: c for c in all_comments}
    
    return {
        "subreddit": subreddit,
        "topic": topic,
        "posts": all_posts,
        "comments": list(unique_comments.values()),
        "posts_count": len(all_posts),
        "comments_count": len(unique_comments),
        "iterations_used": iteration,
        "target_reached": len(unique_comments) >= min_comments,
        "scraped_at": datetime.now().isoformat()
    }


def scrape_arctic_shift_sync(subreddit: str, start_date: str, end_date: str) -> dict:
    """Synchronous wrapper for scraping Arctic Shift."""
    
    # ⚠️ CRITICAL: Use asyncio.run() instead of manual loop management
    posts_file, comments_file = asyncio.run(
        automate_arctic_shift_download(subreddit, start_date, end_date)
    )
    
    return {
        "source": "arctic_shift_reddit",
        "subreddit": subreddit,
        "date_range": {"start": start_date, "end": end_date},
        "posts_file": posts_file,
        "comments_file": comments_file,
        "scraped_at": datetime.now().isoformat(),
        "status": "success"
    }