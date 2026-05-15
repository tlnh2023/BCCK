"""Reddit data crawler module"""

import os
import praw
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class RedditCrawler:
    """Crawler for Reddit posts and comments"""
    
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
    
    def get_subreddit_posts(self, subreddit: str, limit: int = 100) -> List[praw.models.Submission]:
        """
        Fetch posts from a subreddit
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to fetch
            
        Returns:
            List of post objects
        """
        try:
            sub = self.reddit.subreddit(subreddit)
            return list(sub.hot(limit=limit))
        except Exception as e:
            print(f"Error fetching subreddit posts: {e}")
            return []
    
    def parse_post(self, post: praw.models.Submission) -> Dict[str, Any]:
        """
        Parse raw post data into standardized format
        
        Args:
            post: Raw post object from PRAW
            
        Returns:
            Parsed post dictionary
        """
        return {
            "source": "reddit",
            "id": post.id,
            "title": post.title,
            "body": post.selftext,
            "author": post.author.name if post.author else "deleted",
            "created_at": datetime.fromtimestamp(post.created_utc).isoformat(),
            "subreddit": post.subreddit.display_name,
            "score": post.score,
            "comments_count": post.num_comments,
            "url": post.url,
            "is_self": post.is_self
        }
    
    def crawl(self, subreddit: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Crawl posts from a subreddit and parse them
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to fetch
            
        Returns:
            List of parsed posts
        """
        posts = self.get_subreddit_posts(subreddit, limit)
        return [self.parse_post(post) for post in posts]
