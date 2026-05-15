"""GitHub Issues crawler module"""

import os
import requests
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class GitHubCrawler:
    """Crawler for GitHub Issues API"""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def get_issues(self, owner: str, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        """
        Fetch issues from a GitHub repository
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state (open, closed, all)
            
        Returns:
            List of issue dictionaries
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues"
        params = {"state": state, "per_page": 100}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching issues: {e}")
            return []
    
    def parse_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw issue data into standardized format
        
        Args:
            issue: Raw issue data from GitHub API
            
        Returns:
            Parsed issue dictionary
        """
        return {
            "source": "github",
            "id": issue["id"],
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"],
            "author": issue["user"]["login"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "state": issue["state"],
            "comments_count": issue["comments"],
            "url": issue["html_url"],
            "labels": [label["name"] for label in issue["labels"]]
        }
    
    def crawl(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Crawl all issues from a repository and parse them
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            List of parsed issues
        """
        issues = self.get_issues(owner, repo, state="all")
        return [self.parse_issue(issue) for issue in issues]
