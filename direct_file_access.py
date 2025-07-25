import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

class DirectFileAccess:
    """
    System allowing the artificial intelligence API GOOGLE GEMINI 2.0 FLASH to directly access project text files
    to facilitate its autonomous learning.
    Thanks to the Searx search engine, artificial intelligence API GOOGLE GEMINI 2.0 FLASH can learn autonomously during its searches.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initializes the direct file access system.

        Args:
            base_path: Base path of the project. If not specified, uses the current directory.
        """
        self.base_path = Path(base_path) if base_path else Path(os.getcwd())
        self.memory_file = self.base_path / "ai_file_memory.json"
        self.file_memory = self._load_memory()
        self.text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.csv', '.xml', '.yml', '.yaml', '.ini', '.cfg'}
        self.excluded_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.vscode', '.idea'}

        # Define the path to the text data folder
        self.text_data_folder = self.base_path / 'gemini_text_data'
        self.ensure_text_data_folder_exists()

    def ensure_text_data_folder_exists(self):
        """Creates the text data folder if it does not exist."""
        if not self.text_data_folder.exists():
            os.makedirs(self.text_data_folder)
            print(f"Text data folder created: {self.text_data_folder}")
        return self.text_data_folder

    def _load_memory(self) -> Dict[str, Any]:
        """Loads the memory of consulted files."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"files": {}, "last_scan": 0}
        return {"files": {}, "last_scan": 0}

    def _save_memory(self) -> None:
        """Saves the memory of consulted files."""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_memory, f, indent=2)

    def scan_files(self, force: bool = False) -> List[str]:
        """
        Scans the file system to identify all text files.

        Args:
            force: Forces a new scan even if a recent scan has been performed

        Returns:
            List of paths of found text files
        """
        current_time = time.time()

        # Avoid too frequent scanning (max once every 5 minutes)
        if not force and current_time - self.file_memory.get("last_scan", 0) < 300:
            return [file for file in self.file_memory.get("files", {})]

        file_list = []

        for root, dirs, files in os.walk(self.base_path):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.base_path)

                if file_path.suffix.lower() in self.text_extensions:
                    str_path = str(rel_path)
                    file_list.append(str_path)

                    # Update file metadata
                    if str_path not in self.file_memory["files"]:
                        self.file_memory["files"][str_path] = {
                            "last_accessed": None,
                            "access_count": 0,
                            "relevance_score": 0
                        }

        # Remove files that no longer exist
        for file_path in list(self.file_memory["files"].keys()):
            if file_path not in file_list:
                del self.file_memory["files"][file_path]

        self.file_memory["last_scan"] = current_time
        self._save_memory()

        return file_list

    def read_file(self, file_path: str) -> str:
        """
        Reads the content of a text file.

        Args:
            file_path: Relative path of the file to read

        Returns:
            Content of the file
        """
        full_path = self.base_path / file_path

        if not full_path.exists():
            return f"Error: The file {file_path} does not exist."

        if not full_path.is_file():
            return f"Error: {file_path} is not a file."

        if full_path.suffix.lower() not in self.text_extensions:
            return f"Error: {file_path} is not a text file."

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Update file metadata
            rel_path = str(full_path.relative_to(self.base_path))
            if rel_path in self.file_memory["files"]:
                self.file_memory["files"][rel_path]["last_accessed"] = time.time()
                self.file_memory["files"][rel_path]["access_count"] += 1
                self._save_memory()

            return content
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

    def search_in_files(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for files containing the specified text.

        Args:
            query: Text to search for in files
            max_results: Maximum number of results to return

        Returns:
            List of matching files with relevant excerpts
        """
        results = []
        files = self.scan_files()

        for file_path in files:
            full_path = self.base_path / file_path
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                if query.lower() in content.lower():
                    # Find a relevant excerpt
                    index = content.lower().find(query.lower())
                    start = max(0, index - 100)
                    end = min(len(content), index + len(query) + 100)

                    # Adjust to not cut words
                    while start > 0 and content[start] != ' ' and content[start] != '\n':
                        start -= 1

                    while end < len(content) and content[end] != ' ' and content[end] != '\n':
                        end += 1

                    excerpt = content[start:end]
                    if start > 0:
                        excerpt = "..." + excerpt
                    if end < len(content):
                        excerpt = excerpt + "..."

                    results.append({
                        "file_path": file_path,
                        "excerpt": excerpt,
                        "relevance": self._calculate_relevance(file_path, query, content)
                    })

                    if len(results) >= max_results:
                        break
            except Exception:
                continue

        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)

        return results

    def _calculate_relevance(self, file_path: str, query: str, content: str) -> float:
        """
        Calculates a relevance score for a file relative to a query.

        Args:
            file_path: Path to the file
            query: Search terms
            content: Content of the file

        Returns:
            Relevance score
        """
        score = 0.0

        # Number of occurrences of the search term
        occurrences = content.lower().count(query.lower())
        score += min(occurrences * 0.5, 5.0)  # Capped at 5.0

        # Recency of access
        file_info = self.file_memory["files"].get(file_path, {})
        last_accessed = file_info.get("last_accessed")
        if last_accessed:
            days_since_access = (time.time() - last_accessed) / (24 * 3600)
            recency_score = max(0, 3.0 - (days_since_access / 7))  # Score decreases with time
            score += recency_score

        # Frequency of access
        access_count = file_info.get("access_count", 0)
        score += min(access_count * 0.2, 2.0)  # Capped at 2.0

        # Update relevance score in memory
        if file_path in self.file_memory["files"]:
            # Moving average
            current_score = self.file_memory["files"][file_path].get("relevance_score", 0)
            self.file_memory["files"][file_path]["relevance_score"] = current_score * 0.7 + score * 0.3
            self._save_memory()

        return score

    def suggest_relevant_files(self, context: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggests relevant files based on the current context.

        Args:
            context: Current context (question, discussion, etc.)
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested files
        """
        # Extract keywords from the context
        keywords = self._extract_keywords(context)

        # Calculate scores for each file
        scores = {}
        for file_path in self.file_memory["files"]:
            score = 0.0

            # Base score from memory
            file_info = self.file_memory["files"][file_path]
            score += file_info.get("relevance_score", 0) * 0.5

            # Add points for keyword matches in the filename
            for keyword in keywords:
                if keyword.lower() in file_path.lower():
                    score += 2.0

            scores[file_path] = score

        # Return the most relevant files
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:max_suggestions]

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extracts relevant keywords from the text."""
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned.split()

        # Filter short words and common words
        common_words = {"le", "la", "les", "un", "une", "des", "et", "ou", "à", "de", "du", "dans",
                       "en", "sur", "pour", "par", "avec", "sans", "ce", "cette", "ces", "mon", "ma",
                       "mes", "ton", "ta", "tes", "son", "sa", "ses", "qui", "que", "quoi", "comment"} # Common French words
        
        keywords = {word for word in words if len(word) > 3 and word not in common_words}
        return keywords

    def get_file_structure(self) -> Dict[str, Any]:
        """
        Returns a hierarchical structure of project files.

        Returns:
            Dictionary representing the project tree
        """
        structure = {}
        files = self.scan_files()

        for file_path in files:
            parts = Path(file_path).parts
            current = structure

            # Traverse the tree and build it
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # It's a file
                    if "files" not in current:
                        current["files"] = []
                    current["files"].append(part)
                else:  # It's a directory
                    if "dirs" not in current:
                        current["dirs"] = {}
                    if part not in current["dirs"]:
                        current["dirs"][part] = {}
                    current = current["dirs"][part]

        return structure

# Simple interface for AI use
file_access = DirectFileAccess()

def scan_project_files(force=False):
    """Scans the project and returns the list of available text files."""
    return file_access.scan_files(force)

def read_file_content(file_path):
    """Reads the content of a text file."""
    return file_access.read_file(file_path)

def search_files(query, max_results=10):
    """Searches for files containing the specified text."""
    return file_access.search_in_files(query, max_results)

def get_relevant_files(context, max_suggestions=5):
    """Suggests relevant files based on the context."""
    return file_access.suggest_relevant_files(context, max_suggestions)

def get_project_structure():
    """Returns the hierarchical structure of the project."""
    return file_access.get_file_structure()

if __name__ == "__main__":
    # Test the system
    print("Scanning project files...")
    files = scan_project_files(force=True)
    print(f"Number of files found: {len(files)}")

    if files:
        print("\nFirst files found:")
        for i, file in enumerate(files[:5]):
            print(f"- {file}")

        print("\nSearching for files containing 'AI':")
        results = search_files("IA", 3) # 'IA' is French for 'AI'
        for result in results:
            print(f"- {result['file_path']}")
            print(f"  Excerpt: {result['excerpt'][:100]}...")
