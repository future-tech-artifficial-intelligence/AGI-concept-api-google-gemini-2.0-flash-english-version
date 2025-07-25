import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import direct_file_access as dfa

class AILearningSystem:
    """**System allowing artificial intelligence API GOOGLE GEMINI 2.0 FLASH to learn autonomously from project files.**
This system uses direct file access to enhance the capabilities of artificial intelligence API GOOGLE GEMINI 2.0 FLASH;
thanks to the Searx search engine, artificial intelligence API GOOGLE GEMINI 2.0 FLASH can learn autonomously by conducting searches.
    """

    def __init__(self):
        """Initializes the autonomous learning system."""
        self.base_path = Path(os.getcwd())
        self.learning_memory_file = self.base_path / "ai_learning_memory.json"
        self.learning_memory = self._load_learning_memory()
        self.learning_priorities = {
            "code_examples": 5,
            "documentation": 4,
            "data_structures": 4,
            "algorithms": 5,
            "configuration": 3,
            "project_structure": 3
        }

    def _load_learning_memory(self) -> Dict[str, Any]:
        """Loads the learning memory."""
        if self.learning_memory_file.exists():
            try:
                with open(self.learning_memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_default_memory()
        return self._create_default_memory()

    def _create_default_memory(self) -> Dict[str, Any]:
        """Creates a default learning memory structure."""
        return {
            "learned_files": {},
            "knowledge_areas": {
                "code_examples": [],
                "documentation": [],
                "data_structures": [],
                "algorithms": [],
                "configuration": [],
                "project_structure": []
            },
            "learning_sessions": [],
            "last_learning_time": 0
        }

    def _save_learning_memory(self) -> None:
        """Saves the learning memory."""
        with open(self.learning_memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.learning_memory, f, indent=2)

    def classify_file_content(self, file_path: str, content: str) -> List[str]:
        """
        Classifies the content of a file into different knowledge categories.

        Args:
            file_path: File path
            content: File content

        Returns:
            List of knowledge categories associated with the file
        """
        categories = []
        file_ext = Path(file_path).suffix.lower()

        # Classification based on extension
        if file_ext in ['.py', '.js', '.cpp', '.java', '.go']:
            categories.append("code_examples")

            # More detailed content analysis
            if "class " in content and "def " in content:
                categories.append("data_structures")

            if any(algo in content.lower() for algo in ["sort", "search", "algorithm", "optimize",
                                                       "recursive", "iteration"]):
                categories.append("algorithms")

        # Documentation files
        if file_ext in ['.md', '.rst', '.txt'] or "readme" in file_path.lower():
            categories.append("documentation")

        # Configuration files
        if file_ext in ['.json', '.yml', '.yaml', '.ini', '.cfg', '.conf', '.toml']:
            categories.append("configuration")

        # Project structure detection
        if any(term in file_path.lower() for term in ["setup", "main", "init", "config", "structure"]):
            categories.append("project_structure")

        return categories

    def learn_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Learns from the content of a file.

        Args:
            file_path: Path of the file to learn from
            
        Returns:
            Learning results
        """
        content = dfa.read_file_content(file_path)
        if content.startswith("Error:"): # Translated "Erreur:" to "Error:"
            return {"success": False, "message": content}

        # Classify content
        categories = self.classify_file_content(file_path, content)

        # Analyze and memorize content
        insights = self._extract_insights(file_path, content, categories)

        # Update learning memory
        if file_path not in self.learning_memory["learned_files"]:
            self.learning_memory["learned_files"][file_path] = {
                "categories": categories,
                "last_learned": time.time(),
                "learn_count": 1,
                "insights": insights
            }
        else:
            self.learning_memory["learned_files"][file_path]["categories"] = categories
            self.learning_memory["learned_files"][file_path]["last_learned"] = time.time()
            self.learning_memory["learned_files"][file_path]["learn_count"] += 1
            self.learning_memory["learned_files"][file_path]["insights"] = insights

        # Update knowledge areas
        for category in categories:
            if category in self.learning_memory["knowledge_areas"]:
                if file_path not in self.learning_memory["knowledge_areas"][category]:
                    self.learning_memory["knowledge_areas"][category].append(file_path)

        self._save_learning_memory()

        return {
            "success": True,
            "file_path": file_path,
            "categories": categories,
            "insights": insights
        }

    def _extract_insights(self, file_path: str, content: str, categories: List[str]) -> Dict[str, Any]:
        """
        Extracts relevant information from a file's content.

        Args:
            file_path: File path
            content: File content
            categories: Categories associated with the file

        Returns:
            Extracted insights
        """
        insights = {
            "summary": "",
            "key_concepts": [],
            "code_patterns": [],
            "dependencies": []
        }

        # Generate a summary based on file type
        file_size = len(content)
        file_ext = Path(file_path).suffix.lower()
        lines = content.split('\n')

        # Simple summary based on the first non-empty lines
        summary_lines = []
        for line in lines[:20]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 5:
                summary_lines.append(stripped)
                if len(summary_lines) >= 3:
                    break

        insights["summary"] = " ".join(summary_lines)

        # Identify key concepts
        key_concepts = set()

        # For Python files
        if file_ext == '.py':
            # Identify classes and functions
            for line in lines:
                if line.strip().startswith("def "):
                    fn_name = line.strip()[4:].split('(')[0]
                    key_concepts.add(f"function:{fn_name}")
                elif line.strip().startswith("class "):
                    class_name = line.strip()[6:].split('(')[0].split(':')[0]
                    key_concepts.add(f"class:{class_name}")

            # Identify imports
            for line in lines:
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    module = line.strip().split()[1]
                    insights["dependencies"].append(module)

        insights["key_concepts"] = list(key_concepts)[:10]  # Limit to 10 concepts

        # Identify common code patterns
        if "code_examples" in categories:
            if "if __name__ == \"__main__\":" in content:
                insights["code_patterns"].append("main_guard")
            if "try:" in content and "except:" in content:
                insights["code_patterns"].append("error_handling")
            if "class " in content and ("def __init__" in content):
                insights["code_patterns"].append("class_definition")
            if "def " in content and "return " in content:
                insights["code_patterns"].append("function_with_return")
            if "with " in content:
                insights["code_patterns"].append("context_manager")

        return insights

    def execute_learning_session(self, focus_area: Optional[str] = None, max_files: int = 5) -> Dict[str, Any]:
        """
        Executes an autonomous learning session.

        Args:
            focus_area: Knowledge area to prioritize
            max_files: Maximum number of files to analyze

        Returns:
            Learning session results
        """
        session_start = time.time()
        files = dfa.scan_project_files()

        # Filter recently learned files (less than 24h)
        recent_cutoff = time.time() - (24 * 3600)
        learned_recently = {
            file_path for file_path, info in self.learning_memory["learned_files"].items()
            if info["last_learned"] > recent_cutoff
        }

        candidate_files = [file for file in files if file not in learned_recently]

        # If a focus area is specified, prioritize files from that area
        if focus_area and focus_area in self.learning_memory["knowledge_areas"]:
            # Supplement with files from the focus area if they haven't been learned recently
            focus_files = [
                file for file in self.learning_memory["knowledge_areas"][focus_area]
                if file not in learned_recently and file in files
            ]

            # Add other files to reach max_files
            if len(focus_files) < max_files:
                remaining_files = [file for file in candidate_files if file not in focus_files]
                focus_files.extend(remaining_files[:max_files - len(focus_files)])

            files_to_learn = focus_files[:max_files]
        else:
            # Select files to learn
            files_to_learn = candidate_files[:max_files]

        # Learn from each file
        learning_results = []
        for file_path in files_to_learn:
            result = self.learn_from_file(file_path)
            learning_results.append(result)

        # Record the learning session
        session = {
            "timestamp": session_start,
            "duration": time.time() - session_start,
            "focus_area": focus_area,
            "files_learned": [result["file_path"] for result in learning_results if result["success"]],
            "insights_gained": sum(len(result.get("insights", {}).get("key_concepts", []))
                                  for result in learning_results if result["success"])
        }

        self.learning_memory["learning_sessions"].append(session)
        self.learning_memory["last_learning_time"] = time.time()
        self._save_learning_memory()

        return {
            "session": session,
            "results": learning_results
        }

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Generates a summary of the knowledge acquired by the AI.

        Returns:
            Knowledge summary
        """
        summary = {
            "files_learned": len(self.learning_memory["learned_files"]),
            "knowledge_areas": {},
            "recent_sessions": min(5, len(self.learning_memory["learning_sessions"])),
            "total_sessions": len(self.learning_memory["learning_sessions"]),
            "top_insights": []
        }

        # Summarize knowledge areas
        for area, files in self.learning_memory["knowledge_areas"].items():
            summary["knowledge_areas"][area] = len(files)

        # Identify the most frequent key concepts
        concept_frequency = {}
        for file_info in self.learning_memory["learned_files"].values():
            if "insights" in file_info and "key_concepts" in file_info["insights"]:
                for concept in file_info["insights"]["key_concepts"]:
                    if concept in concept_frequency:
                        concept_frequency[concept] += 1
                    else:
                        concept_frequency[concept] = 1

        # Sort concepts by frequency
        sorted_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
        summary["top_insights"] = sorted_concepts[:10]  # Top 10 concepts

        return summary

    def suggest_learning_focus(self) -> str:
        """
        Suggests a knowledge area to prioritize for the next session.

        Returns:
            Recommended knowledge area
        """
        # Analyze knowledge distribution
        area_counts = {area: len(files) for area, files in self.learning_memory["knowledge_areas"].items()}

        # Calculate a score for each area based on priority and number of files learned
        scores = {}
        for area, priority in self.learning_priorities.items():
            count = area_counts.get(area, 0)
            # Formula: higher priority and fewer files lead to a higher score
            scores[area] = priority * (1 + 1/(count + 1))

        # Identify the area with the highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return "code_examples"  # Default value

# Main interface for the AI
ai_learning = AILearningSystem()

def start_learning_session(focus_area=None, max_files=5):
    """Starts an autonomous learning session."""
    return ai_learning.execute_learning_session(focus_area, max_files)

def learn_specific_file(file_path):
    """Learns from a specific file."""
    return ai_learning.learn_from_file(file_path)

def get_learning_summary():
    """Gets a summary of acquired knowledge."""
    return ai_learning.get_knowledge_summary()

def get_suggested_focus():
    """Suggests a knowledge area to prioritize."""
    return ai_learning.suggest_learning_focus()

if __name__ == "__main__":
    # Test the learning system
    print("Starting an autonomous learning session...")
    session = start_learning_session(max_files=3)

    print(f"Session completed. {len(session['results'])} files analyzed.")

    print("\nKnowledge summary:")
    summary = get_learning_summary()
    print(f"Files learned: {summary['files_learned']}")
    print("Knowledge areas:")
    for area, count in summary['knowledge_areas'].items():
        print(f"- {area}: {count} files")

    print(f"\nRecommended focus area: {get_suggested_focus()}")
