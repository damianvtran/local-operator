import fnmatch
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import playwright.async_api as pw

from local_operator.rag import EmbeddingManager, InsightResult


def _get_git_ignored_files(gitignore_path: str) -> Set[str]:
    """Get list of files ignored by git from a .gitignore file.

    Args:
        gitignore_path: Path to the .gitignore file. Defaults to ".gitignore"

    Returns:
        Set of glob patterns for ignored files. Returns empty set if gitignore doesn't exist.
    """
    ignored = set()
    try:
        with open(gitignore_path) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    ignored.add(line)
        return ignored
    except FileNotFoundError:
        return set()


def _should_ignore_file(file_path: str) -> bool:
    """Determine if a file should be ignored based on common ignored paths and git ignored files."""
    # Common ignored directories
    ignored_dirs = {
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        ".git",
        ".idea",
        ".vscode",
        "build",
        "dist",
        "target",
        "bin",
        "obj",
        "out",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        ".tox",
        ".eggs",
        ".env",
        "env",
        "htmlcov",
        "coverage",
        ".DS_Store",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.egg",
        "*.egg-info",
        ".ipynb_checkpoints",
        ".sass-cache",
        ".gradle",
        "tmp",
        "temp",
        "logs",
        "log",
        ".next",
        ".nuxt",
        ".cache",
        ".parcel-cache",
        "public/uploads",
        "uploads",
        "vendor",
        "bower_components",
        "jspm_packages",
        ".serverless",
        ".terraform",
        ".vagrant",
        ".bundle",
        "coverage",
        ".nyc_output",
    }

    # Check if file is in an ignored directory
    path_parts = Path(file_path).parts
    for part in path_parts:
        if part in ignored_dirs:
            return True

    return False


def get_current_directory_info() -> Dict[str, List[Tuple[str, str, int]]]:
    """Walk over the current directory and return a dictionary of files and their
    metadata.  If in a git repo, only shows unignored files. If not in a git repo,
    shows all files.

    Returns:
        Dict mapping directory paths to lists of (filename, file_type, size_bytes) tuples.
        File types are: 'code', 'doc', 'image', 'other'
    """
    directory_index = {}

    # Try to get git ignored files, empty set if not in git repo
    ignored_files = _get_git_ignored_files(".gitignore")

    for root, dirs, files in os.walk("."):
        # Skip .git directory if it exists
        if ".git" in dirs:
            dirs.remove(".git")

        # Skip common ignored files
        files = [f for f in files if not _should_ignore_file(os.path.join(root, f))]

        # Apply glob patterns to filter out ignored files
        filtered_files = []
        for file in files:
            file_path = os.path.join(root, file)
            should_ignore = False
            for ignored_pattern in ignored_files:
                if fnmatch.fnmatch(file_path, ignored_pattern):
                    should_ignore = True
                    break
            if not should_ignore:
                filtered_files.append(file)
        files = filtered_files

        path = Path(root)
        dir_files = []

        for file in sorted(files):
            file_path = os.path.join(root, file)
            size = os.stat(file_path).st_size
            ext = Path(file).suffix.lower()

            # Categorize file type
            if ext in [
                ".py",
                ".js",
                ".java",
                ".cpp",
                ".h",
                ".c",
                ".go",
                ".rs",
                ".ts",
                ".jsx",
                ".tsx",
                ".php",
                ".rb",
                ".cs",
                ".swift",
                ".kt",
                ".scala",
                ".r",
                ".m",
                ".mm",
                ".pl",
                ".sh",
                ".bash",
                ".zsh",
                ".fish",
                ".sql",
                ".vue",
                ".elm",
                ".clj",
                ".ex",
                ".erl",
                ".hs",
                ".lua",
                ".jl",
                ".nim",
                ".ml",
                ".fs",
                ".f90",
                ".f95",
                ".f03",
                ".pas",
                ".groovy",
                ".dart",
                ".coffee",
                ".ls",
            ]:
                file_type = "code"
            elif ext in [
                ".md",
                ".txt",
                ".rst",
                ".json",
                ".yaml",
                ".yml",
                ".ini",
                ".toml",
                ".xml",
                ".html",
                ".htm",
                ".css",
                ".csv",
                ".tsv",
                ".log",
                ".conf",
                ".cfg",
                ".properties",
                ".env",
                ".doc",
                ".docx",
                ".pdf",
                ".rtf",
                ".odt",
                ".tex",
                ".adoc",
                ".org",
                ".wiki",
                ".textile",
                ".pod",
            ]:
                file_type = "doc"
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".svg",
                ".ico",
                ".bmp",
                ".tiff",
                ".tif",
                ".webp",
                ".raw",
                ".psd",
                ".ai",
                ".eps",
                ".heic",
                ".heif",
                ".avif",
            ]:
                file_type = "image"
            else:
                file_type = "other"

            dir_files.append((file, file_type, size))

        if dir_files:
            directory_index[str(path)] = dir_files

    return directory_index


async def browse_single_url(url: str) -> str:
    """Browse to a URL using Playwright to render JavaScript and return the page content.

    Args:
        url: The URL to browse to

    Returns:
        str: The rendered page content
    """
    try:
        async with pw.async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            content = await page.content()
            await browser.close()
            return content
    except Exception as e:
        raise RuntimeError(f"Failed to browse {url}: {str(e)}")


def add_info_to_knowledge_base(rag_manager: EmbeddingManager, info: str) -> None:
    """Add information to the knowledge base and save the manager.  Use this to store
    large amounts of information that shouldn't be printed to the console.

    Args:
        rag_manager: The RAG manager instance
        info: The information to add to the knowledge base
    """
    rag_manager.add_large_text(info)
    rag_manager.save()


def query_knowledge_base(
    rag_manager: EmbeddingManager, query: str, num_results: int = 10, max_distance: float = 3.0
) -> str:
    """
    Query the knowledge base and return the results as a string which
    can be printed to the console.

    Args:
        rag_manager: The RAG manager instance
        query: The query string to search for
        num_results: The number of results to return
        max_distance: The maximum distance for results
    Returns:
        str: A string containing the matching insights, one per line
    """
    results = rag_manager.query_insight(query, k=num_results, max_distance=max_distance)
    return "\n".join(result.insight for result in results)


class ToolRegistry:
    """Registry for tools that can be used by agents.

    The ToolRegistry maintains a collection of callable tools that agents can access and execute.
    It provides methods to initialize with default tools, add custom tools, and retrieve
    tools by name.

    Attributes:
        tools (dict): Dictionary mapping tool names to their callable implementations
    """

    _tools: Dict[str, Callable[..., Any]]

    def __init__(self):
        """Initialize an empty tool registry."""
        # Initialize _tools first before calling super().__init__()
        super().__init__()
        object.__setattr__(self, "_tools", {})

    def init_tools(self):
        """Initialize the registry with default tools.

        Default tools include:
        - browse_single_url: Browse a URL and get page content
        - get_current_directory_info: Get information about the current directory
        """
        self.add_tool("browse_single_url", browse_single_url)
        self.add_tool("get_current_directory_info", get_current_directory_info)

    def add_tool(self, name: str, tool: Callable[..., Any]):
        """Add a new tool to the registry.

        Args:
            name (str): Name to register the tool under
            tool (Callable[..., Any]): The tool implementation function/callable with any arguments
        """
        self._tools[name] = tool
        super().__setattr__(name, tool)

    def get_tool(self, name: str) -> Callable[..., Any]:
        """Retrieve a tool from the registry by name.

        Args:
            name (str): Name of the tool to retrieve

        Returns:
            Callable[..., Any]: The requested tool implementation that can accept any arguments
        """
        return self._tools[name]

    def remove_tool(self, name: str) -> None:
        """Remove a tool from the registry by name.

        Args:
            name (str): Name of the tool to remove
        """
        del self._tools[name]
        delattr(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute on the registry.

        Args:
            name (str): Name of the attribute
            value (Any): Value to set
        """
        # Only add to _tools if it's not _tools itself
        if name != "_tools":
            self._tools[name] = value
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Allow accessing tools as attributes.

        Args:
            name (str): Name of the tool to retrieve

        Returns:
            Callable[..., Any]: The requested tool implementation

        Raises:
            AttributeError: If the requested tool does not exist
        """
        try:
            return self._tools[name]
        except KeyError:
            raise AttributeError(f"Tool '{name}' not found in registry")

    def __iter__(self):
        """Make the registry iterable.

        Returns:
            Iterator[str]: Iterator over tool names in the registry
        """
        return iter(self._tools)
