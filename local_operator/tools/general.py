import asyncio
import base64
import fnmatch
import http.client
import json
import os
import platform
import shutil
import socket
from datetime import datetime, timezone  # Added timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID  # Added UUID

import playwright.async_api as pw
from browser_use import Agent as BrowserAgent
from browser_use import Browser, BrowserConfig
from browser_use import Controller as BrowserController
from pydantic import SecretStr

from local_operator.agents import AgentRegistry
from local_operator.clients.fal import FalClient, FalImageGenerationResponse, ImageSize
from local_operator.clients.radient import (
    RadientClient,
    RadientImageGenerationResponse,
    RadientSearchResponse,
    RadientSendEmailResponseData,
)
from local_operator.clients.serpapi import SerpApiClient, SerpApiResponse
from local_operator.clients.tavily import TavilyClient, TavilyResponse
from local_operator.credentials import CredentialManager
from local_operator.mocks import ChatMock, ChatNoop
from local_operator.model.configure import ModelConfiguration
from local_operator.tools.google import (
    GOOGLE_ACCESS_TOKEN_KEY,
    create_calendar_event_tool,
    create_gmail_draft_tool,
    delete_calendar_event_tool,
    delete_gmail_draft_tool,
    download_drive_file_tool,
    get_gmail_message_tool,
    list_calendar_events_tool,
    list_drive_files_tool,
    list_gmail_messages_tool,
    send_gmail_draft_tool,
    send_gmail_message_tool,
    update_calendar_event_tool,
    update_drive_file_content_tool,
    update_drive_file_metadata_tool,
    update_gmail_draft_tool,
    upload_drive_file_tool,
)
from local_operator.types import Schedule, ScheduleUnit


def _get_git_ignored_files(gitignore_path: str) -> Set[str]:
    """Get list of files ignored by git from a .gitignore file.

    Args:
        gitignore_path: Path to the .gitignore file. Defaults to ".gitignore"

    Returns:
        Set of glob patterns for ignored files. Returns empty set if gitignore doesn't exist.
    """
    ignored = set()
    try:
        with open(gitignore_path, "r") as f:
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


def list_working_directory(max_depth: int = 3) -> Dict[str, List[Tuple[str, str, int]]]:
    """List the files in the current directory showing files and their metadata.
    If in a git repo, only shows unignored files. If not in a git repo, shows all files.

    Args:
        max_depth: Maximum directory depth to traverse. Defaults to 3.

    Returns:
        Dict mapping directory paths to lists of (filename, file_type, size_bytes) tuples.
        File types are: 'code', 'doc', 'data', 'image', 'config', 'other'
    """
    directory_index = {}

    # Try to get git ignored files, empty set if not in git repo
    ignored_files = _get_git_ignored_files(".gitignore")

    for root, dirs, files in os.walk("."):
        # Skip if we've reached max depth
        depth = root.count(os.sep)
        if depth >= max_depth:
            dirs.clear()  # Clear dirs to prevent further recursion
            continue

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
            try:
                size = os.stat(file_path).st_size
            except Exception:
                # Skip files that can't be accessed
                continue

            ext = Path(file).suffix.lower()
            filename = Path(file).name.lower()

            # Categorize file type
            if filename in [
                # Version Control
                ".gitignore",
                ".gitattributes",
                ".gitmodules",
                ".hgignore",
                ".svnignore",
                # Docker
                ".dockerignore",
                "Dockerfile",
                "docker-compose.yml",
                "docker-compose.yaml",
                # Node/JS
                ".npmignore",
                ".npmrc",
                ".nvmrc",
                "package.json",
                "package-lock.json",
                "yarn.lock",
                # Python
                ".flake8",
                "pyproject.toml",
                "setup.cfg",
                "setup.py",
                "requirements.txt",
                "requirements-dev.txt",
                "Pipfile",
                "Pipfile.lock",
                "poetry.lock",
                "tox.ini",
                # Code Style/Linting
                ".eslintrc",
                ".eslintignore",
                ".prettierrc",
                ".editorconfig",
                ".stylelintrc",
                ".pylintrc",
                "mypy.ini",
                ".black",
                ".isort.cfg",
                "prettier.config.js",
                # Build/CI
                ".travis.yml",
                ".circleci/config.yml",
                ".github/workflows/*.yml",
                "Jenkinsfile",
                "azure-pipelines.yml",
                ".gitlab-ci.yml",
                "bitbucket-pipelines.yml",
                # Environment/Config
                ".env",
                ".env.example",
                ".env.template",
                ".env.sample",
                ".env.local",
                ".env.development",
                ".env.production",
                ".env.test",
                # Build Systems
                "Makefile",
                "CMakeLists.txt",
                "build.gradle",
                "pom.xml",
                "build.sbt",
                # Web/Frontend
                "tsconfig.json",
                "webpack.config.js",
                "babel.config.js",
                ".babelrc",
                "rollup.config.js",
                "vite.config.js",
                "next.config.js",
                "nuxt.config.js",
                # Other Languages
                "composer.json",
                "composer.lock",
                "Gemfile",
                "Gemfile.lock",
                "cargo.toml",
                "mix.exs",
                "rebar.config",
                "stack.yaml",
                "deno.json",
                "go.mod",
                "go.sum",
            ]:
                file_type = "config"
            elif ext in [
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
                ".csv",
                ".tsv",
                ".xlsx",
                ".xls",
                ".parquet",
                ".arrow",
                ".feather",
                ".hdf5",
                ".h5",
                ".dta",
                ".sas7bdat",
                ".sav",
                ".arff",
                ".ods",
                ".fods",
                ".dbf",
                ".mdb",
                ".accdb",
            ]:
                file_type = "data"
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


async def get_page_html_content(url: str) -> str:
    """Browse to a URL using Playwright to render JavaScript and return the full HTML page content.  Use this for any URL that you want to get the full HTML content of for scraping and understanding the HTML format of the page.

    Uses stealth mode and waits for network idle to avoid bot detection.

    Args:
        url: The URL to browse to

    Returns:
        str: The rendered page content

    Raises:
        RuntimeError: If page loading fails or bot detection is triggered
    """  # noqa: E501
    try:
        async with pw.async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
            )
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = await context.new_page()

            # Add stealth mode
            await page.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {get: () => false});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                window.chrome = { runtime: {} };
            """
            )

            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)  # Wait additional time for dynamic content

            content = await page.content()
            await browser.close()
            return content

    except Exception as e:
        raise RuntimeError(f"Failed to get raw page content for {url}: {str(e)}")


async def get_page_text_content(url: str) -> str:
    """Browse to a URL using Playwright to render JavaScript and extract clean text content.  Use this for any URL that you want to read the content for, for research purposes. Extracts text from semantic elements like headings, paragraphs, lists etc. and returns a cleaned text representation of the page content.

    Uses stealth mode and waits for network idle to avoid bot detection.
    Extracts text from semantic elements and returns cleaned content.

    Args:
        url: The URL to get the text content of

    Returns:
        str: The cleaned text content extracted from the page's semantic elements

    Raises:
        RuntimeError: If page loading or text extraction fails
    """  # noqa: E501
    try:
        async with pw.async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
            )
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = await context.new_page()

            # Add stealth mode
            await page.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {get: () => false});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                window.chrome = { runtime: {} };
            """
            )

            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)  # Wait additional time for dynamic content

            # Extract text from semantic elements
            text_elements = await page.evaluate(
                """
                () => {
                    const selectors = 'h1, h2, h3, h4, h5, h6, p, li, td, th, figcaption, pre, blockquote, code, a';
                    const elements = document.querySelectorAll(selectors);
                    return Array.from(elements)
                        .map(el => el.textContent)
                        .filter(text => text && text.trim())
                        .map(text => text.trim())
                        .map(text => text.replace(/\\s+/g, ' '));
                }
            """  # noqa: E501
            )

            await browser.close()

            # Clean and join the text elements
            cleaned_text = "\n".join(text_elements)
            return cleaned_text

    except Exception as e:
        raise RuntimeError(f"Failed to extract text content from {url}: {str(e)}")


def generate_altered_image_tool(
    fal_client: FalClient | None, radient_client: RadientClient | None
) -> Callable[..., Any]:
    """Alter existing images using the FAL or Radient API.

    Makes a request to the FAL or Radient API using the provided API key to modify existing images
    based on text prompts. Uses the FLUX.1 image-to-image model for FAL or the appropriate
    model for Radient.

    Args:
        fal_client (FalClient | None): The FAL API client to use
        radient_client (RadientClient | None): The Radient API client to use

    Returns:
        Callable: A function that alters images based on text prompts
    """

    def generate_altered_image(
        image_path: str,
        prompt: str,
        strength: float = 0.95,
        num_inference_steps: int = 40,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_images: int = 1,
    ) -> FalImageGenerationResponse | RadientImageGenerationResponse:
        """Alter an existing image using the FAL or Radient API. This tool allows you to modify an existing image based on a text prompt. You must provide a path to an image file on disk and a detailed prompt describing how you want to modify the image. When using this tool, save the resulting image to a file so that the user can access it on their computer.

        Args:
            image_path (str): Path to the image file on disk to modify
            prompt (str): Text description of how to modify the image
            strength (float, optional): Strength of the modification (0.0-1.0). Higher values
                result in more dramatic changes. Defaults to 0.95.
            num_inference_steps (int, optional): Number of inference steps. Higher values
                may produce better quality but take longer. Defaults to 40.
            seed (Optional[int], optional): Seed for reproducible generation. Defaults to None.
            guidance_scale (float, optional): How closely to follow the prompt (1-10).
                Defaults to 7.5.
            num_images (int, optional): Number of images to generate. Defaults to 1.

        Returns:
            FalImageGenerationResponse | RadientImageGenerationResponse: A response containing the generated image URLs and metadata

        Raises:
            RuntimeError: If no API client is available or the request fails
            FileNotFoundError: If the image file does not exist
        """  # noqa: E501
        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read the image file and convert to base64
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{base64_image}"

        # Try Radient first if available
        if radient_client:
            try:
                # Generate the image with image-to-image mode using Radient with polling
                response = radient_client.generate_image(
                    prompt=prompt,
                    source_url=data_uri,  # Pass the base64 data URI as the source_url
                    strength=strength,
                    num_images=num_images,
                    sync_mode=False,  # Force async mode to get a request_id for polling
                    max_wait_time=60,
                    poll_interval=2,
                )
                return response
            except Exception as e:
                # If Radient fails and FAL is available, fall back to FAL
                if fal_client:
                    print(f"Radient image alteration failed: {str(e)}. Falling back to FAL.")
                else:
                    raise RuntimeError(f"Radient image alteration failed: {str(e)}")

        # Use FAL if Radient is not available or failed
        if fal_client:
            # Generate the image with image-to-image mode using FAL
            response = fal_client.generate_image(
                prompt=prompt,
                image_url=data_uri,  # Pass the base64 data URI as the image_url
                strength=strength,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                sync_mode=False,  # Use the polling mechanism to get the result
                num_images=num_images,
                enable_safety_checker=True,
            )

            # Ensure we're returning a FalImageGenerationResponse
            if isinstance(response, FalImageGenerationResponse):
                return response
            else:
                # This shouldn't happen with sync_mode=True, but just in case
                raise RuntimeError("Failed to alter image: Unexpected response type")
        else:
            raise RuntimeError("No image generation API client is available")

    return generate_altered_image


def generate_image_tool(
    fal_client: FalClient | None, radient_client: RadientClient | None
) -> Callable[..., Any]:
    """Generate images using the FAL or Radient API.

    Makes a request to the FAL or Radient API using the provided API key to generate images
    from text prompts. Uses the FLUX.1 text-to-image model for FAL or the appropriate
    model for Radient.

    Args:
        fal_client (FalClient | None): The FAL API client to use
        radient_client (RadientClient | None): The Radient API client to use

    Returns:
        Callable: A function that generates images from text prompts
    """

    def generate_image(
        prompt: str,
        image_size: str = "landscape_4_3",
        num_inference_steps: int = 28,
        seed: Optional[int] = None,
        guidance_scale: float = 5.0,
        num_images: int = 1,
    ) -> FalImageGenerationResponse | RadientImageGenerationResponse:
        """Generate an image from a text prompt using the FAL or Radient API. This tool allows you to generate images from text descriptions. You must come up with a detailed prompt to describe the image that you want to generate. When using this tool, save the image to a file so that the user can access it on their computer.  The supported image sizes are: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9". Defaults to "landscape_4_3".

        Args:
            prompt (str): The text description to generate an image from
            image_size (str, optional): Size/aspect ratio of the generated image.
                Allowed values: "square_hd", "square", "portrait_4_3", "portrait_16_9",
                "landscape_4_3", "landscape_16_9". Defaults to "landscape_4_3".
            num_inference_steps (int, optional): Number of inference steps. Higher values
                may produce better quality but take longer. Defaults to 28.
            seed (Optional[int], optional): Seed for reproducible generation. Defaults to None.
            guidance_scale (float, optional): How closely to follow the prompt (1-10).
                Defaults to 5.0.
            num_images (int, optional): Number of images to generate. Defaults to 1.

        Returns:
            FalImageGenerationResponse | RadientImageGenerationResponse: A response containing the generated image URLs and metadata

        Raises:
            RuntimeError: If no API client is available or the request fails
        """  # noqa: E501
        # Try Radient first if available
        if radient_client:
            try:
                # Generate the image using Radient with polling
                response = radient_client.generate_image(
                    prompt=prompt,
                    image_size=image_size,
                    num_images=num_images,
                    sync_mode=False,  # Force async mode to get a request_id for polling
                    max_wait_time=60,
                    poll_interval=2,
                )
                return response
            except Exception as e:
                # If Radient fails and FAL is available, fall back to FAL
                if fal_client:
                    print(f"Radient image generation failed: {str(e)}. Falling back to FAL.")
                else:
                    raise RuntimeError(f"Radient image generation failed: {str(e)}")

        # Use FAL if Radient is not available or failed
        if fal_client:
            # Convert string image_size to enum
            try:
                size = ImageSize(image_size)
            except ValueError:
                raise ValueError(
                    f"Invalid image_size: {image_size}. Valid options are: "
                    f"{', '.join([s.value for s in ImageSize])}"
                )

            # Generate the image with FAL
            response = fal_client.generate_image(
                prompt=prompt,
                image_size=size,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                sync_mode=False,  # Use the polling mechanism to get the result
                num_images=num_images,
                enable_safety_checker=True,
            )

            # Ensure we're returning a FalImageGenerationResponse
            if isinstance(response, FalImageGenerationResponse):
                return response
            else:
                # This shouldn't happen with sync_mode=True, but just in case
                raise RuntimeError("Failed to generate image: Unexpected response type")
        else:
            raise RuntimeError("No image generation API client is available")

    return generate_image


def search_web_tool(
    serp_api_client: SerpApiClient | None,
    tavily_client: TavilyClient | None,
    radient_client: RadientClient | None,
) -> Callable[..., Any]:
    """Search the web using SERP API, Tavily API, or Radient API.

    Makes a request to the available search API using the provided API key to search the web.
    Supports multiple search providers and configurable result limits.

    Args:
        serp_api_client (SerpApiClient | None): The SERP API client to use
        tavily_client (TavilyClient | None): The Tavily API client to use
        radient_client (RadientClient | None): The Radient API client to use

    Returns:
        Callable: A function that searches the web and returns results
    """

    def search_web(
        query: str, search_engine: str = "google", max_results: int = 20
    ) -> SerpApiResponse | TavilyResponse | RadientSearchResponse:
        """Search the web using the Radient, SERP, or Tavily API and return the results.

        This tool allows the agent to search the internet for information. The results
        must be printed to the console. It will try Radient first, then SERP API, and
        finally Tavily API if available.

        Args:
            query (str): The search query string.
            search_engine (str, optional): Search engine to use (e.g., "google", "bing").
                Defaults to "google".
            max_results (int, optional): Maximum number of results to return. Defaults to 20.

        Returns:
            SerpApiResponse | TavilyResponse | RadientSearchResponse: A
            structured response containing search results.

        Raises:
            RuntimeError: If no search provider is available.
        """
        # Try Radient first if available
        if radient_client:
            try:
                return radient_client.search(query, max_results=max_results)
            except Exception as e:
                print(f"Radient search failed: {str(e)}. Trying other providers.")
                # Continue to other providers if Radient fails

        # Try SERP API next
        if serp_api_client:
            try:
                return serp_api_client.search(query, search_engine, max_results)
            except Exception as e:
                if not tavily_client:
                    raise e
                print(f"SERP API search failed: {str(e)}. Trying Tavily.")

        # Try Tavily last
        if tavily_client:
            return tavily_client.search(query, max_results=max_results)

        raise RuntimeError("No search API provider available")

    return search_web


def get_credential_tool(credential_manager: CredentialManager) -> Callable[..., SecretStr]:
    """Create a tool function to retrieve a credential as a SecretStr."""

    def get_credential(name: str) -> SecretStr:
        """Retrieve a credential by name as a SecretStr.  This tool is used to get the pydantic SecretStr for a credential, which should then be used in code that you write that requires the credential.  Never try to print the credential, and never send the credential anywhere other than the intended providers.

        Args:
            name (str): The credential name

        Returns:
            SecretStr: The secret credential value
        """  # noqa: E501
        secret_value = credential_manager.get_credential(name)
        return secret_value

    return get_credential


def list_credentials_tool(credential_manager: CredentialManager) -> Callable[..., List[str]]:
    """Create a tool function to list available credential names."""

    def list_credentials() -> List[str]:
        """List all available credential names.  Use this to check what the available credentials are if you are being asked to do something that might require a credential that has been added by the user through the configuration settings.  Never try to print the credential, and never send the credential anywhere other than the intended providers.

        Returns:
            List[str]: List of credential names
        """  # noqa: E501
        return credential_manager.list_credential_keys()

    return list_credentials


def _get_browser_path() -> Optional[str]:
    """Attempt to find the path to a supported browser executable.

    Searches for browsers in the following order: Arc, Chrome, Edge, Firefox, Opera, Brave.

    Returns:
        Optional[str]: The path to the first found browser executable, or None if not found.
    """
    system = platform.system()
    # Ordered list of browsers to check with their typical executable names or paths
    browsers_mac = [
        ("/Applications/Arc.app/Contents/MacOS/Arc", "Arc"),
        ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "Google Chrome"),
        ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "Microsoft Edge"),
        ("/Applications/Firefox.app/Contents/MacOS/firefox", "Firefox"),
        ("/Applications/Opera.app/Contents/MacOS/Opera", "Opera"),
        ("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser", "Brave Browser"),
    ]
    browsers_windows = [
        (
            os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                "The Browser Company\\Arc\\Application\\arc.exe",
            ),
            "Arc",
        ),
        (
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs\\Arc\\arc.exe"),
            "Arc",
        ),  # User install
        (
            os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                "Google\\Chrome\\Application\\chrome.exe",
            ),
            "Google Chrome",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                "Google\\Chrome\\Application\\chrome.exe",
            ),
            "Google Chrome",
        ),
        (
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""), "Google\\Chrome\\Application\\chrome.exe"
            ),
            "Google Chrome",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                "Microsoft\\Edge\\Application\\msedge.exe",
            ),
            "Microsoft Edge",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                "Microsoft\\Edge\\Application\\msedge.exe",
            ),
            "Microsoft Edge",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"), "Mozilla Firefox\\firefox.exe"
            ),
            "Mozilla Firefox",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                "Mozilla Firefox\\firefox.exe",
            ),
            "Mozilla Firefox",
        ),
        (
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs\\Opera\\launcher.exe"),
            "Opera",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"), "Opera\\launcher.exe"
            ),
            "Opera",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                "BraveSoftware\\Brave-Browser\\Application\\brave.exe",
            ),
            "Brave Browser",
        ),
        (
            os.path.join(
                os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                "BraveSoftware\\Brave-Browser\\Application\\brave.exe",
            ),
            "Brave Browser",
        ),
        (
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "BraveSoftware\\Brave-Browser\\Application\\brave.exe",
            ),
            "Brave Browser",
        ),
    ]
    browsers_linux = [
        ("arc", "Arc"),  # Placeholder, Arc not officially on Linux yet
        ("google-chrome", "Google Chrome"),
        ("google-chrome-stable", "Google Chrome Stable"),
        ("microsoft-edge", "Microsoft Edge"),
        ("microsoft-edge-stable", "Microsoft Edge Stable"),
        ("firefox", "Firefox"),
        ("opera", "Opera"),
        ("brave-browser", "Brave Browser"),
        ("chromium-browser", "Chromium"),  # Fallback
        ("chromium", "Chromium"),  # Fallback
    ]

    possible_paths_os: List[Tuple[str, str]] = []
    if system == "Darwin":
        possible_paths_os = browsers_mac
    elif system == "Windows":
        possible_paths_os = browsers_windows
    elif system == "Linux":
        possible_paths_os = browsers_linux

    for path_candidate, browser_name in possible_paths_os:
        # For Linux, shutil.which is better for finding executables in PATH
        if system == "Linux" and not os.path.isabs(path_candidate):
            found_path = shutil.which(path_candidate)
            if found_path:
                return found_path
        # For absolute paths or other OSes, check existence directly
        elif os.path.exists(path_candidate):
            return path_candidate
        # On macOS, also check user-specific application paths
        if system == "Darwin":
            user_path = os.path.expanduser(f"~/Applications/{Path(path_candidate).name}")
            if os.path.exists(user_path):
                return user_path
            # Arc specific user path on macOS
            if browser_name == "Arc":
                user_arc_path = os.path.expanduser("~/Applications/Arc.app/Contents/MacOS/Arc")
                if os.path.exists(user_arc_path):
                    return user_arc_path

    # Fallback for Linux if specific paths weren't found but command might be in PATH
    if system == "Linux":
        linux_fallbacks = [
            "google-chrome",
            "microsoft-edge",
            "firefox",
            "opera",
            "brave-browser",
            "chromium",
        ]
        for browser_cmd in linux_fallbacks:
            found_path = shutil.which(browser_cmd)
            if found_path:
                # print(f"Found {browser_cmd} (fallback) at {found_path}") # Optional: for debugging
                return found_path
    return None


DEFAULT_DEBUGGING_PORTS = [
    9222,
    9223,
    9224,
    9225,
    9229,
    9230,
]  # Common ports for Chrome, Edge, Brave, Opera, Arc etc.


def _is_port_open(host: str, port: int, timeout: float = 0.1) -> bool:
    """Quickly check if a port is open on a host."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _get_browser_connection_urls_from_port(
    host: str, port: int, timeout: float = 1.0
) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """
    Attempts to get WebSocket debugger URL and construct CDP URL from /json/version.
    Returns a tuple (wss_url, cdp_url).
    """
    wss_url: Optional[str] = None
    cdp_url: Optional[str] = f"http://{host}:{port}"  # CDP URL is typically the http endpoint

    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/json/version")
        response = conn.getresponse()
        if response.status == 200:
            data = response.read()
            version_info = json.loads(data.decode("utf-8"))
            if "webSocketDebuggerUrl" in version_info and version_info["webSocketDebuggerUrl"]:
                retrieved_wss_url = version_info["webSocketDebuggerUrl"]
                if retrieved_wss_url.startswith("ws://") or retrieved_wss_url.startswith("wss://"):
                    wss_url = retrieved_wss_url
            # If we got a 200, even without wss_url, the cdp_url might be valid
            conn.close()
            return wss_url, cdp_url
        conn.close()
    except (
        http.client.HTTPException,  # Includes IncompleteRead, etc.
        socket.error,  # Includes timeout, connection refused if _is_port_open was too optimistic
        json.JSONDecodeError,
        ConnectionRefusedError,  # Explicitly catch if connect fails here
        UnicodeDecodeError,
    ):
        # If HTTP request fails, cdp_url might still be considered invalid or unreliable
        return None, None  # Indicate failure to get reliable connection details
    return wss_url, cdp_url  # Returns (None, cdp_url) if wss_url not found but HTTP 200


def _scan_for_browser_connection_urls() -> Optional[Tuple[Optional[str], Optional[str]]]:
    """
    Scans common remote debugging ports for an active browser session.
    Returns a tuple (wss_url, cdp_url) if a connectable session is found.
    Prefers wss_url if available.
    """
    host = "127.0.0.1"
    for port in DEFAULT_DEBUGGING_PORTS:
        if _is_port_open(host, port, timeout=0.2):
            connection_urls = _get_browser_connection_urls_from_port(host, port, timeout=0.5)
            if connection_urls and (connection_urls[0] or connection_urls[1]):
                # Return if either wss_url or cdp_url is found
                return connection_urls
    return None


def run_browser_task_tool(model_config: ModelConfiguration) -> Callable[..., Any]:
    """Create a tool function to run a browser automation task.

    Args:
        model_config (ModelConfiguration): The configured model to use for the agent.

    Returns:
        Callable: An async function that runs a browser automation task.
    """

    async def run_browser_task(task: str) -> str:
        """
        Run a browser automation task using the browser-use library.
        This will open the user's browser and have a browser agent control it to carry out
        the task. It will then return the result as a string.
        Review console output to see if the task was done according to instructions.
        Ensure instructions to the agent are clear and specific, including any websites,
        commands, and context the agent will need.  HOW TO HANDLE ERRORS: The agent will attempt 3 times to launch or connect to a browser.  If it fails all 3 times, it will raise an error that you will need to help the user to handle.  Often it's related to an existing browser instance that doesn't have a debug port open, which will need to be closed and reopened when this method runs.  This happens on Arc, but can happen on Chrome and other browsers as well.

        Args:
            task (str): The task to be performed by the agent.

        Returns:
            str: The result of the browser automation task.

        Raises:
            RuntimeError: If Chrome executable is not found or model instance is not available.
            TypeError: If the configured model instance is not a supported LLM type.
        """  # noqa: E501
        if not model_config or not model_config.instance:
            raise RuntimeError(
                "ModelConfiguration or model instance is not available for "
                "the browser task tool."
            )

        if not model_config.api_key or not model_config.api_key.get_secret_value():
            raise RuntimeError("API key not set for the model configuration.")

        # Ensure the LLM instance is not a mock or noop, as BrowserAgent needs a functional LLM
        if isinstance(model_config.instance, (ChatMock, ChatNoop)):
            error_message = (
                "Browser task tool cannot use LLM client of type "
                f"{type(model_config.instance).__name__}. "
                "A functional LLM client (e.g., ChatOpenAI, ChatAnthropic) is required."
            )
            raise TypeError(error_message)

        connection_urls = await asyncio.to_thread(_scan_for_browser_connection_urls)
        existing_wss_url, existing_cdp_url = connection_urls if connection_urls else (None, None)

        # Common settings for BrowserConfig
        headless_setting = False  # Default to non-headless for visibility
        keep_alive_setting = True  # As per original code's intent

        browser_config: BrowserConfig
        if existing_cdp_url:
            print(f"Attempting to connect to existing browser session via CDP: {existing_cdp_url}")
            browser_config = BrowserConfig(
                cdp_url=existing_cdp_url,  # type: ignore - browser-use might not type this
                headless=headless_setting,
                keep_alive=keep_alive_setting,  # type: ignore - browser-use type issue
            )
        elif existing_wss_url:
            print(f"Attempting to connect to existing browser session via WSS: {existing_wss_url}")
            browser_config = BrowserConfig(
                wss_url=existing_wss_url,
                headless=headless_setting,
                keep_alive=keep_alive_setting,  # type: ignore - browser-use type issue
            )
        else:
            print("No existing browser session found or connectable, launching new instance.")
            browser_path = _get_browser_path()
            if not browser_path:
                error_msg = (
                    "No supported browser found (Arc, Chrome, Edge, Firefox, Opera, Brave) "
                    "for launching. Ensure one is installed and in PATH, or provide path manually."
                )
                raise RuntimeError(error_msg)
            browser_config = BrowserConfig(
                browser_binary_path=browser_path,
                headless=headless_setting,
                keep_alive=keep_alive_setting,  # type: ignore - browser-use type issue
            )

        browser_instance = Browser(config=browser_config)
        controller = BrowserController()

        # Need to set this due to a browser-use bug
        os.environ["OPENAI_API_KEY"] = model_config.api_key.get_secret_value()

        agent = BrowserAgent(
            task=task,
            llm=model_config.instance,
            browser=browser_instance,
            controller=controller,
        )

        try:
            result = await agent.run()
            return str(result)  # Ensure result is a string
        finally:
            await browser_instance.close()

    return run_browser_task


def schedule_task_tool(
    tool_registry: "ToolRegistry",
    agent_registry: AgentRegistry,
    scheduler_service: Optional[Any],
) -> Callable[..., str]:
    """Factory to create the schedule_task tool with AgentRegistry and SchedulerService
    dependency. This closure expects to be bound as a method of ToolRegistry."""

    def schedule_task(
        agent_id: str,
        prompt: str,
        interval: int,
        unit: str,
        start_time_utc: datetime,
        end_time_utc: Optional[datetime] = None,
        one_time: bool = False,
    ) -> str:
        """Schedule a new task for for you or another agent to run at a specified frequency.  The agent ID should be your ID from the agent identity in your system prompt, or the ID of the agent that you were asked to schedule the task for.  Provide a prompt which will be a note about what should be done on each trigger.  It should be written from the perspective of a user that is going to be asking the agent (you) to do the task in the future.  The agent (you or another agent) will receive this prompt on each trigger as if from the user.  Specify the interval as an integer with the unit as one of "minutes", "hours", or "days".  Always specify a start time for the schedule as a datetime object, it can either be right now or at a point in the future.  Optionally specify an end time for the schedule as a datetime object to stop the recurrence at a certain point. If no end time is provided, the schedule will run indefinitely, otherwise it will stop at the end time.  Specify one_time as True if the user has asked for a one time reminder, otherwise omit it or set it to False for a recurring task with or without start or end times.  If you use one_time, you MUST provide a start time.  The start_time will be the exact timestamp in UTC that the task will be scheduled to run, and the interval and unit will be ignored (you can provide 1 minute as a placeholder).

        If status_queue is present on the tool registry, send a message through the queue to request scheduling, otherwise call scheduler_service directly if available.

        Returns:
            str: Confirmation message with the new schedule ID.
        """  # noqa: E501
        # self is the ToolRegistry instance if this is bound as a method
        try:
            agent_uuid = UUID(agent_id)
            schedule_unit = ScheduleUnit(unit.lower())
        except ValueError as e:
            raise ValueError(f"Error: Invalid agent_id or unit: {str(e)}")

        if one_time and not start_time_utc and (not interval or not unit):
            raise ValueError(
                "Error: start_time_utc, or interval and unit is required when one_time is True."
            )

        # Load current agent state to get existing schedules
        try:
            agent_state = agent_registry.load_agent_state(agent_id)
            schedules = agent_state.schedules
        except KeyError:
            raise ValueError(f"Error: Agent with ID '{agent_id}' not found.")
        except Exception as e:
            raise ValueError(f"Error loading agent state: {str(e)}")

        try:
            new_schedule = Schedule(
                agent_id=agent_uuid,
                prompt=prompt,
                interval=interval,
                unit=schedule_unit,
                start_time_utc=start_time_utc,
                end_time_utc=end_time_utc,
                created_at=datetime.now(timezone.utc),
                is_active=True,
                one_time=one_time,
            )
        except ValueError as ve:  # Catch Pydantic validation errors
            raise ValueError(f"Error creating schedule: {str(ve)}")

        schedules.append(new_schedule)
        agent_state.schedules = schedules

        try:
            # Use status_queue if present on the ToolRegistry instance
            status_queue = tool_registry.status_queue

            if status_queue is not None:
                status_queue.put(("schedule_add", new_schedule))

                if tool_registry.tool_execution_callback:
                    tool_registry.tool_execution_callback("schedule_task", new_schedule)

            elif scheduler_service:
                agent_registry.save_agent_state(agent_id, agent_state)
                scheduler_service.add_or_update_job(new_schedule)
            else:
                raise ValueError("Error: SchedulerService not available.")

            return f"Task scheduled successfully with ID: {new_schedule.id}"
        except Exception as e:
            return f"Error saving or scheduling new task: {str(e)}"

    return schedule_task


def stop_schedule_tool(
    tool_registry: "ToolRegistry",
    agent_registry: AgentRegistry,
    scheduler_service: Optional[Any],
) -> Callable[..., str]:
    """Factory to create the stop_schedule tool with AgentRegistry and SchedulerService
    dependency. This closure expects to be bound as a method of ToolRegistry."""

    def stop_schedule(agent_id: str, schedule_id: str) -> str:
        """Stop an active_schedule for an agent.

        If status_queue is present on the tool registry, send a message through the queue
        to request schedule removal, otherwise call scheduler_service directly if available.

        Returns:
            str: Confirmation or error message.
        """
        try:
            target_schedule_id = UUID(schedule_id)
        except ValueError:
            raise ValueError("Error: Invalid schedule_id format.")

        try:
            agent_state = agent_registry.load_agent_state(agent_id)
            schedules = agent_state.schedules
        except KeyError:
            raise ValueError(f"Error: Agent with ID '{agent_id}' not found.")
        except Exception as e:
            raise ValueError(f"Error loading agent state: {str(e)}")

        schedule_found = False
        for sched in schedules:
            if sched.id == target_schedule_id:
                sched.is_active = False
                schedule_found = True
                break

        if not schedule_found:
            raise ValueError(
                f"Error: Schedule with ID '{schedule_id}' not found for agent '{agent_id}'."
            )

        agent_state.schedules = schedules

        try:
            status_queue = tool_registry.status_queue

            if status_queue is not None:
                status_queue.put(("schedule_remove", target_schedule_id))

                if tool_registry.tool_execution_callback:
                    tool_registry.tool_execution_callback("stop_schedule", target_schedule_id)

            elif scheduler_service:
                agent_registry.save_agent_state(agent_id, agent_state)
                scheduler_service.remove_job(target_schedule_id)
            else:
                raise ValueError("Error: SchedulerService not available.")

            return f"Schedule '{schedule_id}' stopped successfully."
        except Exception as e:
            raise ValueError(
                f"Error saving updated schedule list or removing from scheduler: {str(e)}"
            )

    return stop_schedule


def list_schedules_tool(agent_registry: AgentRegistry) -> Callable[..., List[Dict[str, Any]]]:
    """Factory to create the list_schedules tool with AgentRegistry dependency."""

    def list_schedules(agent_id: str) -> List[Dict[str, Any]]:
        """List all active schedules for a given agent.

        Args:
            agent_id (str): The ID of the agent whose schedules are to be listed.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an active schedule.
                                 Returns empty list if agent not found or no active schedules.
        """
        try:
            agent_state = agent_registry.load_agent_state(agent_id)
            active_schedules = [
                s.model_dump(mode="json") for s in agent_state.schedules if s.is_active
            ]
            return active_schedules
        except KeyError:
            raise ValueError(f"Error: Agent with ID '{agent_id}' not found.")
        except Exception as e:
            raise ValueError(f"Error listing schedules for agent {agent_id}: {str(e)}")

    return list_schedules


def send_email_to_user_tool(
    radient_client: RadientClient,
) -> Callable[..., RadientSendEmailResponseData]:
    """Create a tool function to send an email to the authenticated user via Radient API.

    Args:
        radient_client (RadientClient): The Radient API client to use.

    Returns:
        Callable: A function that sends an email with a subject and body.
    """

    def send_email_to_user(subject: str, body: str) -> RadientSendEmailResponseData:
        """Send an email to the authenticated user, it will go to email address associated with the current Radient API key. This tool uses the Radient API to send an email with the provided subject and body to the email address associated with the current Radient API key.  Make sure the body is neatly formatted with HTML formatting (<br /> for line breaks, and other tags for bold, italics, and other formatting) and easy to read, and contains all the information the user needs.  Make sure to add a signature to the email with your agent name so that the user knows that it is from you.

        Args:
            subject (str): The subject of the email. Must be between 1 and 255 characters.
            body (str): The body of the email. Can be HTML or plain text. Must be at least 1 character.

        Returns:
            RadientSendEmailResponseData: A response containing a confirmation message and optionally a message ID.

        Raises:
            RuntimeError: If the Radient client is not available or the request fails.
        """  # noqa: E501
        if not radient_client.api_key:
            raise RuntimeError("RADIENT_API_KEY is not configured. Cannot send email.")

        # Basic validation based on OpenAPI spec
        if not (1 <= len(subject) <= 255):
            raise ValueError("Subject must be between 1 and 255 characters.")
        if not (len(body) >= 1):
            raise ValueError("Body must be at least 1 character long.")

        return radient_client.send_email_to_self(subject=subject, body=body)

    return send_email_to_user


class ToolRegistry:
    """Registry for tools that can be used by agents.

    The ToolRegistry maintains a collection of callable tools that agents can access and execute.
    It provides methods to initialize with default tools, add custom tools, and retrieve
    tools by name.

    Attributes:
        tools (dict): Dictionary mapping tool names to their callable implementations
    """

    _tools: Dict[str, Callable[..., Any]]
    serp_api_client: SerpApiClient | None = None
    tavily_client: TavilyClient | None = None
    fal_client: FalClient | None = None
    radient_client: RadientClient | None = None
    credential_manager: CredentialManager | None = None
    model_configuration: Optional[ModelConfiguration] = None
    agent_registry: Optional[AgentRegistry] = None
    scheduler_service: Optional[Any] = None
    status_queue: Optional[Any] = None
    tool_execution_callback: Optional[Callable[[str, Any], None]] = None

    def __init__(self):
        """Initialize an empty tool registry."""
        super().__init__()
        object.__setattr__(self, "_tools", {})

    def set_tool_execution_callback(self, tool_execution_callback: Callable[[str, Any], None]):
        """Set the tool execution callback for the registry.  This is used to bridge the boundary
        between the job execution environment and the server environment.

        Args:
            tool_execution_callback (Callable[[str, Any], None]): The tool
            execution callback to set.
        """
        self.tool_execution_callback = tool_execution_callback

    def set_status_queue(self, status_queue: Any):
        """Set the status queue for broadcasting tool/scheduler updates.

        Args:
            status_queue (Any): The status queue to set.
        """
        self.status_queue = status_queue

    def set_serp_api_client(self, serp_api_client: SerpApiClient):
        """Set the SERP API client for the registry.

        Args:
            serp_api_client (SerpApiClient): The SERP API client to set
        """
        self.serp_api_client = serp_api_client

    def set_tavily_client(self, tavily_client: TavilyClient):
        """Set the Tavily API client for the registry.

        Args:
            tavily_client (TavilyClient): The Tavily API client to set
        """
        self.tavily_client = tavily_client

    def set_fal_client(self, fal_client: FalClient):
        """Set the FAL API client for the registry.

        Args:
            fal_client (FalClient): The FAL API client to set
        """
        self.fal_client = fal_client

    def set_radient_client(self, radient_client: RadientClient):
        """Set the Radient API client for the registry.

        Args:
            radient_client (RadientClient): The Radient API client to set
        """
        self.radient_client = radient_client

    def set_credential_manager(self, credential_manager: CredentialManager):
        """Set the credential manager for the registry.

        Args:
            credential_manager (CredentialManager): The credential manager to set
        """
        self.credential_manager = credential_manager

    def set_model_configuration(self, model_config: ModelConfiguration):
        """Set the ModelConfiguration for the registry.

        Args:
            model_config (ModelConfiguration): The ModelConfiguration to set.
        """
        self.model_configuration = model_config

    def set_agent_registry(self, agent_registry: AgentRegistry):
        """Set the AgentRegistry for the tool registry.

        Args:
            agent_registry (AgentRegistry): The AgentRegistry instance.
        """
        self.agent_registry = agent_registry

    def set_scheduler_service(self, scheduler_service: Any):
        """Set the SchedulerService for the tool registry.

        Args:
            scheduler_service (SchedulerService): The SchedulerService instance.
        """
        self.scheduler_service = scheduler_service

    def init_tools(self):
        """Initialize the registry with default tools.

        Default tools include:
        - get_page_html_content: Browse a URL and get page HTML content
        - get_page_text_content: Browse a URL and get page text content
        - list_working_directory: Index files in current directory
        - search_web: Search the web using available search APIs
        - generate_image: Generate images using available image generation APIs
        - generate_altered_image: Alter existing images using available image generation APIs
        - get_credential: Retrieve a secret credential
        - list_credentials: List available credentials
        - run_browser_task: Perform a task using an automated browser
        - schedule_task: Schedule a new task for an agent
        - stop_schedule: Stop an active schedule for an agent
        - list_schedules: List active schedules for an agent
        - send_email_to_user: Send an email to the authenticated user
        """
        self.add_tool("get_page_html_content", get_page_html_content)
        self.add_tool("get_page_text_content", get_page_text_content)
        self.add_tool("list_working_directory", list_working_directory)

        # Add search tool if any search client is available
        if self.radient_client or self.serp_api_client or self.tavily_client:
            self.add_tool(
                "search_web",
                search_web_tool(self.serp_api_client, self.tavily_client, self.radient_client),
            )

        # Add image generation tools if any image generation client is available
        if self.radient_client or self.fal_client:
            self.add_tool(
                "generate_image", generate_image_tool(self.fal_client, self.radient_client)
            )
            self.add_tool(
                "generate_altered_image",
                generate_altered_image_tool(self.fal_client, self.radient_client),
            )

        if self.credential_manager:
            self.add_tool("get_credential", get_credential_tool(self.credential_manager))
            self.add_tool("list_credentials", list_credentials_tool(self.credential_manager))

            has_google_access_token = bool(
                self.credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY).get_secret_value()
            )

            if has_google_access_token:
                # Add Gmail tools if credential manager is available
                self.add_tool(
                    "list_gmail_messages", list_gmail_messages_tool(self.credential_manager)
                )
                self.add_tool("get_gmail_message", get_gmail_message_tool(self.credential_manager))
                self.add_tool(
                    "create_gmail_draft", create_gmail_draft_tool(self.credential_manager)
                )
                self.add_tool(
                    "send_gmail_message", send_gmail_message_tool(self.credential_manager)
                )
                self.add_tool("send_gmail_draft", send_gmail_draft_tool(self.credential_manager))
                self.add_tool(
                    "update_gmail_draft", update_gmail_draft_tool(self.credential_manager)
                )
                self.add_tool(
                    "delete_gmail_draft", delete_gmail_draft_tool(self.credential_manager)
                )
                # Add Google Calendar tools
                self.add_tool(
                    "list_calendar_events", list_calendar_events_tool(self.credential_manager)
                )
                self.add_tool(
                    "create_calendar_event", create_calendar_event_tool(self.credential_manager)
                )
                self.add_tool(
                    "update_calendar_event", update_calendar_event_tool(self.credential_manager)
                )
                self.add_tool(
                    "delete_calendar_event", delete_calendar_event_tool(self.credential_manager)
                )
                # Add Google Drive tools
                self.add_tool("list_drive_files", list_drive_files_tool(self.credential_manager))
                self.add_tool(
                    "download_drive_file", download_drive_file_tool(self.credential_manager)
                )
                self.add_tool("upload_drive_file", upload_drive_file_tool(self.credential_manager))
                self.add_tool(
                    "update_drive_file_metadata",
                    update_drive_file_metadata_tool(self.credential_manager),
                )
                self.add_tool(
                    "update_drive_file_content",
                    update_drive_file_content_tool(self.credential_manager),
                )

        if self.model_configuration:  # Ensure model_configuration is set
            self.add_tool("run_browser_task", run_browser_task_tool(self.model_configuration))
        else:
            # Optionally log a warning or skip adding the tool if model_config is None
            print("Warning: ModelConfiguration not set, skipping run_browser_task tool.")

        if self.agent_registry:  # Ensure agent_registry is set
            self.add_tool(
                "schedule_task",
                schedule_task_tool(
                    self,
                    self.agent_registry,
                    self.scheduler_service,
                ),
            )
            self.add_tool(
                "stop_schedule",
                stop_schedule_tool(
                    self,
                    self.agent_registry,
                    self.scheduler_service,
                ),
            )
            self.add_tool("list_schedules", list_schedules_tool(self.agent_registry))
        else:
            print("Warning: AgentRegistry not set, skipping schedule tools.")

        # Add send_email_to_user tool if Radient client and API key are available
        if self.radient_client and self.radient_client.api_key:
            self.add_tool("send_email_to_user", send_email_to_user_tool(self.radient_client))

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
