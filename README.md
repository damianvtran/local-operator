# Local Operator

Local Operator is a Python-based agent that runs locally on your device, enabling secure execution of commands through a conversational chat interface. It provides a safe environment for running Python code while maintaining system security through built-in safety checks and user confirmation prompts.

This repository is open source and free to use, with an MIT license.  Feel free to incorporate it into your own projects as needed.  Though, we would love to hear your feedback and any contributions to the project will greatly help the community!

Artificial intelligence tools like these should be open and freely available to the majority of people due to the exponential impact that they have on personal productivity.  We hope to make this a reality for everyone!

## Key Features

- **Interactive CLI Interface**: Chat with an AI assistant that can execute Python code locally
- **Code Safety Verification**: Built-in safety checks analyze code for potentially dangerous operations
- **Contextual Execution**: Maintains execution context between code blocks
- **Conversation History**: Tracks the full interaction history for context-aware responses
- **DeepSeek Integration**: Uses DeepSeek's AI models through LangChain's ChatOpenAI implementation
- **Asynchronous Execution**: Safe code execution with async/await pattern
- **Environment Configuration**: Uses credential manager for API key management

The Local Operator provides a command-line interface where you can:
1. Interact with the AI assistant in natural language
2. Execute Python code blocks marked with ```python``` syntax
3. Get safety warnings before executing potentially dangerous operations
4. View execution results and error messages
5. Maintain context between code executions

## Setup

To run the operator CLI, you need to have a DeepSeek API key.  You can get one [here](https://platform.deepseek.com/api_keys).

Once you have the API key, install the operator CLI with the following command:

```bash
pip install local-operator
```

If you would like to run with a local Ollama model, you will need to install Ollama first from [here](https://ollama.ai/download), and fetch a model using `ollama pull`.  Make sure that the ollama server is running with `ollama serve`.

## Usage

Run the operator CLI with the following command:

#### Run with DeepSeek

```bash
local-operator --hosting deepseek --model deepseek-chat
```

#### Run with a local Ollama model

Download and install Ollama first from [here](https://ollama.ai/download).

```bash
local-operator --hosting ollama --model qwen2.5:14b
```

#### Run with OpenAI

```bash
local-operator --hosting openai --model gpt-4o
```

This will run the operator starting in the current working directory.  It will prompt you for any missing API keys or configuration on first run.  Everything else is handled by the agent 😊

Quit by typing `exit` or `quit`.

Run `local-operator --help` for more information about parameters and configuration.

## Examples

### Hello World

```shell
You: > make a new file called test.txt and write Hello World in it

╭─ Agent Response ────────────────────────────────
│ ```python
│ with open("test.txt", "w") as file:
│     file.write("Hello World")
│ ```
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks ─────────────────────────

│ Executing:
with open("test.txt", "w") as file:
    file.write("Hello World")

│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:

╰──────────────────────────────────────────────────
```

```shell
# test.txt
Hello World
```

### Creating a new git repository

```shell
You: > make a git repo in the folder ~/mypackage for a project called "mypackage".  This is a python package, so add the necessary boilerplate files.  The initial branch should be "main"

╭─ Agent Response ────────────────────────────────
│ ```python
│ import os
│ import subprocess
│ 
│ # Define the directory and package name
│ package_dir = os.path.expanduser("~/mypackage")
│ package_name = "mypackage"
│ 
│ # Create the directory if it doesn't exist
│ os.makedirs(package_dir, exist_ok=True)
│ 
│ # Initialize the git repository
│ subprocess.run(["git", "init", package_dir], check=True)
│ 
│ # Change to the package directory
│ os.chdir(package_dir)
│ 
│ # Create the necessary boilerplate files
│ with open("README.md", "w") as f:
│     f.write(f"# {package_name}\n\nThis is a Python package.")
│ 
│ with open("setup.py", "w") as f:
│     f.write(f"""from setuptools import setup, find_packages
│ 
│ setup(
│     name="{package_name}",
│     version="0.1",
│     packages=find_packages(),
│     install_requires=[],
│     author="Your Name",
│     author_email="your.email@example.com",
│     description="A short description of your package",
│     long_description=open("README.md").read(),
│     long_description_content_type="text/markdown",
│     url="https://github.com/yourusername/{package_name}",
│ )
│ """)
│ 
│ with open(f"{package_name}/__init__.py", "w") as f:
│     f.write("# Package initialization\n")
│ 
│ # Create a .gitignore file
│ with open(".gitignore", "w") as f:
│     f.write("""# Ignore Python compiled files
│ __pycache__/
│ *.pyc
│ *.pyo
│ *.pyd
│ 
│ # Ignore virtual environments
│ venv/
│ env/
│ 
│ # Ignore distribution files
│ dist/
│ build/
│ *.egg-info/
│ """)
│ 
│ # Add all files to git
│ subprocess.run(["git", "add", "."], check=True)
│ 
│ # Commit the initial files
│ subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
│ 
│ # Rename the default branch to 'main'
│ subprocess.run(["git", "branch", "-M", "main"], check=True)
│ 
│ print(f"Git repository initialized in {package_dir} with branch 'main'.")
│ ```
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks ─────────────────────────

│ Executing:
import os
import subprocess

# Define the directory and package name
package_dir = os.path.expanduser("~/mypackage")
package_name = "mypackage"

# Create the directory if it doesn't exist
os.makedirs(package_dir, exist_ok=True)

# Initialize the git repository
subprocess.run(["git", "init", package_dir], check=True)

# Change to the package directory
os.chdir(package_dir)

# Create the necessary boilerplate files
with open("README.md", "w") as f:
    f.write(f"# {package_name}\n\nThis is a Python package.")

with open("setup.py", "w") as f:
    f.write(f"""from setuptools import setup, find_packages

setup(
    name="{package_name}",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/{package_name}",
)
""")

with open(f"{package_name}/__init__.py", "w") as f:
    f.write("# Package initialization\n")

# Create a .gitignore file
with open(".gitignore", "w") as f:
    f.write("""# Ignore Python compiled files
__pycache__/
*.pyc
*.pyo
*.pyd

# Ignore virtual environments
venv/
env/

# Ignore distribution files
dist/
build/
*.egg-info/
""")

# Add all files to git
subprocess.run(["git", "add", "."], check=True)

# Commit the initial files
subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)

# Rename the default branch to 'main'
subprocess.run(["git", "branch", "-M", "main"], check=True)

print(f"Git repository initialized in {package_dir} with branch 'main'.")

Reinitialized existing Git repository in /Users/damiantran/mypackage/.git/
Reinitialized existing Git repository in /Users/damiantran/mypackage/.git/
[main (root-commit) 03d2a18] Initial commit
 4 files changed, 32 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 create mode 100644 mypackage/__init__.py
 create mode 100644 setup.py
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Git repository initialized in /Users/damiantran/mypackage with branch 'main'.

╰──────────────────────────────────────────────────
```

### Network Requests

```shell
You: > make a get request to https://google.com

╭─ Agent Response ────────────────────────────────
│ ```python
│ import requests
│ 
│ response = requests.get("https://google.com")
│ print(f"Status Code: {response.status_code}")
│ print(f"Response Content:\n{response.text[:500]}...")  # Print first 500 characters of the response
│ ```
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks ─────────────────────────

│ Executing:
import requests

response = requests.get("https://google.com")
print(f"Status Code: {response.status_code}")
print(f"Response Content:\n{response.text[:500]}...")  # Print first 500 characters of the response

│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Status Code: 200
Response Content:
<!doctype html><html itemscope="" itemtype="http://schema.org/WebPage" lang="en-CA"><head><meta content="text/html; charset=UTF-8" http-equiv="Content-Type"><meta content="/images/branding/googleg/1x/googleg_standard_color_128dp.png" itemprop="image"><title>Google</title><script nonce="LWl9h5SwfBVE3Apb6sypQw">(function(){var _g={kEI:'nXOWZ9unCMS-p84Pr92lwQ8',kEXPI:'0,3700317,1067,507102,31559,2872,2891,73050,16105,78219,266577,45786,9779,68707,87779,7734,39348,11342,19569,5240757,764,48,8834865,...

╰──────────────────────────────────────────────────
```

### Installing missing dependencies

```shell
You (/Users/damiantran/local-operator): > make a mock csv that has a datetime column and a person name column, with 10 rows.  Use pandas and numpy

╭─ Agent Response (Step 1) ───────────────────────
│ ```python
│ import pandas as pd
│ import numpy as np
│ from datetime import datetime, timedelta
│ 
│ # Generate mock data with datetime and person name columns
│ start_date = datetime.now()
│ names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack']
│ 
│ data = {
│     'datetime': [start_date + timedelta(days=i) for i in range(10)],
│     'person_name': np.random.choice(names, 10)
│ }
│ 
│ # Create DataFrame
│ df = pd.DataFrame(data)
│ 
│ # Save to CSV
│ df.to_csv('mock_data_with_datetime.csv', index=False)
│ print("mock_data_with_datetime.csv has been created with mock data.")
│ ```
│ 
│ DONE
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 1) ───────────────

│ Executing:
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate mock data with datetime and person name columns
start_date = datetime.now()
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack']

data = {
    'datetime': [start_date + timedelta(days=i) for i in range(10)],
    'person_name': np.random.choice(names, 10)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('mock_data_with_datetime.csv', index=False)
print("mock_data_with_datetime.csv has been created with mock data.")

Collecting pandas
  Downloading pandas-2.2.3-cp313-cp313-macosx_11_0_arm64.whl.metadata (89 kB)
Requirement already satisfied: numpy in ./.venv/lib/python3.13/site-packages (2.2.2)
Collecting python-dateutil>=2.8.2 (from pandas)
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas)
  Downloading pytz-2024.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas)
  Downloading tzdata-2025.1-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas)
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Downloading pandas-2.2.3-cp313-cp313-macosx_11_0_arm64.whl (11.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.3/11.3 MB 11.4 MB/s eta 0:00:00
Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading pytz-2024.2-py2.py3-none-any.whl (508 kB)
Downloading tzdata-2025.1-py2.py3-none-any.whl (346 kB)
Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Installing collected packages: pytz, tzdata, six, python-dateutil, pandas
Successfully installed pandas-2.2.3 python-dateutil-2.9.0.post0 pytz-2024.2 six-1.17.0 tzdata-2025.1
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
mock_data_with_datetime.csv has been created with mock data.

╰──────────────────────────────────────────────────
```

### Performing unsafe operations

```shell
You: > remove the ~/mypackage folder 

╭─ Agent Response ────────────────────────────────
│ ```python
│ import shutil
│ import os
│ 
│ folder_path = os.path.expanduser('~/mypackage')
│ if os.path.exists(folder_path):
│     shutil.rmtree(folder_path)
│     print(f"Folder '{folder_path}' removed successfully.")
│ else:
│     print(f"Folder '{folder_path}' does not exist.")
│ ```
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks ─────────────────────────

│ Executing:
import shutil
import os

folder_path = os.path.expanduser('~/mypackage')
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    print(f"Folder '{folder_path}' removed successfully.")
else:
    print(f"Folder '{folder_path}' does not exist.")

Warning: Potentially dangerous operation detected. Proceed? (y/n): y
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Folder '/Users/damiantran/mypackage' removed successfully.

╰──────────────────────────────────────────────────
```

## Safety Features

The system includes multiple layers of protection:
- Automatic detection of dangerous operations (file access, system commands, etc.)
- User confirmation prompts for potentially unsafe code
- Agent prompt with safety focused execution policy
- Support for local Ollama models to prevent sending local system data to 3rd parties

## Requirements

- Python 3.12+
- For 3rd party hosting: DeepSeek API key or OpenAI API key (prompted for on first run)
- For local hosting: Ollama model installed and running

## License

This project is licensed under the MIT License - see the LICENSE file for details.
