<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/local-operator-icon-2-dark-clear.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/local-operator-icon-2-light-clear.png">
  <img alt="Shows a black Local Operator Logo in light color mode and a white one in dark color mode." src="./static/local-operator-icon-2-light-clear.png"  width="full">
</picture>

<h1 align="center">Local Operator: On-device Agentic Task Execution</h1>

**Local Operator** is a Python-based agent that runs locally on your device, enabling secure execution of commands through a conversational chat interface. The agent can plan a series of steps to achieve a user goal, and execute the code sequentially with self-correction to achieve the goal.  When run with local models, it provides a safe environment for running Python code while maintaining system security through built-in safety checks and user confirmation prompts.  It is also compatible with 3rd party cloud-hosted models such as those from OpenAI and DeepSeek.  

This repository is open source and free to use, with an MIT license.  Feel free to incorporate it into your own projects as needed.  Though, we would love to hear your feedback and any contributions to the project will greatly help the community!

Artificial intelligence tools like these should be open and freely available to the majority of people due to the exponential impact that they have on personal productivity.  We hope to make this a reality for everyone!

## Key Features

- **Interactive CLI Interface**: Chat with an AI assistant that can execute Python code locally
- **Code Safety Verification**: Built-in safety checks analyze code for potentially dangerous operations
- **Contextual Execution**: Maintains execution context between code blocks
- **Conversation History**: Tracks the full interaction history for context-aware responses
- **Local Model Support**: Supports closed-circuit on-device execution with Ollama.
- **LangChain Integration**: Uses 3rd party cloud-hosted LLM models through LangChain's ChatOpenAI implementation
- **Asynchronous Execution**: Safe code execution with async/await pattern
- **Environment Configuration**: Uses credential manager for API key management

The Local Operator provides a command-line interface where you can:

1. Interact with the AI assistant in natural language
2. Execute Python code blocks marked with ```python``` syntax
3. Get safety warnings before executing potentially dangerous operations
4. View execution results and error messages
5. Maintain context between code executions

## Setup

To run the operator CLI with a 3rd party cloud-hosted LLM model, you need to have an API key.  You can get one from OpenAI, DeepSeek, Anthropic, or other providers.

Once you have the API key, install the operator CLI with the following command:

```bash
pip install local-operator
```

If you would like to run with a local Ollama model, you will need to install Ollama first from [here](https://ollama.ai/download), and fetch a model using `ollama pull`.  Make sure that the ollama server is running with `ollama serve`.

## Usage

Run the operator CLI with the following command:

### Run with a local Ollama model

Download and install Ollama first from [here](https://ollama.ai/download).

```bash
local-operator --hosting ollama --model qwen2.5:14b
```

### Run with DeepSeek

```bash
local-operator --hosting deepseek --model deepseek-chat
```

### Run with OpenAI

```bash
local-operator --hosting openai --model gpt-4o
```

This will run the operator starting in the current working directory.  It will prompt you for any missing API keys or configuration on first run.  Everything else is handled by the agent 😊

Quit by typing `exit` or `quit`.

Run `local-operator --help` for more information about parameters and configuration.

## Examples

### Hello World

<details>
<summary>Simple file creation with generated content</summary>
<br />

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

</details>

### Creating a new git repository

<details>
<summary>Creating a new git repository with some boilerplate requirements</summary>
<br />

The agent runs the generated code in a first step which fails due to a sub-folder not being created.  In the second step, the agent creates the sub-folder and then runs the code again to acheive the user goal.

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

</details>

### Network Requests

<details>
<summary>Simple get request with data fetch</summary>
<br />

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

</details>

### Installing missing dependencies

<details>
<summary>Missing dependency discovery through trial and error</summary>
<br />

The agent fails on the first attempt and then performs the necessary installation on the second attempt.

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

</details>

### Performing unsafe operations

<details>
<summary>Removing a folder</summary>
<br />

The agent asks for confirmation before performing the operation.

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

</details>

### Multi-step Agent Execution

<details>
<summary>Simple two-step Hello World</summary>
<br />

```shell
You (/Users/damiantran/local-operator): > do a two step execution, add more text and then read the text in two steps

╭─ Agent Response (Step 1) ───────────────────────
│ I'll break this into two steps as requested:
│ 
│ **Step 1: Add more text to the file**
│ ```python
│ additional_text = """\n\nThis is additional text added in step 1.
│ It will be followed by the file content reading in step 2."""
│ 
│ with open('test.txt', 'a') as file:
│     file.write(additional_text)
│ 
│ print("Additional text added to 'test.txt' successfully.")
│ ```
│ 
│ After you confirm this step, I'll proceed with step 2 to read the file contents.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 1) ───────────────

│ Executing:
additional_text = """\n\nThis is additional text added in step 1.
It will be followed by the file content reading in step 2."""

with open('test.txt', 'a') as file:
    file.write(additional_text)

print("Additional text added to 'test.txt' successfully.")

│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Additional text added to 'test.txt' successfully.

╰──────────────────────────────────────────────────

╭─ Agent Response (Step 2) ───────────────────────
│ Code execution output:
│ Additional text added to 'test.txt' successfully.
│ 
│ 
│ Current working directory: /Users/damiantran/local-operator**Step 2: Read the file contents**
│ ```python
│ with open('test.txt', 'r') as file:
│     content = file.read()
│ 
│ print("File contents:\n")
│ print(content)
│ ```
│ 
│ This will read and display the complete contents of `test.txt`, including the original poem, Lorem Ipsum text, and the additional text we just added.
│ 
│ DONE
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 2) ───────────────

│ Executing:
with open('test.txt', 'r') as file:
    content = file.read()

print("File contents:\n")
print(content)

│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
File contents:

Roses are red,
Violets are blue,
Sugar is sweet,
And so are you!

Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

This is additional text added in step 1.
It will be followed by the file content reading in step 2.

╰──────────────────────────────────────────────────
```

</details>

<details>
<summary>Iterative improvement loop</summary>
<br />

```shell
You (/Users/damiantran/local-operator): > create a mock data csv and then read it and think about improvements.  Enact the improvements and make it better

╭─ Agent Response (Step 1) ───────────────────────
│ Let's break this task into logical steps:
│ 
│ 1. Create mock data CSV
│ 2. Read and analyze the data
│ 3. Identify potential improvements
│ 4. Implement improvements
│ 5. Save improved version
│ 
│ Let's start with step 1 - creating mock data. I'll create a CSV with some basic customer data.
│ 
│ ```python
│ import csv
│ import random
│ from datetime import datetime, timedelta
│ 
│ # Generate mock data
│ data = []
│ for i in range(100):
│     customer_id = f"CUST{i+1:03}"
│     signup_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
│     purchases = random.randint(0, 50)
│     total_spent = round(random.uniform(0, 5000), 2)
│     last_purchase = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d')
│     data.append([customer_id, signup_date, purchases, total_spent, last_purchase])
│ 
│ # Write to CSV
│ with open('mock_customers.csv', 'w', newline='') as file:
│     writer = csv.writer(file)
│     writer.writerow(['customer_id', 'signup_date', 'purchases', 'total_spent', 'last_purchase'])
│     writer.writerows(data)
│ 
│ print("Mock data CSV created: mock_customers.csv")
│ ```
│ 
│ Let me know when you want to proceed to step 2 (reading and analyzing the data).
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 1) ───────────────

│ Executing:
import csv
import random
from datetime import datetime, timedelta

# Generate mock data
data = []
for i in range(100):
    customer_id = f"CUST{i+1:03}"
    signup_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
    purchases = random.randint(0, 50)
    total_spent = round(random.uniform(0, 5000), 2)
    last_purchase = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d')
    data.append([customer_id, signup_date, purchases, total_spent, last_purchase])

# Write to CSV
with open('mock_customers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['customer_id', 'signup_date', 'purchases', 'total_spent', 'last_purchase'])
    writer.writerows(data)

print("Mock data CSV created: mock_customers.csv")

│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Mock data CSV created: mock_customers.csv

╰──────────────────────────────────────────────────

╭─ Agent Response (Step 2) ───────────────────────
│ Mock data CSV created: mock_customers.csv
│ 
│ Now let's proceed to step 2 - reading and analyzing the data to identify potential improvements.
│ 
│ ```python
│ import pandas as pd
│ 
│ # Read the CSV
│ df = pd.read_csv('mock_customers.csv')
│ 
│ # Basic analysis
│ print("First 5 rows:")
│ print(df.head())
│ print("\nData types:")
│ print(df.dtypes)
│ print("\nSummary statistics:")
│ print(df.describe())
│ print("\nMissing values:")
│ print(df.isnull().sum())
│ 
│ # Identify potential improvements
│ improvements = [
│     "1. Convert dates to datetime format",
│     "2. Add calculated columns (e.g., average purchase value)",
│     "3. Add customer segmentation based on spending",
│     "4. Add email addresses for more realistic data",
│     "5. Add location data",
│     "6. Add product categories purchased",
│     "7. Add customer status (active/inactive)"
│ ]
│ 
│ print("\nPotential improvements:")
│ for improvement in improvements:
│     print(f"- {improvement}")
│ ```
│ 
│ Let me know if you want to proceed with implementing any of these improvements or if you have other suggestions.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 2) ───────────────

│ Executing:
import pandas as pd

# Read the CSV
df = pd.read_csv('mock_customers.csv')

# Basic analysis
print("First 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Identify potential improvements
improvements = [
    "1. Convert dates to datetime format",
    "2. Add calculated columns (e.g., average purchase value)",
    "3. Add customer segmentation based on spending",
    "4. Add email addresses for more realistic data",
    "5. Add location data",
    "6. Add product categories purchased",
    "7. Add customer status (active/inactive)"
]

print("\nPotential improvements:")
for improvement in improvements:
    print(f"- {improvement}")

│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
First 5 rows:
  customer_id signup_date  purchases  total_spent last_purchase
0     CUST001  2024-09-16         40       971.22    2024-12-09
1     CUST002  2024-04-19         32      2537.07    2024-10-29
2     CUST003  2024-11-22          2      1120.01    2025-01-21
3     CUST004  2024-12-04         11       875.93    2024-12-13
4     CUST005  2024-11-19         13      2601.22    2024-12-29

Data types:
customer_id       object
signup_date       object
purchases          int64
total_spent      float64
last_purchase     object
dtype: object

Summary statistics:
        purchases  total_spent
count  100.000000   100.000000
mean    25.000000  2643.808800
std     14.784649  1483.624621
min      0.000000   112.700000
25%     13.000000  1279.070000
50%     26.500000  2630.115000
75%     37.000000  3999.202500
max     49.000000  4985.270000

Missing values:
customer_id      0
signup_date      0
purchases        0
total_spent      0
last_purchase    0
dtype: int64

Potential improvements:
- 1. Convert dates to datetime format
- 2. Add calculated columns (e.g., average purchase value)
- 3. Add customer segmentation based on spending
- 4. Add email addresses for more realistic data
- 5. Add location data
- 6. Add product categories purchased
- 7. Add customer status (active/inactive)

╰──────────────────────────────────────────────────

╭─ Agent Response (Step 3) ───────────────────────
│ Let's proceed with implementing the improvements. I'll start with the first few improvements:
│ 
│ 1. Convert dates to datetime format
│ 2. Add calculated columns (average purchase value)
│ 3. Add customer segmentation based on spending
│ 4. Add email addresses
│ 
│ ```python
│ import pandas as pd
│ import numpy as np
│ 
│ # Read the CSV
│ df = pd.read_csv('mock_customers.csv')
│ 
│ # Improvement 1: Convert dates to datetime
│ df['signup_date'] = pd.to_datetime(df['signup_date'])
│ df['last_purchase'] = pd.to_datetime(df['last_purchase'])
│ 
│ # Improvement 2: Add calculated columns
│ df['avg_purchase_value'] = np.where(df['purchases'] > 0, 
│                                    df['total_spent'] / df['purchases'], 
│                                    0)
│ 
│ # Improvement 3: Add customer segmentation
│ def create_segment(row):
│     if row['total_spent'] > 4000:
│         return 'VIP'
│     elif row['total_spent'] > 2000:
│         return 'Loyal'
│     elif row['total_spent'] > 1000:
│         return 'Regular'
│     else:
│         return 'New'
│ 
│ df['segment'] = df.apply(create_segment, axis=1)
│ 
│ # Improvement 4: Add email addresses
│ df['email'] = df['customer_id'].str.lower() + '@example.com'
│ 
│ # Save improved version
│ df.to_csv('improved_customers.csv', index=False)
│ 
│ # Show first 5 rows of improved data
│ print("Improved data preview:")
│ print(df.head())
│ print("\nNew columns added:")
│ print(df.columns)
│ print("\nSegment distribution:")
│ print(df['segment'].value_counts())
│ ```
│ 
│ The improved dataset now includes:
│ - Proper datetime formatting
│ - Average purchase value calculation
│ - Customer segmentation (VIP, Loyal, Regular, New)
│ - Email addresses
│ - Saved as 'improved_customers.csv'
│ 
│ DONE
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 3) ───────────────

│ Executing:
import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('mock_customers.csv')

# Improvement 1: Convert dates to datetime
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['last_purchase'] = pd.to_datetime(df['last_purchase'])

# Improvement 2: Add calculated columns
df['avg_purchase_value'] = np.where(df['purchases'] > 0, 
                                   df['total_spent'] / df['purchases'], 
                                   0)

# Improvement 3: Add customer segmentation
def create_segment(row):
    if row['total_spent'] > 4000:
        return 'VIP'
    elif row['total_spent'] > 2000:
        return 'Loyal'
    elif row['total_spent'] > 1000:
        return 'Regular'
    else:
        return 'New'

df['segment'] = df.apply(create_segment, axis=1)

# Improvement 4: Add email addresses
df['email'] = df['customer_id'].str.lower() + '@example.com'

# Save improved version
df.to_csv('improved_customers.csv', index=False)

# Show first 5 rows of improved data
print("Improved data preview:")
print(df.head())
print("\nNew columns added:")
print(df.columns)
print("\nSegment distribution:")
print(df['segment'].value_counts())

│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Improved data preview:
  customer_id signup_date  purchases  total_spent last_purchase  avg_purchase_value  segment                email
0     CUST001  2024-09-16         40       971.22    2024-12-09           24.280500      New  cust001@example.com
1     CUST002  2024-04-19         32      2537.07    2024-10-29           79.283438    Loyal  cust002@example.com
2     CUST003  2024-11-22          2      1120.01    2025-01-21          560.005000  Regular  cust003@example.com
3     CUST004  2024-12-04         11       875.93    2024-12-13           79.630000      New  cust004@example.com
4     CUST005  2024-11-19         13      2601.22    2024-12-29          200.093846    Loyal  cust005@example.com

New columns added:
Index(['customer_id', 'signup_date', 'purchases', 'total_spent',
       'last_purchase', 'avg_purchase_value', 'segment', 'email'],
      dtype='object')

Segment distribution:
segment
Loyal      36
VIP        25
Regular    22
New        17
Name: count, dtype: int64

╰──────────────────────────────────────────────────
```

</details>

<details>
<summary>Automatic web scraper creation</summary>

```shell
You (/Users/damiantran/local-operator-sandbox): > Scrape the web page https://webscraper.io/test-sites/tables into a csv.  Get the site structure first and then based on that make a suitable scraper

╭─ Agent Response (Step 1) ───────────────────────
│ To understand the structure of the webpage and create a suitable scraper, we can start by printing out a portion of the HTML content. This will help us identify the relevant elements and their structure.
│
│ Let's fetch the webpage and print a portion of its HTML content for inspection:
│
│ ```python
│ import requests
│ from bs4 import BeautifulSoup
│
│ # URL of the test site with tables
│ url = "https://webscraper.io/test-sites/tables"
│
│ # Fetch the webpage
│ response = requests.get(url)
│ soup = BeautifulSoup(response.content, 'html.parser')
│
│ # Print a portion of the HTML content for inspection
│ print(soup.prettify()[:2000])  # Print the first 2000 characters of the HTML content
│```
│
│ Executing this code will help us inspect the structure of the webpage. Let's proceed with this step.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 1) ───────────────

│ Executing:
import requests
from bs4 import BeautifulSoup

# URL of the test site with tables

url = "<https://webscraper.io/test-sites/tables>"

# Fetch the webpage

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Print a portion of the HTML content for inspection

print(soup.prettify()[:2000])  # Print the first 2000 characters of the HTML content
│ Result:
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
<!DOCTYPE html>
<html lang="en">
 <head>
  <!-- Google Tag Manager -->
  <script nonce="BUHLfj9g94X6UEYzdoIToq91rpTAGjAl">
   (function (w, d, s, l, i) {
  w[l] = w[l] || [];
    // ... Shortened for the example

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 2) ───────────────────────
│ The HTML content provides some initial structure, but we need to look further into the specific table elements to identify where the names are located. Let's focus on extracting and inspecting the table elements specifically.
│
│ I'll adjust the code to print out the table structure so we can determine the correct approach for extracting the names:
│
│ ```python
│ # Extract and print the table structure
│ tables = soup.find_all('table')
│ for idx, table in enumerate(tables):
│     print(f"Table {idx + 1}:")
│     print(table.prettify()[:1000])  # Print the first 1000 characters of each table
│     print("\n" + "="*40 + "\n")
│```
│
│ This code will print the structure of each table on the webpage, allowing us to identify the relevant data. Let's execute this step.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 2) ───────────────

│ Executing:

# Extract and print the table structure

tables = soup.find_all('table')
for idx, table in enumerate(tables):
    print(f"Table {idx + 1}:")
    print(table.prettify()[:1000])  # Print the first 1000 characters of each table
    print("\n" + "="*40 + "\n")
│ Result:
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Table 1:
<table class="table table-bordered">
 <thead>
  <tr>
   <th>
    #
   </th>
   <th>
    First Name
   </th>
   <th>
    Last Name
   </th>
   <th>
    Username
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
    1
   </td>
   <td>
    Mark
   </td>
   <td>
    Otto
   </td>
   <td>
    @mdo
   </td>
  </tr>
  // ... Shortened for the example

========================================

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 3) ───────────────────────
│ The webpage contains two tables, each with columns for `First Name`, `Last Name`, and `Username`. We can extract the `First Name` and `Last Name` from both tables and save them to a CSV file.
│
│ Let's proceed with extracting the names and writing them to a CSV file:
│
│ ```python
│ import csv
│
│ # Extract names from both tables
│ names = []
│ for table in tables:
│     for row in table.find_all('tr')[1:]:  # Skip the header row
│         columns = row.find_all('td')
│         if columns:
│             first_name = columns[1].get_text(strip=True)
│             last_name = columns[2].get_text(strip=True)
│             names.append({'First Name': first_name, 'Last Name': last_name})
│
│ # Write names to CSV
│ csv_file_path = 'names_list.csv'
│ with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
│     fieldnames = ['First Name', 'Last Name']
│     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
│
│     writer.writeheader()
│     for name in names:
│         writer.writerow(name)
│
│ # Read and print the contents of the CSV file
│ with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
│     content = csvfile.read()
│     print(content)
│```
│
│ This code will extract the names from both tables and save them to a CSV file named `names_list.csv`. Let's execute this step.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 3) ───────────────

│ Executing:
import csv

# Extract names from both tables

names = []
for table in tables:
    for row in table.find_all['tr'](1:):  # Skip the header row
        columns = row.find_all('td')
        if columns:
            first_name = columns[1].get_text(strip=True)
            last_name = columns[2].get_text(strip=True)
            names.append({'First Name': first_name, 'Last Name': last_name})

# Write names to CSV

csv_file_path = 'names_list.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['First Name', 'Last Name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for name in names:
        writer.writerow(name)

# Read and print the contents of the CSV file

with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    content = csvfile.read()
    print(content)
│ Result:
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
First Name,Last Name
Mark,Otto
Jacob,Thornton
Larry,the Bird
Harry,Potter
John,Snow
Tim,Bean

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 4) ───────────────────────
│ The names have been successfully extracted from the tables and saved to the CSV file `names_list.csv`. The contents of the file are as follows:
│
│ ```
│ First Name,Last Name
│ Mark,Otto
│ Jacob,Thornton
│ Larry,the Bird
│ Harry,Potter
│ John,Snow
│ Tim,Bean
│ ```
│
│ This CSV file contains the first and last names from both tables on the webpage.
│
│ DONE
╰──────────────────────────────────────────────────

```

</details>

<details>
<summary>Complex web scraping example</summary>

```shell
╭──────────────────────────────────────────────────╮
│ Local Executor Agent CLI                         │
│──────────────────────────────────────────────────│
│ You are interacting with a helpful CLI agent     │
│ that can execute tasks locally on your device    │
│ by running Python code.                          │
│──────────────────────────────────────────────────│
│ Type 'exit' or 'quit' to quit                    │
╰──────────────────────────────────────────────────╯

You (/Users/damiantran/python-sandbox): > get the full list of sanctioned individuals from https://www.international.gc.ca/world-monde/international_relations-relations_internationales/sanctions/consolidated-consolide.aspx?lang=eng and save them to a csv.  Review the site structure first to determine how to scrape and then make sure to get every single name from the site from beginning to end

╭─ Agent Response (Step 1) ───────────────────────
│ To accomplish this task, I will first review the structure of the webpage to understand how the data is organized. This will help determine the best approach to scrape the list of sanctioned individuals. Let's start by fetching the HTML content of the page.
│ 
│ ```python
│ import requests
│ 
│ # Fetch the HTML content of the page
│ url = "https://www.international.gc.ca/world-monde/international_relations-relations_internationales/sanctions/consolidated-consolide.aspx?lang=eng"
│ response = requests.get(url)
│ 
│ # Print the first 500 characters of the HTML content to understand the structure
│ print(response.text[:500])
│ ```
│ 
│ This code will help us understand the structure of the webpage and identify where the list of sanctioned individuals is located. Once we have this information, we can proceed with the next steps to extract the data.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 1) ───────────────

│ Executing:
import requests

# Fetch the HTML content of the page
url = "https://www.international.gc.ca/world-monde/international_relations-relations_internationales/sanctions/consolidated-consolide.aspx?lang=eng"
response = requests.get(url)

# Print the first 500 characters of the HTML content to understand the structure
print(response.text[:500])
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
<!DOCTYPE html>  <html class="no-js" lang="en" dir="ltr">    <head>      <meta charset="utf-8">      <!-- IW --><!--<iw_root><iw><iw_components><iw_component type="Page" display_name_eng="Page" display_name_fra="Page" vpath="//server/dfait-maeci/main/internet-internet/international/world-monde/WORKAREA/content-contenu/templatedata/pages-pages/generic-generique/data/international_relations-relations_internationales/sanctions/consolidated-consolide"/></iw_components></iw></iw_root>-->      <!-- We

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 2) ───────────────────────
│ The initial HTML snippet indicates that the page uses a templated structure, but it doesn't provide enough information about where the list of sanctioned individuals is located. To proceed, I'll fetch a larger portion of the HTML content to better understand the structure and identify the relevant section.
│ 
│ ```python
│ # Print the first 2000 characters of the HTML content to get more context
│ print(response.text[:2000])
│ ```
│ 
│ This will give us a more comprehensive view of the HTML structure and help us locate the section containing the list of sanctioned individuals.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 2) ───────────────

│ Executing:
# Print the first 2000 characters of the HTML content to get more context
print(response.text[:2000])
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
<!DOCTYPE html>  <html class="no-js" lang="en" dir="ltr">    <head>      <meta charset="utf-8">      <!-- IW --><!--<iw_root><iw><iw_components><iw_component type="Page" display_name_eng="Page" display_name_fra="Page" vpath="//server/dfait-maeci/main/internet-internet/international/world-monde/WORKAREA/content-contenu/templatedata/pages-pages/generic-generique/data/international_relations-relations_internationales/sanctions/consolidated-consolide"/></iw_components></iw></iw_root>-->      <!-- Web Experience Toolkit (WET) / Boîte à outils de l'expérience Web (BOEW)        wet-boew.github.io/wet-boew/License-en.html / wet-boew.github.io/wet-boew/Licence-fr.html -->      <!-- WET v4.0.70.1 -->      <title>Consolidated Canadian Autonomous Sanctions List</title>      <!-- Meta data begins -->      <meta content="width=device-width,initial-scale=1" name="viewport">      <meta name="description" content="The Consolidated Canadian Autonomous Sanctions List includes individuals and entities subject to specific sanctions regulations made under the Special Economic Measures Act (SEMA) and the Justice for Victims of Corrupt Foreign Officials Act (JVCFOA)." />      <meta name="dcterms.title" content="Consolidated Canadian Autonomous Sanctions List" />      <meta name="dcterms:service" content="GAC-AMC_International" />      <meta name="dcterms.accessRights" content="2" />      <meta name="dcterms.creator" content="Global Affairs Canada" />      <meta name="dcterms.issued" title="W3CDTF" content="2015-10-19" />      <meta name="dcterms.modified" title="W3CDTF" content="2025-01-27" />      <meta name="dcterms.subject" title="scheme" content="Foreign service" />      <meta name="dcterms.language" title="ISO639-2/T" content="eng" />       <!-- REVIEW -->                <meta name="twitter:card" content="summary_large_image" />        <meta property="og:type" content="article" />        <meta property="og:site_name" content="GAC" />                                            <link re

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 3) ───────────────────────
│ The HTML content fetched so far mainly contains metadata and does not reveal the structure of the list of sanctioned individuals. To effectively locate the data, I'll search for any downloadable files, such as CSV or Excel files, that might contain the list. This is a common practice for such datasets.
│ 
│ Let's search for any links to downloadable files in the HTML content.
│ 
│ ```python
│ from bs4 import BeautifulSoup
│ 
│ # Parse the HTML content using BeautifulSoup
│ soup = BeautifulSoup(response.text, 'html.parser')
│ 
│ # Find all links in the HTML content
│ links = soup.find_all('a', href=True)
│ 
│ # Filter links that point to downloadable files (e.g., CSV, Excel)
│ download_links = [link['href'] for link in links if link['href'].endswith(('.csv', '.xls', '.xlsx'))]
│ 
│ # Print the download links
│ print(download_links)
│ ```
│ 
│ This code will help identify if there are any downloadable files containing the list of sanctioned individuals. If such files are found, we can proceed to download and process them.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 3) ───────────────

│ Executing:
from bs4 import BeautifulSoup

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links in the HTML content
links = soup.find_all('a', href=True)

# Filter links that point to downloadable files (e.g., CSV, Excel)
download_links = [link['href'] for link in links if link['href'].endswith(('.csv', '.xls', '.xlsx'))]

# Print the download links
print(download_links)
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
[]

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 4) ───────────────────────
│ It appears that there are no direct links to downloadable CSV or Excel files in the HTML content. This suggests that the list of sanctioned individuals might be embedded within the webpage itself, possibly in a table or another structured format.
│ 
│ Let's search for any tables in the HTML content that might contain the data we need.
│ 
│ ```python
│ # Find all tables in the HTML content
│ tables = soup.find_all('table')
│ 
│ # Print the number of tables found
│ print(f"Number of tables found: {len(tables)}")
│ 
│ # If tables are found, print the first few rows of the first table for inspection
│ if tables:
│     first_table = tables[0]
│     rows = first_table.find_all('tr')
│     for row in rows[:5]:  # Print first 5 rows
│         columns = row.find_all(['td', 'th'])
│         print([column.get_text(strip=True) for column in columns])
│ ```
│ 
│ This code will help us identify if there is a table containing the list of sanctioned individuals and allow us to inspect its structure.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 4) ───────────────

│ Executing:
# Find all tables in the HTML content
tables = soup.find_all('table')

# Print the number of tables found
print(f"Number of tables found: {len(tables)}")

# If tables are found, print the first few rows of the first table for inspection
if tables:
    first_table = tables[0]
    rows = first_table.find_all('tr')
    for row in rows[:5]:  # Print first 5 rows
        columns = row.find_all(['td', 'th'])
        print([column.get_text(strip=True) for column in columns])
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Number of tables found: 1
['Regulation', 'Entity', 'Title', 'Last Name', 'Given Names', 'Aliases', 'Date of Birth(YYYY/MM/DD)', 'Schedule', 'Item #', 'Date of Listing(YYYY/MM/DD)']
['Belarus', '', '', 'Atabekov', 'Khazalbek Bakhtibekovich', '', '', '1, Part 1', '1', '2020-09-28']
['Belarus', '', '', 'Balaba', 'Dmitry Vladimirovich', '', '1972', '1, Part 1', '2', '2020-09-28']
['Belarus', '', '', 'Barsukov', 'Aleksandr Petrovich', '', '1965-04-29', '1, Part 1', '3', '2020-09-28']
['Belarus', '', '', 'Dmukhailo', 'Yelena Nikolaevna', '', '1971', '1, Part 1', '4', '2020-09-28']

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 5) ───────────────────────
│ We have successfully identified a table containing the list of sanctioned individuals. The table includes columns such as "Regulation," "Entity," "Title," "Last Name," "Given Names," "Aliases," "Date of Birth," "Schedule," "Item #," and "Date of Listing."
│ 
│ Next, let's extract the data from this table and save it to a CSV file.
│ 
│ ```python
│ import csv
│ 
│ # Extract table headers
│ headers = [header.get_text(strip=True) for header in tables[0].find_all('th')]
│ 
│ # Extract table rows
│ data = []
│ for row in tables[0].find_all('tr')[1:]:  # Skip header row
│     columns = row.find_all('td')
│     data.append([column.get_text(strip=True) for column in columns])
│ 
│ # Save the data to a CSV file
│ csv_file_path = 'sanctioned_individuals.csv'
│ with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
│     writer = csv.writer(file)
│     writer.writerow(headers)
│     writer.writerows(data)
│ 
│ print(f"Data saved to {csv_file_path}")
│ ```
│ 
│ This code will extract the data from the table and save it to a CSV file named `sanctioned_individuals.csv`. Let's execute this step to complete the task.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 5) ───────────────

│ Executing:
import csv

# Extract table headers
headers = [header.get_text(strip=True) for header in tables[0].find_all('th')]

# Extract table rows
data = []
for row in tables[0].find_all('tr')[1:]:  # Skip header row
    columns = row.find_all('td')
    data.append([column.get_text(strip=True) for column in columns])

# Save the data to a CSV file
csv_file_path = 'sanctioned_individuals.csv'
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data)

print(f"Data saved to {csv_file_path}")
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Data saved to sanctioned_individuals.csv

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 6) ───────────────────────
│ The data has been successfully extracted from the table and saved to a CSV file named `sanctioned_individuals.csv` in your current working directory. You can now access the full list of sanctioned individuals from this file.
│ 
│ DONE
╰──────────────────────────────────────────────────
You (/Users/damiantran/python-sandbox): > show the content of the csv

╭─ Agent Response (Step 1) ───────────────────────
│ Let's read and display the content of the `sanctioned_individuals.csv` file to verify its contents.
│ 
│ ```python
│ import pandas as pd
│ 
│ # Read the CSV file and display its contents
│ csv_file_path = 'sanctioned_individuals.csv'
│ data = pd.read_csv(csv_file_path)
│ 
│ # Display the first few rows of the CSV file
│ print(data.head())
│ ```
│ 
│ This code will read the CSV file using the `pandas` library and display the first few rows for inspection. Let's execute this to see the content.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 1) ───────────────

│ Executing:
import pandas as pd

# Read the CSV file and display its contents
csv_file_path = 'sanctioned_individuals.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the CSV file
print(data.head())

✗ Error during execution:
╞══════════════════════════════════════════╡
│ Error:
No module named 'pandas'
╞══════════════════════════════════════════╡
│ Attempting to fix the error...
╰══════════════════════════════════════════╯

✗ Error during execution (attempt 1):
╞══════════════════════════════════════════╡
│ Error:
invalid syntax (<string>, line 2)
│ Another attempt will be made...

✗ Error during execution (attempt 2):
╞══════════════════════════════════════════╡
│ Error:
invalid syntax (<string>, line 2)
│ Result: 
✗ Code Execution Failed after 3 attempts
╞══════════════════════════════════════════╡
│ Error:
No module named 'pandas'
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 2) ───────────────────────
│ It seems that the `pandas` library is not installed in your environment. Let's install it first and then proceed to read and display the CSV file.
│ 
│ ```python
│ # Install pandas package
│ !pip install pandas
│ ```
│ 
│ Once pandas is installed, we can proceed to read and display the contents of the CSV file.
│ 
│ ```python
│ import pandas as pd
│ 
│ # Read the CSV file and display its contents
│ csv_file_path = 'sanctioned_individuals.csv'
│ data = pd.read_csv(csv_file_path)
│ 
│ # Display the first few rows of the CSV file
│ print(data.head())
│ ```
│ 
│ Let's execute these steps to ensure the code runs successfully.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 2) ───────────────

│ Executing:
# Install pandas package
!pip install pandas
Warning: Potentially dangerous operation detected. Proceed? (y/n): y

✗ Error during execution:
╞══════════════════════════════════════════╡
│ Error:
invalid syntax (<string>, line 2)
╞══════════════════════════════════════════╡
│ Attempting to fix the error...
╰══════════════════════════════════════════╯
Collecting pandas
  Downloading pandas-2.2.3-cp312-cp312-macosx_11_0_arm64.whl.metadata (89 kB)
Requirement already satisfied: numpy>=1.26.0 in /Users/damiantran/.local/pipx/venvs/local-operator/lib/python3.12/site-packages (from pandas) (2.2.2)
Requirement already satisfied: python-dateutil>=2.8.2 in /Users/damiantran/.local/pipx/venvs/local-operator/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
Collecting pytz>=2020.1 (from pandas)
  Using cached pytz-2024.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas)
  Using cached tzdata-2025.1-py2.py3-none-any.whl.metadata (1.4 kB)
Requirement already satisfied: six>=1.5 in /Users/damiantran/.local/pipx/venvs/local-operator/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
Downloading pandas-2.2.3-cp312-cp312-macosx_11_0_arm64.whl (11.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.4/11.4 MB 10.6 MB/s eta 0:00:00
Using cached pytz-2024.2-py2.py3-none-any.whl (508 kB)
Using cached tzdata-2025.1-py2.py3-none-any.whl (346 kB)
Installing collected packages: pytz, tzdata, pandas
Successfully installed pandas-2.2.3 pytz-2024.2 tzdata-2025.1
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
  Regulation Entity Title  Last Name  ... Date of Birth(YYYY/MM/DD)   Schedule Item # Date of Listing(YYYY/MM/DD)
0    Belarus    NaN   NaN   Atabekov  ...                       NaN  1, Part 1      1                  2020-09-28
1    Belarus    NaN   NaN     Balaba  ...                      1972  1, Part 1      2                  2020-09-28
2    Belarus    NaN   NaN   Barsukov  ...                1965-04-29  1, Part 1      3                  2020-09-28
3    Belarus    NaN   NaN  Dmukhailo  ...                      1971  1, Part 1      4                  2020-09-28
4    Belarus    NaN   NaN     Ipatov  ...                1964-10-30  1, Part 1      5                  2020-09-28

[5 rows x 10 columns]

│ Error Output:
[No error output]

│ Executing:
import pandas as pd

# Read the CSV file and display its contents
csv_file_path = 'sanctioned_individuals.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the CSV file
print(data.head())
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
  Regulation Entity Title  Last Name  ... Date of Birth(YYYY/MM/DD)   Schedule Item # Date of Listing(YYYY/MM/DD)
0    Belarus    NaN   NaN   Atabekov  ...                       NaN  1, Part 1      1                  2020-09-28
1    Belarus    NaN   NaN     Balaba  ...                      1972  1, Part 1      2                  2020-09-28
2    Belarus    NaN   NaN   Barsukov  ...                1965-04-29  1, Part 1      3                  2020-09-28
3    Belarus    NaN   NaN  Dmukhailo  ...                      1971  1, Part 1      4                  2020-09-28
4    Belarus    NaN   NaN     Ipatov  ...                1964-10-30  1, Part 1      5                  2020-09-28

[5 rows x 10 columns]

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 3) ───────────────────────
│ The content of the `sanctioned_individuals.csv` file has been successfully read and displayed. Here are the first few rows of the file:
│ 
│ ```
│   Regulation Entity Title  Last Name  ... Date of Birth(YYYY/MM/DD)   Schedule Item # Date of Listing(YYYY/MM/DD)
│ 0    Belarus    NaN   NaN   Atabekov  ...                       NaN  1, Part 1      1                  2020-09-28
│ 1    Belarus    NaN   NaN     Balaba  ...                      1972  1, Part 1      2                  2020-09-28
│ 2    Belarus    NaN   NaN   Barsukov  ...                1965-04-29  1, Part 1      3                  2020-09-28
│ 3    Belarus    NaN   NaN  Dmukhailo  ...                      1971  1, Part 1      4                  2020-09-28
│ 4    Belarus    NaN   NaN     Ipatov  ...                1964-10-30  1, Part 1      5                  2020-09-28
│ ```
│ 
│ This output shows the first five rows of the data, including columns such as "Regulation," "Last Name," "Given Names," "Date of Birth," and others.
│ 
│ DONE
╰──────────────────────────────────────────────────
You (/Users/damiantran/python-sandbox): > how many rows are there in total?

╭─ Agent Response (Step 1) ───────────────────────
│ To determine the total number of rows in the CSV file, we can simply check the length of the DataFrame. Let's do that:
│ 
│ ```python
│ # Get the total number of rows in the CSV file
│ total_rows = len(data)
│ 
│ # Print the total number of rows
│ print(f"Total number of rows: {total_rows}")
│ ```
│ 
│ This code will output the total number of rows in the `sanctioned_individuals.csv` file. Let's execute it.
╰──────────────────────────────────────────────────

╭─ Executing Code Blocks (Step 1) ───────────────

│ Executing:
# Get the total number of rows in the CSV file
total_rows = len(data)

# Print the total number of rows
print(f"Total number of rows: {total_rows}")
│ Result: 
✓ Code Execution Successful
╞══════════════════════════════════════════╡
│ Output:
Total number of rows: 4495

│ Error Output:
[No error output]
╰──────────────────────────────────────────────────

╭─ Agent Response (Step 2) ───────────────────────
│ The `sanctioned_individuals.csv` file contains a total of 4,495 rows. 
│ 
│ DONE
╰──────────────────────────────────────────────────
```

</details>

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
