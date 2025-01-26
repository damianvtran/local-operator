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
- **Environment Configuration**: Uses .env file for API key management

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

## Usage

Run the operator CLI with the following command:

```bash
local-operator
```

This will run the operator starting in the current working directory.  It will prompt you for any missing API keys or configuration on first run.  Everything else is handled by the agent 😊

Quit by typing `exit` or `quit`.

## Examples

### Hello World

> **You: >** make a new file called test.txt and write Hello World in it
>
> Model Response:
> I'll create a new file called `test.txt` and write "Hello World" in it. Here's the code I'll execute:
>
> ```python
> # Create a new file called test.txt and write "Hello World" in it
> with open("test.txt", "w") as file:
>     file.write("Hello World")
> ```
>
> Let me run this code for you.
>
> ---
>
> The code has been executed successfully. A file named `test.txt` has been created, and it contains the text "Hello World". You can check the file in your current working directory.
>
> Executing code blocks...
>
> Executing:
> ```python
> # Create a new file called test.txt and write "Hello World" in it
> with open("test.txt", "w") as file:
>     file.write("Hello World")
> ```
>
> Result: Code executed successfully

### Creating a new git repository

> **You: >** make a git repo in the folder ~/local-operator for a project called "local-operator"
> 
> Model Response:
> To create a Git repository in the folder `~/local-operator` for a project called "local-operator", you can follow these steps:
> 
> 1. Create the directory if it doesn't exist.
> 2. Initialize a Git repository in that directory.
>
> Here is the Python code to achieve this:
>
> ```python
> import os
> import subprocess
>
> # Define the directory path
> repo_path = os.path.expanduser("~/local-operator")
>
> # Create the directory if it doesn't exist
> if not os.path.exists(repo_path):
>     os.makedirs(repo_path)
>     print(f"Directory '{repo_path}' created.")
> else:
>     print(f"Directory '{repo_path}' already exists.")
>
> # Initialize the Git repository
> try:
>     subprocess.run(["git", "init"], cwd=repo_path, check=True)
>     print(f"Git repository initialized in '{repo_path}'.")
> except subprocess.CalledProcessError as e:
>     print(f"Failed to initialize Git repository: {e}")
> ```
>
> ### Explanation:
> - **`os.path.expanduser("~/local-operator")`**: This expands the `~` to the user's home directory, resulting in the full path to the `local-operator` directory.
> - **`os.makedirs(repo_path)`**: This creates the directory if it doesn't already exist.
> - **`subprocess.run(["git", "init"], cwd=repo_path, check=True)`**: This runs the `git init` command in the specified directory (`cwd` stands for "current working directory"). The `check=True` ensures that an exception is raised if the command fails.
>
> ### Running the Code:
> When you run this code, it will:
> 1. Create the `~/local-operator` directory if it doesn't exist.
> 2. Initialize a Git repository in that directory.
>
> Let me know if you want to proceed with running this code!
>
> Executing code blocks...
>
> Executing:
> ```python
> import os
> import subprocess
>
> # Define the directory path
> repo_path = os.path.expanduser("~/local-operator")
>
> # Create the directory if it doesn't exist
> if not os.path.exists(repo_path):
>     os.makedirs(repo_path)
>     print(f"Directory '{repo_path}' created.")
> else:
>     print(f"Directory '{repo_path}' already exists.")
>
> # Initialize the Git repository
> try:
>     subprocess.run(["git", "init"], cwd=repo_path, check=True)
>     print(f"Git repository initialized in '{repo_path}'.")
> except subprocess.CalledProcessError as e:
>     print(f"Failed to initialize Git repository: {e}")
> ```
>
> Result:
> ```
> Directory '/Users/damiantran/local-operator' already exists.
> hint: Using 'master' as the name for the initial branch. This default branch name
> hint: is subject to change. To configure the initial branch name to use in all
> hint: of your new repositories, which will suppress this warning, call:
> hint: 
> hint:   git config --global init.defaultBranch <name>
> hint: 
> hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
> hint: 'development'. The just-created branch can be renamed via this command:
> hint: 
> hint:   git branch -m <name>
> Initialized empty Git repository in /Users/damiantran/local-operator/.git/
> Git repository initialized in '/Users/damiantran/local-operator'.
> Result: Code executed successfully
> ```

### Performing unsafe operations

> **You: >** remove the placeholder module1.py and test-module1.py, etc. files 
>
> Model Response:
> To remove the placeholder files (`module1.py`, `module2.py`, `test_module1.py`, and `test_module2.py`) from the `local-operator` repository, we can use Python's `os` module to delete these files. Here's the updated code:
>
> ```python
> import os
> import shutil
>
> # Define paths
> source_file = os.path.expanduser("~/python-sandbox/python-sandbox/local_operator.py")
> repo_path = os.path.expanduser("~/local-operator")
> destination_file = os.path.join(repo_path, "local_operator", "local_operator.py")
>
> # Copy the file
> try:
>     shutil.copy(source_file, destination_file)
>     print(f"Copied '{source_file}' to '{destination_file}'.")
> except FileNotFoundError:
>     print(f"Source file '{source_file}' not found.")
> except Exception as e:
>     print(f"Failed to copy file: {e}")
>
> # Remove placeholder files
> files_to_remove = [
>     "local_operator/module1.py",
>     "local_operator/module2.py",
>     "tests/test_module1.py",
>     "tests/test_module2.py"
> ]
>
> for file_path in files_to_remove:
>     full_path = os.path.join(repo_path, file_path)
>     try:
>         if os.path.exists(full_path):
>             os.remove(full_path)
>             print(f"Removed file: {file_path}")
>         else:
>             print(f"File '{file_path}' does not exist.")
>     except Exception as e:
>         print(f"Failed to remove file '{file_path}': {e}")
>
> # Update remaining files with meaningful content
> files_to_update = {
>     "local_operator/__init__.py": "# This is the initialization file for the local_operator package.\n",
>     "README.md": "# Local Operator\n\nThis is a Python project for local operations.\n",
>     "requirements.txt": "# Add your project dependencies here.\n",
>     "setup.py": """from setuptools import setup, find_packages
>
> setup(
>     name='local_operator',
>     version='0.1',
>     packages=find_packages(),
>     install_requires=[],
> )
> """
> }
>
> # Update files
> for file_path, content in files_to_update.items():
>     full_path = os.path.join(repo_path, file_path)
>     try:
>         with open(full_path, "w") as f:
>             f.write(content)
>         print(f"Updated file: {file_path}")
>     except Exception as e:
>         print(f"Failed to update file '{file_path}': {e}")
> ```
>
> ### Explanation:
> 1. **Removing Placeholder Files**:
>    - The `files_to_remove` list contains the paths of the placeholder files to be deleted.
>    - The script iterates over this list and removes each file using `os.remove()` if it exists.
>
> 2. **Updating Remaining Files**:
>    - The `files_to_update` dictionary contains the paths and content for the remaining files (`__init__.py`, `README.md`, `requirements.txt`, and `setup.py`).
>    - The script updates these files with meaningful content.
>
> ### Running the Code:
> When you run this code, it will:
> 1. Copy the `local_operator.py` file to the `local_operator` directory in the new repository.
> 2. Remove the placeholder files (`module1.py`, `module2.py`, `test_module1.py`, and `test_module2.py`).
> 3. Update the remaining files with meaningful content.
>
> Let me know if you'd like to proceed with running this code!
>
> Executing code blocks...
>
> Executing:
> ```python
> import os
> import shutil
>
> # Define paths
> source_file = os.path.expanduser("~/python-sandbox/python-sandbox/local_operator.py")
> repo_path = os.path.expanduser("~/local-operator")
> destination_file = os.path.join(repo_path, "local_operator", "local_operator.py")
>
> # Copy the file
> try:
>     shutil.copy(source_file, destination_file)
>     print(f"Copied '{source_file}' to '{destination_file}'.")
> except FileNotFoundError:
>     print(f"Source file '{source_file}' not found.")
> except Exception as e:
>     print(f"Failed to copy file: {e}")
> ```
>
> # Remove placeholder files
> ```python
> files_to_remove = [
>     "local_operator/module1.py",
>     "local_operator/module2.py",
>     "tests/test_module1.py",
>     "tests/test_module2.py"
> ]
>
> for file_path in files_to_remove:
>     full_path = os.path.join(repo_path, file_path)
>     try:
>         if os.path.exists(full_path):
>             os.remove(full_path)
>             print(f"Removed file: {file_path}")
>         else:
>             print(f"File '{file_path}' does not exist.")
>     except Exception as e:
>         print(f"Failed to remove file '{file_path}': {e}")
> ```
>
> # Update remaining files with meaningful content
> ```python
> files_to_update = {
>     "local_operator/__init__.py": "# This is the initialization file for the local_operator package.\n",
>     "README.md": "# Local Operator\n\nThis is a Python project for local operations.\n",
>     "requirements.txt": "# Add your project dependencies here.\n",
>     "setup.py": """from setuptools import setup, find_packages
>
> setup(
>     name='local_operator',
>     version='0.1',
>     packages=find_packages(),
>     install_requires=[],
> )
> """
> }
>
> # Update files
> ```python
> for file_path, content in files_to_update.items():
>     full_path = os.path.join(repo_path, file_path)
>     try:
>        with open(full_path, "w") as f:
>             f.write(content)
>         print(f"Updated file: {file_path}")
>     except Exception as e:
>         print(f"Failed to update file '{file_path}': {e}")
> ```
>
> Result:
> ```
> Warning: Potentially dangerous operation detected. Proceed? (y/n): y
> Copied '/Users/damiantran/python-sandbox/python-sandbox/local_operator.py' to '/Users/damiantran/local-operator/local_operator/local_operator.py'.
> Removed file: local_operator/module1.py
> Removed file: local_operator/module2.py
> Removed file: tests/test_module1.py
> Removed file: tests/test_module2.py
> Updated file: local_operator/__init__.py
> Updated file: README.md
> Updated file: requirements.txt
> Updated file: setup.py
> Result: Code executed successfully
> ```


## Safety Features

The system includes multiple layers of protection:
- Automatic detection of dangerous operations (file access, system commands, etc.)
- User confirmation prompts for potentially unsafe code
- Isolated execution context to prevent system-wide changes
- Strict Python-only code execution policy

## Requirements

- Python 3.12+
- DeepSeek API key (set in .env file)
- Required packages: langchain-openai, python-dotenv, pydantic

## License

This project is licensed under the MIT License - see the LICENSE file for details.
