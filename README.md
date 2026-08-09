<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/local-operator-icon-2-dark-clear.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/local-operator-icon-2-light-clear.png">
  <img alt="Shows a black Local Operator Logo in light color mode and a white one in dark color mode."
       src="./static/local-operator-icon-2-light-clear.png">
</picture>

<h1 align="center">Local Operator: AI Agent Assistants On Your Device</h1>
<div align="center">
  <h2>🤖 Personal AI Assistants that Turn Ideas into Action</h2>
  <p><i>Real-time code execution on your device through natural conversation</i></p>
</div>

<br />

<p align="center">
  <img src="./static/preview-example-ui.gif" alt="Local Operator UI Dashboard Example" style="width: 640px">
</div>

<p align="center"><i>Local Operator server powering the open source UI.  The frontend is optional and available <a href="https://github.com/damianvtran/local-operator-ui">here</a></i> or by downloading from the <a href="https://local-operator.com">website</a></p>

<br />

**<span style="color: #38C96A">Local Operator</span>** empowers you to run Python code safely on your own machine through an intuitive chat interface. The AI agent:

🎯 **Plans & Executes** - Breaks down complex goals into manageable steps and executes them with precision.

🔒 **Prioritizes Security** - Built-in safety checks by independent AI review and user confirmations keep your system protected

🌐 **Flexible Deployment** - Run completely locally with Ollama models or leverage cloud providers like OpenAI

🔧 **Problem Solving** - Intelligently handles errors and roadblocks by adapting approaches and finding alternative solutions

This project is proudly open source under the MIT license. We believe AI tools should be accessible to everyone, given their transformative impact on productivity. Your contributions and feedback help make this vision a reality!

> "Democratizing AI-powered productivity, one conversation at a time."

<div align="center">
  <a href="#-contributing">Contribute</a> •
  <a href="https://local-operator.com">Learn More</a> •
  <a href="#-examples">Examples</a>
</div>

## 📚 Table of Contents

- [🔑 Key Features](#-key-features)
- [💻 Requirements](#-requirements)
- [🚀 Getting Started](#-getting-started)
  - [🛠️ Installing Local Operator](#️-installing-local-operator)
    - [📦 Install via pip](#-install-via-pip)
    - [📦 Install via Nix Flake](#-install-via-nix-flake)
  - [🐋 Running Local Operator in Docker](#-running-local-operator-in-docker)
- [🖥️ Usage (CLI)](#️-usage-cli)
  - [🦙 Run with a local Ollama model](#-run-with-a-local-ollama-model)
  - [🐳 Run with DeepSeek](#-run-with-deepseek)
  - [🤖 Run with OpenAI](#-run-with-openai)
  - [🔂 Run Single Execution Mode](#-run-single-execution-mode)
  - [📡 Running in Server Mode](#-running-in-server-mode)
  - [🧠 Running in Agent mode](#-running-in-agent-mode)
  - [🔧 Configuration Values](#-configuration-values)
  - [🛠️ Configuration Options](#️-configuration-options)
  - [🔐 Credentials](#-credentials)
- [🌟 Radient Agent Hub and Automatic Model Selection](#-radient-agent-hub-and-automatic-model-selection)
- [📝 Examples](#-examples)
- [👥 Contributing](#-contributing)
- [🔒 Safety Features](#-safety-features)
- [📜 License](#-license)

## 🔑 Key Features

- **Interactive CLI Interface**: Chat with an AI assistant that can execute Python code locally
- **Server Mode**: Run the operator as a FastAPI server to interact with the agent through a web interface
- **Code Safety Verification**: Built-in safety checks analyze code for potentially dangerous operations
- **Contextual Execution**: Maintains execution context between code blocks
- **Conversation History**: Tracks the full interaction history for context-aware responses
- **Local Model Support**: Supports closed-circuit on-device execution with Ollama.
- **LangChain Integration**: Uses 3rd party cloud-hosted LLM models through LangChain's ChatOpenAI implementation
- **Asynchronous Execution**: Safe code execution with async/await pattern
- **Environment Configuration**: Uses credential manager for API key management
- **Image Generation**: Create and modify images using the FLUX.1 model from FAL AI
- **Web Search**: Load-balanced search across DuckDuckGo, Tavily, Perplexity, Brave, Exa, SerpApi, or SearXNG

The Local Operator provides a command-line interface where you can:

1. Interact with the AI assistant in natural language
2. Execute Python code blocks marked with `python` syntax
3. Get safety warnings before executing potentially dangerous operations
4. View execution results and error messages
5. Maintain context between code executions

Visit the [Local Operator website](https://local-operator.com) for visualizations and information about the project.

## 💻 Requirements

- Python 3.12+ with `pip` installed
- For 3rd party hosting: [OpenRouter](https://openrouter.ai/keys), [OpenAI](https://platform.openai.com/api-keys), [DeepSeek](https://platform.deepseek.ai/), [Anthropic](https://console.anthropic.com/), [Google](https://ai.google.dev/), or other API key (prompted for on first run)
- For local hosting: [Ollama](https://ollama.com/download) model installed and running

## 🚀 Getting Started

### 🛠️ Installing Local Operator

To run Local Operator with a 3rd party cloud-hosted LLM model, you need to have an API key. You can get one from OpenAI, DeepSeek, Anthropic, or other providers.

### 📦 Install via pip

> ⚠️ **Linux Installs (Ubuntu 23.04+, Fedora 38+, Debian 12+)**  
> Due to recent changes in how Python is managed on modern Linux distributions (see [PEP 668](https://peps.python.org/pep-0668/)), you **cannot use `pip install` globally** on system Python.

- MacOS & Windows

  ```bash
  pip install local-operator
  ```

- Linux

  ```bash
  pipx install local-operator
  ```

- 📌 (Optional) Virtual python

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install local-operator
  ```

- 📌 (Optional) Extra features

  The default install is deliberately small and fast — it carries only what the
  agent itself needs. Optional features live behind extras, which you can add
  at any time:

  | Extra | Adds |
  | --- | --- |
  | `server` | The HTTP API server (`local-operator serve`) and the background scheduler |
  | `mcp` | Model Context Protocol client support, for connecting external MCP servers |
  | `images` | HEIC/HEIF image attachment decoding |
  | `tokenizer` | Exact BPE token counting (without it, token usage is estimated) |
  | `all` | Everything above |

  ```bash
  pip install "local-operator[server]"
  pip install "local-operator[all]"
  ```

  Quote the requirement — most shells try to glob the square brackets. If you
  reach a feature whose extra is missing, the agent tells you which one to
  install rather than failing with an import error.

- 📌 (Optional) Enabling Web Browsing

  This is not necessary to use the web browsing tool, as the agent will automatically install the browsers when they are needed, but it can be faster to install them ahead of start up if you know you will need them.

  ```bash
  playwright install
  ```

- 📌 (Optional) Configuring Web Search

  Web search works on first run: DuckDuckGo and Tavily's documented keyless
  endpoint are enabled and requests rotate between them. If one provider is
  rate-limited, challenged, or unavailable, the same request falls through to
  the next enabled provider.

  ```bash
  local-operator search list
  local-operator search test "Python 3.13 release notes"
  local-operator search enable perplexity
  local-operator search disable duckduckgo
  local-operator search order tavily,perplexity,duckduckgo
  local-operator search balance round_robin  # or: ordered
  local-operator search off                  # `on` restores the master switch
  ```

  The same non-secret controls are available inside the TUI with `/search`.
  Provider setup is explicit:

  ```bash
  local-operator search setup tavily --oauth
  local-operator search setup tavily --api-key
  local-operator search setup brave --api-key
  local-operator search setup searxng --url https://search.example.com
  ```

  Tavily OAuth adds the official remote MCP server and uses its connected
  `tavily-search` tool from the built-in load-balancing chain. Start or reload a
  session to complete sign-in. Without OAuth, Tavily uses `TAVILY_API_KEY` when
  configured and its rate-limited keyless mode otherwise.

  | Provider | Access | Default |
  | --- | --- | --- |
  | DuckDuckGo | Credential-free HTML search | Enabled |
  | Tavily | Keyless, `TAVILY_API_KEY`, or OAuth MCP | Enabled |
  | Perplexity | Anonymous search or `PERPLEXITY_API_KEY` | Disabled |
  | Brave | `BRAVE_API_KEY` | Disabled |
  | Exa | `EXA_API_KEY` | Disabled |
  | SerpApi | `SERPAPI_API_KEY` (legacy `SERP_API_KEY` also works) | Disabled |
  | SearXNG | Self-hosted endpoint URL | Disabled |

  Model context is capped at 6,000 characters, with shorter per-answer and
  per-snippet limits. Complete source URLs are retained whenever a result is
  included; the agent can open one with `browser` for the full page. In the TUI,
  expand a completed `web_search` card to see each candidate's page name, URL,
  and short provider-supplied snippet.

- 📌 (Optional) Enabling Image Generation

  To enable image generation capabilities, you'll need to get a FAL AI API key from [FAL AI](https://fal.ai/dashboard/keys). The Local Operator uses the FLUX.1 model from FAL AI to generate and modify images.

  1. Get your API key and then configure the `FAL_API_KEY` credential:

     ```bash
     local-operator credential update <FAL_API_KEY>
     ```

### 📦 Install via Nix Flake

If you use [Nix](https://nixos.org/) for development, this project provides a `flake.nix` for easy, reproducible setup. The flake ensures all dependencies are available and configures a development environment with a single command.

1. **Enter the development shell:**

   ```bash
   nix develop
   ```

   This will drop you into a shell with all required dependencies (Python, pip, etc.) set up for development.

2. **Run the project as usual:**

   You can now use the CLI or run scripts as described in the rest of this README.

#### Benefits

- No need to manually install Python or other dependencies.
- Ensures a consistent environment across all contributors.
- Works on Linux, macOS, and (with [nix-darwin](https://github.com/LnL7/nix-darwin)) on macOS.

For more information about Nix flakes, see the [NixOS flake documentation](https://nixos.wiki/wiki/Flakes).

### 🐋 Running Local Operator in Docker

To run Local Operator in docker, ensure docker is running and run

```bash
docker compose up --d
```

## 🖥️ Usage (CLI)

Run the operator CLI with the following command:

### 🦙 Run with a local Ollama model

Download and install Ollama first from [here](https://ollama.ai/download).

```bash
local-operator --hosting ollama --model qwen2.5:14b
```

### 🐳 Run with DeepSeek

```bash
local-operator --hosting deepseek --model deepseek-chat
```

### 🤖 Run with OpenAI

```bash
local-operator --hosting openai --model gpt-4o
```

This will run the operator starting in the current working directory. It will prompt you for any missing API keys or configuration on first run. Everything else is handled by the agent 😊

Quit by typing `exit` or `quit`.

Run `local-operator --help` for more information about parameters and configuration.

### 🔂 Run Single Execution Mode

The operator can be run in a single execution mode where it will execute a single task and then exit. This is useful for running the operator in a non-interactive way such as in a script.

```bash
local-operator exec "Make a new file called test.txt and write Hello World in it"
```

This will execute the task and then exit with a code 0 if successful, or a non-zero code if there was an error.

### 📡 Running in Server Mode

To run the operator as a server, use the following command:

```bash
local-operator serve
```

This will start the FastAPI server app and host at `http://localhost:8080` by default with uvicorn. You can change the host and port by using the `--host` and `--port` arguments.

To view the API documentation, navigate to `http://localhost:8080/docs` in your browser for Swagger UI or `http://localhost:8080/redoc` for ReDoc.

For development, use the `--reload` argument to enable hot reloading.

### 🧠 Running in Agent mode

The agents mode is helpful for passing on knowledge between agents and between runs. It is also useful for creating reusable agentic experiences learned through conversation with the user.

The agents CLI command can be used to create, edit, and delete agents. Agents are
metadata and persistence for conversation history. They are an easy way to create replicable conversation experiences based on "training" through conversation with the user.

To create a new agent, use the following command:

```bash
local-operator agents create "My Agent"
```

This will create a new agent with the name "My Agent" and a default conversation history. The agent will be saved in the `~/.local-operator/agents` directory.

To list all agents, use the following command:

```bash
local-operator agents list
```

To delete an agent, use the following command:

```bash
local-operator agents delete "My Agent"
```

You can then apply an agent in any of the execution modes by using the `--agent` argument to invoke that agent by name.

For example:

```bash
local-operator --agent "My Agent"
```

or

```bash
local-operator --hosting openai --model gpt-4o exec "Make a new file called test.txt and write Hello World in it" --agent "My Agent"
```

### 🔧 Configuration Values

The operator uses a configuration file to manage API keys and other settings. It can be created at `~/.local-operator/config.yml` with the `local-operator config create` command. You can edit this file directly to change the configuration.

To create a new configuration file, use the following command:

```bash
local-operator config create
```

To edit a configuration value via the CLI, use the following command:

```bash
local-operator config edit <key> <value>
```

To edit a configuration value via the configuration file directly, use the following command:

```bash
local-operator config open
```

To list all available configuration options and their descriptions, use the following command:

```bash
local-operator config list
```

### 🛠️ Configuration Options

- `conversation_length`: The number of messages to keep in the conversation history. Defaults to 100.
- `detail_length`: The number of messages to keep in the detail history. All messages beyond this number excluding the primary system prompt will be summarized into a shorter form to reduce token costs. Defaults to 35.
- `hosting`: The hosting platform to use. Avoids needing to specify the `--hosting` argument every time.
- `model_name`: The name of the model to use. Avoids needing to specify the `--model` argument every time.
- `max_learnings_history`: The maximum number of learnings to keep in the learnings history. Defaults to 50.
- `auto_save_conversation`: Whether to automatically save the conversation history to a file. Defaults to `false`.

### 🔐 Credentials

Credentials are stored in `~/.local-operator/.env`. Update them with `local-operator credential update <credential_name>` or use `local-operator search setup <provider> --api-key`.

Example:

```bash
local-operator credential update SERPAPI_API_KEY
```

To clear a credential, use the following command:

```bash
local-operator credential delete SERPAPI_API_KEY
```

- `SERPAPI_API_KEY`: SerpApi key for Google-backed results. The legacy
  `SERP_API_KEY` name remains accepted.
- `TAVILY_API_KEY`: Tavily key for account-backed limits; Tavily also supports
  keyless requests and `local-operator search setup tavily --oauth`.
- `PERPLEXITY_API_KEY`: Perplexity Sonar API key; anonymous search remains
  available without one when the provider is enabled.
- `BRAVE_API_KEY`: Brave Search API key.
- `EXA_API_KEY`: Exa semantic search API key.

- `FAL_API_KEY`: The API key for the FAL AI API from [FAL AI](https://fal.ai/dashboard/keys). This enables image generation capabilities using the FLUX.1 text-to-image model. With this key, the agent can generate images from text descriptions and modify existing images based on prompts. The FAL AI API provides high-quality image generation with various customization options like image size, guidance scale, and inference steps.

- `OPENROUTER_API_KEY`: The API key for the OpenRouter API. This is used to access the OpenRouter service with a wide range of models. It is the best option for being able to easily switch between models with less configuration.

- `OPENAI_API_KEY`: The API key for the OpenAI API. This is used to access the OpenAI model.

- `DEEPSEEK_API_KEY`: The API key for the DeepSeek API. This is used to access the DeepSeek model.

- `ANTHROPIC_API_KEY`: The API key for the Anthropic API. This is used to access the Anthropic model.

- `GOOGLE_API_KEY`: The API key for the Google API. This is used to access the Google model.

- `MISTRAL_API_KEY`: The API key for the Mistral API. This is used to access the Mistral model.

---

## 🌟 Radient Agent Hub and Automatic Model Selection

Radient enables seamless sharing, hosting, and auto-selection of AI agents and models through the Agent Hub in Local Operator. The Agent Hub is public and available to all for downloading agents, however to publish an agent you will need to set up an account on the [Radient Console](https://console.radienthq.com). You can push your agents to the Radient Hub, pull agents shared by others, and leverage Radient's automatic model selection for optimal performance and cost reductions.

### Setting Up a Radient Account

1. **Sign Up & Create an Application**

   - Go to [https://console.radienthq.com](https://console.radienthq.com) and sign up for a free account.
   - After logging in, create a new application in the Radient Console Applications section.
   - Copy your generated **RADIENT_API_KEY** from the application creation dialog.

2. **Configure Your API Key in Local Operator**

   - Set your Radient API key using the credentials manager:

     ```bash
     local-operator credential update RADIENT_API_KEY
     ```

### Pushing and Pulling Agents

- **Push an Agent to Radient**

  - You must be logged in (RADIENT_API_KEY configured) to push agents.
  - Use either the agent's name or ID:

    ```bash
    local-operator agents push --name "<agent_name>"
    ```

    or

    ```bash
    local-operator agents push --id "<agent_id>"
    ```

  - This uploads your agent to the Radient Agents Hub for sharing or backup.

- **Pull an Agent from Radient**

  - Download an agent by its Radient ID (no RADIENT_API_KEY required):

    ```bash
    local-operator agents pull --id "<agent_id>"
    ```

### Using Radient Hosting for Model Auto-Selection

Radient can automatically select the best model for your task, removing the need to specify a model manually.

1. **Configure Your API Key** (if not already done):

   ```bash
   local-operator credential update RADIENT_API_KEY
   ```

2. **Run Local Operator with Radient Hosting**:

   ```bash
   local-operator --hosting radient
   ```

   - No `--model` argument is needed; Radient will select the optimal model automatically. The model will be selected on a step-by-step basis to optimize for the best model for the job and reduce agentic AI costs.

#### Example Workflow

```bash
# Set up your Radient API key
local-operator credential update RADIENT_API_KEY

# Push an agent to Radient
local-operator agents push --name "My Agent"

# Pull an agent from Radient
local-operator agents pull --id "radient-agent-id-123"

# Use Radient hosting for automatic model selection
local-operator --hosting radient
```

> **Note:** You must have a valid RADIENT_API_KEY configured to push agents or use Radient hosting.

For more details, visit the [Radient Console](https://console.radienthq.com) or see the [Local Operator documentation](https://local-operator.com).

## 📝 Examples

👉 Check out the [example notebooks](./examples/notebooks/) for detailed examples of tasks completed with Local Operator in Jupyter notebook format.

These notebooks were created in Local Operator by asking the agent to save the conversation history to a notebook each time after asking the agent to complete tasks. You can generally replicate them by asking the same user prompts with the same configuration settings.

Some examples of helpful tasks completed with Local Operator:

- 🔄 **[Automated Git Commit Message Generation](examples/notebooks/github_commit.ipynb)**: Generates commit messages from git diffs using `qwen/qwen-2.5-72b-instruct`.
- 🔀 **[End-to-End Pull Request Workflow Automation](examples/notebooks/github_pr.ipynb)**: Automates pull request creation, code review, and template completion.
- 🔢 **[MNIST Digit Recognition with Deep Learning](examples/notebooks/kaggle_digit_recognizer.ipynb)**: End-to-end solution for Kaggle Digit Recognizer competition, achieving 99.3% accuracy.
- 🏠 **[Advanced House Price Prediction with XGBoost](examples/notebooks/kaggle_home_data_competition.ipynb)**: Tackles Kaggle Home Data competition using XGBoost, achieving a top 5% score.
- 🚢 **[Titanic Survival Prediction using LightGBM](examples/notebooks/kaggle_titanic_competition.ipynb)**: Predicts Titanic survival using LightGBM, achieving 77% accuracy.
- 🌐 **[Web Research and Data Extraction Techniques](examples/notebooks/web_research_scraping.ipynb)**: Extracts Canadian sanctions list using web scraping with `qwen/qwen-2.5-72b-instruct` and SERP API.
- 📈 **[Business Pricing and Margin Calculation](examples/notebooks/business_pricing_margin.ipynb)**: Assists with business pricing decisions by calculating optimal subscription prices.

## 👥 Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to:

- Submit bug reports and feature requests
- Set up your development environment
- Submit pull requests
- Follow our coding standards and practices
- Join our community discussions

Your contributions help make Local Operator better for everyone. We appreciate all forms of help, from code improvements to documentation updates.

## 🔒 Safety Features

The system includes multiple layers of protection:

- Automatic detection of dangerous operations (file access, system commands, etc.)
- User confirmation prompts for potentially unsafe code
- Agent prompt with safety focused execution policy
- Support for local Ollama models to prevent sending local system data to 3rd parties

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
