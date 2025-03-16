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

<picture>
  <img src="./static/preview-example-ui.png" alt="Local Operator UI Dashboard Example">
</picture>

<p align="center"><i>Local Operator server powering an open source UI.  The frontend is optional and available <a href="https://github.com/damianvtran/local-operator-ui">here</a></i></p>

**<span style="color: #38C96A">Local Operator</span>** empowers you to run Python code safely on your own machine through an intuitive chat interface. The AI agent:

🎯 **Plans & Executes** - Breaks down complex goals into manageable steps and executes them with precision.

🔒 **Prioritizes Security** - Built-in safety checks by independent AI review and user confirmations keep your system protected

🌐 **Flexible Deployment** - Run completely locally with Ollama models or leverage cloud providers like OpenAI

🔧 **Problem Solving** - Intelligently handles errors and roadblocks by adapting approaches and finding alternative solutions

This project is proudly open source under the GPL 3.0 license. We believe AI tools should be accessible to everyone, given their transformative impact on productivity. Your contributions and feedback help make this vision a reality!

> "Democratizing AI-powered productivity, one conversation at a time."

<div align="center">
  <a href="#-contributing">Contribute</a> •
  <a href="https://local-operator.com">Learn More</a> •
  <a href="#-examples">Examples</a>
</div>

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

The Local Operator provides a command-line interface where you can:

1. Interact with the AI assistant in natural language
2. Execute Python code blocks marked with ```python``` syntax
3. Get safety warnings before executing potentially dangerous operations
4. View execution results and error messages
5. Maintain context between code executions

Visit the [Local Operator website](https://local-operator.com) for visualizations and information about the project.

## 💻 Requirements

- Python 3.12+
- For 3rd party hosting: [OpenRouter](https://openrouter.ai/keys), [OpenAI](https://platform.openai.com/api-keys), [DeepSeek](https://platform.deepseek.ai/), [Anthropic](https://console.anthropic.com/), [Google](https://ai.google.dev/), or other API key (prompted for on first run)
- For local hosting: [Ollama](https://ollama.com/download) model installed and running

## 🛠️ Setup

To run Local Operator with a 3rd party cloud-hosted LLM model, you need to have an API key.  You can get one from OpenAI, DeepSeek, Anthropic, or other providers.

Once you have the API key, install the operator CLI with the following command:

```bash
pip install local-operator
```

### Web Browsing

To enable web browsing, you can install the playwright browsers with the following command:

```bash
playwright install
```

This is not necessary to use the web browsing tool, as the agent will automatically install the browsers when they are needed, but it can be faster to install them ahead of start up if you know you will need them.

If you would like to run with a local Ollama model, you will need to install Ollama first from [here](https://ollama.ai/download), and fetch a model using `ollama pull`.  Make sure that the ollama server is running with `ollama serve`.

### Web Search

To enable web search, you will need to get a free SERP API key from [SerpApi](https://serpapi.com/users/sign_up).  On the free plan, you get 100 credits per month which is generally sufficient for light to moderate personal use.  

The agent uses a web search tool integrated with SERP API to fetch information from the web if you have the `SERP_API_KEY` set up in the Local Operator credentials.  The agent can still browse the web without it, though information access will be less efficient.

Get your API key and then configure the `SERP_API_KEY` credential:

```bash
local-operator credential update SERP_API_KEY
```

Once the credential is set, the agent will be able to search the web for information
using its web search tool on the next start up.

## 🖥️ Usage (CLI)

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

## 🚀 Usage (Single Execution Mode)

The operator can be run in a single execution mode where it will execute a single task and then exit.  This is useful for running the operator in a non-interactive way such as in a script.

```bash
local-operator exec "Make a new file called test.txt and write Hello World in it"
```

This will execute the task and then exit with a code 0 if successful, or a non-zero code if there was an error.

## 📡 Usage (Server)

To run the operator as a server, use the following command:

```bash
local-operator serve
```

This will start the FastAPI server app and host at `http://localhost:8080` by default with uvicorn.  You can change the host and port by using the `--host` and `--port` arguments.  

To view the API documentation, navigate to `http://localhost:8080/docs` in your browser for Swagger UI or `http://localhost:8080/redoc` for ReDoc.

For development, use the `--reload` argument to enable hot reloading.

## 🧠 Usage (Agents)

The agents mode is helpful for passing on knowledge between agents and between runs.  It is also useful for creating reusable agentic experiences learned through conversation with the user.

The agents CLI command can be used to create, edit, and delete agents.  Agents are
metadata and persistence for conversation history.  They are an easy way to create replicable conversation experiences based on "training" through conversation with the user.

To create a new agent, use the following command:

```bash
local-operator agents create "My Agent"
```

This will create a new agent with the name "My Agent" and a default conversation history.  The agent will be saved in the `~/.local-operator/agents` directory.

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

## 🔑 Configuration

### Configuration Values

The operator uses a configuration file to manage API keys and other settings.  It can be created at `~/.local-operator/config.yml` with the `local-operator config create` command.  You can edit this file directly to change the configuration.

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

### Configuration Options

- `conversation_length`: The number of messages to keep in the conversation history.  Defaults to 100.
- `detail_length`: The number of messages to keep in the detail history.  All messages beyond this number excluding the primary system prompt will be summarized into a shorter form to reduce token costs.  Defaults to 35.
- `hosting`: The hosting platform to use.  Avoids needing to specify the `--hosting` argument every time.
- `model_name`: The name of the model to use.  Avoids needing to specify the `--model` argument every time.
- `max_learnings_history`: The maximum number of learnings to keep in the learnings history.  Defaults to 50.
- `auto_save_conversation`: Whether to automatically save the conversation history to a file.  Defaults to `false`.

### Credentials

Credentials are stored in the `~/.local-operator/credentials.yml` file.  Credentials can be updated at any time by running `local-operator credential update <credential_name>`.

Example:

```bash
local-operator credential update SERP_API_KEY
```

To clear a credential, use the following command:

```bash
local-operator credential delete SERP_API_KEY
```

- `SERP_API_KEY`: The API key for the SERP API from [SerpApi](https://serpapi.com/users/sign_up).  This is used to search the web for information.  This is required for the agent to be able to do real time searches of the web using search engines.  The agent can still browse the web without it, though information access will be less efficient.

- `TAVILY_API_KEY`: The API key for the Tavily API from [Tavily](https://tavily.com/signup).  Alternative to SERP API with pay as you go pricing.  The per unit cost is lower
for personal use if you go over the SERP API 100 requests per month limit.  The disadvantage is that the search results are not based off of Google like SERP API so the search depth is not as extensive.  Good for if you have run into the SERP API limit for the month.

- `OPENROUTER_API_KEY`: The API key for the OpenRouter API.  This is used to access the OpenRouter service with a wide range of models.  It is the best option for being able to easily switch between models with less configuration.

- `OPENAI_API_KEY`: The API key for the OpenAI API.  This is used to access the OpenAI model.

- `DEEPSEEK_API_KEY`: The API key for the DeepSeek API.  This is used to access the DeepSeek model.

- `ANTHROPIC_API_KEY`: The API key for the Anthropic API.  This is used to access the Anthropic model.

- `GOOGLE_API_KEY`: The API key for the Google API.  This is used to access the Google model.

- `MISTRAL_API_KEY`: The API key for the Mistral API.  This is used to access the Mistral model.

## 📝 Examples

👉 Check out the [example notebooks](./examples/notebooks/) for detailed examples of tasks completed with Local Operator in Jupyter notebook format.  

These notebooks were created in Local Operator by asking the agent to save the conversation history to a notebook each time after asking the agent to complete tasks.  You can generally replicate them by asking the same user prompts with the same configuration settings.

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

## 📝 License

This project is licensed under the GPL 3.0 License - see the LICENSE file for details.
