# Dependency Graph

Local Operator is built with a modular architecture that separates concerns into distinct components. The Mermaid diagram below illustrates the key components and their relationships:

- **CLI Interface**: The entry point and user interaction layer
- **Core Components**: The main business logic and execution engine
- **Configuration & State**: Settings, credentials and shared types
- **Agent Management**: Handles creation and coordination of AI agents
- **External Services**: Third-party integrations like LLM providers

The arrows indicate dependencies and data flow between components.

```mermaid
graph TB
    subgraph Entry Points
        CLI[cli.py] -.-> Operator
        CLI -.-> FastAPI
    end

    subgraph Core Components
        Operator[operator.py]
        Executor[executor.py]
        Tools[tools.py]
        Model[model.py]
        Console[console.py]
        FastAPI[server.py]
    end

    subgraph Configuration
        Config[config.py]
        Credentials[credentials.py]
        Types[types.py]
        Prompts[prompts.py]
    end

    subgraph Agent Management
        Agents[agents.py]
    end

    subgraph External Services
        LLMProviders[LLM Providers]
        Browser[Playwright Browser]
        SerpAPIClient[serpapi.py]
    end

    subgraph Admin
        AdminModule[admin.py]
    end

    %% Core component relationships
    Operator --> Executor
    Operator --> Console
    Executor --> Tools
    Executor --> Model
    FastAPI --> Operator
    FastAPI --> AdminModule
    
    %% Configuration flows
    Config --> Operator
    Config --> FastAPI
    Credentials --> Model
    Types --> Executor
    Types --> FastAPI
    Prompts --> Executor
    
    %% Agent management
    Agents --> Operator
    
    %% External integrations  
    Model --> LLMProviders
    Tools --> Browser
    Tools --> SerpAPIClient

    AdminModule --> Agents
    AdminModule --> Config
    AdminModule --> Executor
    AdminModule --> Operator
    AdminModule --> Tools

    %% Component descriptions
    classDef entryPoint fill:#ff9999,stroke:#ff0000,stroke-width:2px,color:#000
    classDef core fill:#4a90e2,stroke:#2c3e50,stroke-width:2px,color:#fff
    classDef config fill:#95DAC1,stroke:#2c3e50,stroke-width:2px,color:#000
    classDef agent fill:#FD6F96,stroke:#2c3e50,stroke-width:2px,color:#fff
    classDef external fill:#FFEBA1,stroke:#2c3e50,stroke-width:2px,color:#000
    classDef admin fill:#c778dd,stroke:#2c3e50,stroke-width:2px,color:#fff

    class CLI entryPoint
    class Operator,Executor,Tools,Model,Console,FastAPI core
    class Config,Credentials,Types,Prompts config
    class Agents agent
    class LLMProviders,Browser,SerpAPIClient external
    class AdminModule admin

    %% Descriptions
    CLI[CLI - Main entry point]
    Operator[Operator - Environment manager]
    Executor[Executor - Code execution & safety]
    Tools[Tools - Agent capabilities]
    Model[Model - LLM configuration]
    Console[Console - Terminal UI]
    FastAPI[FastAPI Server]
    Config[Config - Settings]
    Credentials[Credentials - API keys]
    Types[Types - Data structures]
    Prompts[Prompts - System prompts]
    Agents[Agents - Agent definitions]
    LLMProviders[OpenAI/Anthropic/etc]
    Browser[Web browsing]
    SerpAPIClient[SerpAPI Client]
    AdminModule[Admin Module]
```
