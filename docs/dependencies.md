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
        CLI[cli.py]
        SingleExec[Single Execution Mode]
    end

    subgraph Core Components
        Operator[operator.py]
        Executor[executor.py]
        Tools[tools.py]
        Model[model.py]
        Console[console.py]
        FastAPI[FastAPI Server]
    end

    subgraph Configuration
        Config[config.py]
        Credentials[credentials.py]
        Types[types.py]
        Prompts[prompts.py]
    end

    subgraph Agent Management
        Agents[agents.py]
        AgentRegistry[AgentRegistry]
    end

    subgraph External Services
        LLMProviders[LLM Providers]
        Browser[Playwright Browser]
    end

    %% Entry point flows
    CLI --> |Interactive Mode| Operator
    CLI --> |Server Mode| FastAPI
    CLI --> |Single Execution| SingleExec
    SingleExec --> Operator
    FastAPI --> Operator

    %% Core component relationships
    Operator --> Executor
    Operator --> Console
    Executor --> Tools
    Executor --> Model
    FastAPI --> Executor
    
    %% Configuration flows
    Config --> Operator
    Config --> FastAPI
    Credentials --> Model
    Types --> Executor
    Types --> FastAPI
    Prompts --> Executor
    
    %% Agent management
    Agents --> Operator
    AgentRegistry --> Operator
    AgentRegistry --> FastAPI
    
    %% External integrations  
    Model --> LLMProviders
    Tools --> Browser

    %% Component descriptions
    classDef entryPoint fill:#ff9999,stroke:#ff0000,stroke-width:2px,color:#000
    classDef core fill:#4a90e2,stroke:#2c3e50,stroke-width:2px,color:#fff
    classDef config fill:#95DAC1,stroke:#2c3e50,stroke-width:2px,color:#000
    classDef agent fill:#FD6F96,stroke:#2c3e50,stroke-width:2px,color:#fff
    classDef external fill:#FFEBA1,stroke:#2c3e50,stroke-width:2px,color:#000

    class CLI,SingleExec entryPoint
    class Operator,Executor,Tools,Model,Console,FastAPI core
    class Config,Credentials,Types,Prompts config
    class Agents,AgentRegistry agent
    class LLMProviders,Browser external

    %% Descriptions
    CLI[CLI - Main entry point]
    SingleExec[Single Execution - One-off tasks]
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
    AgentRegistry[AgentRegistry - Agent state]
    LLMProviders[OpenAI/Anthropic/etc]
    Browser[Web browsing]
```
