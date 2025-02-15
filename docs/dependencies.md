# Dependency Graph

Local Operator is built with a modular architecture that separates concerns into distinct components. The Mermaid diagram below illustrates the key components and their relationships:

- **CLI Domain**: The command line interface for user interaction
- **Configuration Domain**: Manages configuration settings and API credentials
- **Execution Domain**: Handles code execution, safety, and console output
- **Modeling Domain**: Defines language models, data types, and prompt engineering
- **Agent Management Domain**: Coordinates AI agents and admin tools
- **Tooling Domain**: Integrates with external tools and services
- **API Domain**: Provides a FastAPI server interface
- **External Tools**: Third-party integrations like web browsing and search

The arrows indicate dependencies and data flow between components. Key relationships include:

- The CLI interfaces with configuration and manages agents
- The executor orchestrates model configuration, tool integration, and agent management
- The operator coordinates execution while managing configuration, credentials and types
- The API server provides a REST interface to core functionality
- External tools are integrated through the tooling layer

The following was generated with ❤️ by Local Operator for humans 🤝

```mermaid
graph LR
    classDef cli fill:#1f77b4,stroke:#333,stroke-width:2px
    classDef config fill:#ff7f0e,stroke:#333,stroke-width:2px
    classDef execution fill:#2ca02c,stroke:#333,stroke-width:2px
    classDef modeling fill:#d62728,stroke:#333,stroke-width:2px
    classDef agents fill:#9467bd,stroke:#333,stroke-width:2px
    classDef tools fill:#8c564b,stroke:#333,stroke-width:2px
    classDef api fill:#e377c2,stroke:#333,stroke-width:2px
    classDef external fill:#7f7f7f,stroke:#333,stroke-width:2px

    subgraph CLI [CLI Domain]
        direction TB
        cli(CLI - Command Line Interface):::cli
    end

    subgraph Configuration [Configuration Domain]
        direction TB
        config(Config - Configuration Management):::config
        credentials(Credentials - API Key Management):::config
    end

    subgraph Execution [Execution Domain]
        direction TB
        executor(Executor - Code Execution & Safety):::execution
        console(Console - Output Formatting):::execution
    end

    subgraph Modeling [Modeling Domain]
        direction TB
        model(Model - Language Model Configuration):::modeling
        types(Types - Data Type Definitions):::modeling
        prompts(Prompts - Prompt Engineering):::modeling
    end

    subgraph Agents [Agent Management Domain]
        direction TB
        agents(Agents - Agent Registry & Management):::agents
        admin(Admin - Admin Tools):::agents
    end

    subgraph Tools [Tooling Domain]
        direction TB
        tools(Tools - External Tool Integration):::tools
    end

    subgraph API [API Domain]
        direction TB
        server(Server - FastAPI API):::api
    end

    subgraph External [External Tools]
        direction TB
        web_browsing(Web Browsing - Playwright):::external
        serp_api(SERP API - Search Engine Results):::external
    end


    cli --> config:::config
    cli -- "uses" --> operator:::execution
    cli -- "manages" --> agents:::agents

    executor -- "configures" --> model:::modeling
    executor -- "integrates" --> tools:::tools
    executor -- "defines" --> types:::modeling
    executor -- "manages" --> agents:::agents
    executor --> console:::execution
    executor -- "uses" --> web_browsing:::external
    executor -- "uses" --> serp_api:::external

    operator -- "orchestrates" --> executor:::execution
    operator -- "reads" --> config:::config
    operator -- "secures" --> credentials:::config
    operator -- "defines" --> types:::modeling
    operator -- "manages" --> agents:::agents

    agents -- "defines" --> types:::modeling

    admin -- "manages" --> agents:::agents
    admin -- "reads" --> config:::config
    admin -- "uses" --> executor:::execution
    admin -- "integrates" --> tools:::tools

    server -- "manages" --> agents:::agents
    server -- "reads" --> config:::config
    server -- "secures" --> credentials:::config
    server -- "uses" --> executor:::execution
    server -- "configures" --> model:::modeling
    server -- "orchestrates" --> operator:::execution
    server -- "defines" --> types:::modeling

    prompts -- "integrates" --> tools:::tools
    tools -- "uses" --> web_browsing:::external
    tools -- "uses" --> serp_api:::external

    style cli fill:#1f77b4,stroke:#333,stroke-width:2px
    style config fill:#ff7f0e,stroke:#333,stroke-width:2px
    style execution fill:#2ca02c,stroke:#333,stroke-width:2px
    style modeling fill:#d62728,stroke:#333,stroke-width:2px
    style agents fill:#9467bd,stroke:#333,stroke-width:2px
    style tools fill:#8c564b,stroke:#333,stroke-width:2px
    style api fill:#e377c2,stroke:#333,stroke-width:2px
    style external fill:#7f7f7f,stroke:#333,stroke-width:2px
```
