"""
This module defines the data models and registry for managing language models,
including their pricing information and configuration.

It exposes key classes and data structures for:
- Defining model information (ModelInfo) such as pricing, context window,
  and supported features.
- Calculating the cost of a request based on token usage (CostCalculator).
- Registering known models from various providers like OpenAI, Anthropic,
  and others, along with their specific configurations.
- Configuring language models (configure_model) from different hosting
  platforms (e.g., OpenAI, Anthropic, Ollama) using the Langchain library.
- Validating the existence of a model and the validity of an API key
  (validate_model).

The module also includes utility functions for checking if a model exists
in a provider's response data (_check_model_exists_payload).
"""
