from setuptools import find_packages, setup

setup(
    name="local-operator",
    packages=find_packages(),
    py_modules=["local_operator.cli"],
    entry_points={
        "console_scripts": [
            "local-operator = local_operator.cli:main",
        ],
    },
    install_requires=[
        "langchain-openai>=0.3.2",
        "langchain-ollama>=0.2.2",
        "langchain-anthropic>=0.3.3",
        "langchain-google-genai>=2.0.8",
        "langchain>=0.3.14",
        "langchain-community>=0.3.14",
        "python-dotenv>=1.0.1",
        "pydantic>=2.10.6",
        "tiktoken>=0.8.0",
        "uvicorn>=0.22.0",
        "fastapi>=0.115.8",
        "playwright>=1.49.1",
        "requests>=2.32.3",
    ],
    python_requires=">=3.12",
    extras_require={
        "dev": [
            "black",
            "isort",
            "pylint",
            "pyright",
            "pytest",
            "pytest-asyncio",
            "pip-audit",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
