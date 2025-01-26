from setuptools import setup, find_packages

setup(
    name="local-operator",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "local-operator = local_operator.main:main",
        ],
    },
    install_requires=[
        "langchain-openai>=0.3.2",
        "python-dotenv>=1.0.1",
        "pydantic>=2.10.6",
        "flake8>=7.1.0",
        "black>=24.4.2",
        "isort>=5.13.2",
        "pyright>=1.1.368",
    ],
    python_requires=">=3.12",
    extras_require={
        "dev": [
            "black",
            "isort",
            "pylint",
            "pyright",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
