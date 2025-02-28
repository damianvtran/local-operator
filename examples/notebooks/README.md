# 📚 Local Operator Example Notebooks 🖥️

Welcome to our collection of Jupyter notebooks showcasing Local Operator's capabilities across various domains. These notebooks are exported from actual conversations with Local Operator agents and demonstrate how Local Operator agents can assist with tasks ranging from software development workflows to data science competitions and web research. Each notebook provides a practical, hands-on example of how to leverage AI assistance for complex tasks while maintaining full control over the execution environment.

The notebooks are a visualization of the conversation history between the user and the Local Operator agents, but the real magic happens when you install Local Operator yourself and ask the agent the same questions with the same/similar parameters. ✨

💡 Each notebook can be replicated by installing the `local-operator` package version in the configuration summary and asking the agent the same questions with the same/similar parameters.

📝 Notebooks can be created from conversations simply by asking the agent at the end of some task to save the conversation as a notebook.

## 🔄 [Automated Git Commit Message Generation](github_commit.ipynb)

This interactive notebook demonstrates a Local Operator agent session powered by the `qwen/qwen-2.5-72b-instruct` model. The Local Operator agent automatically reads the diffs from the current git commit and generates a suitable and concise commit message. The notebook shows how AI can help developers maintain clear version history and improve their git workflow with contextually relevant commit messages.\

## 🔀 [End-to-End Pull Request Workflow Automation](github_pr.ipynb)

This comprehensive notebook walks through the complete pull request creation process with a Local Operator agent. It demonstrates how Local Operator agents can systematically review code diffs (including detailed changes to `local_operator/prompts.py`), exclude unstaged changes from your PR, properly target the main branch, and complete PR templates with thorough descriptions and testing information. Ideal for developers looking to streamline their code review and contribution workflows.

## 🔢 [MNIST Digit Recognition with Deep Learning](kaggle_digit_recognizer.ipynb)

This detailed notebook provides an end-to-end solution for the Kaggle Digit Recognizer competition generated from a single question to Local Operator. The agent plans a methodical approach including exploratory data analysis, preprocessing techniques (normalization and scaling of pixel values), CNN architecture selection with implementation details, data augmentation strategies to improve model robustness, and ensemble methods to boost accuracy. The notebook captures a complete pipeline from initial data exploration to final submission preparation with performance metrics.

🏆 This notebook was submitted to the [Digit Recognizer Kaggle competition](https://www.kaggle.com/competitions/digit-recognizer) achieving a 99.3% accuracy, visible at the Notebook submission [here](https://www.kaggle.com/code/damianvtran/local-operator-mnist-digits-auto-ml-99-3).

## 🏠 [Advanced House Price Prediction with XGBoost](kaggle_home_data_competition.ipynb)

This advanced notebook shows a Local Operator agent tackling the Kaggle Home Data competition using sophisticated modeling techniques. It showcases systematic hyperparameter tuning with XGBoost, featuring comprehensive cross-validation strategies and optimization approaches. The notebook includes detailed training logs, feature importance analysis, and demonstrates how to iteratively refine prediction accuracy through methodical parameter optimization and model evaluation.

🏆 This notebook was submitted to the [Home Data Kaggle competition](https://www.kaggle.com/competitions/home-data-for-ml-course/overview) achieving a top 5% score, visible at the Notebook submission [here](https://www.kaggle.com/code/damianvtran/local-operator-housing-prices-automl-top-5).

## 🚢 [Titanic Survival Prediction using LightGBM](kaggle_titanic_competition.ipynb)

This practical notebook shows a Local Operator agent tackling the classic Kaggle Titanic survival prediction competition. It implements LightGBM with detailed feature engineering, handling of missing values, and categorical encoding techniques. The notebook includes comprehensive training logs showing model convergence, demonstrates row-wise multi-threading optimizations for performance, and illustrates a complete machine learning workflow from data preparation to final prediction.

🏆 This notebook was submitted to the [Titanic Kaggle competition](https://www.kaggle.com/competitions/titanic) achieving an accuracy of 77%, visible at the Notebook submission [here](https://www.kaggle.com/code/damianvtran/local-operator-titanic-survivors-auto-ml).

## 🌐 [Web Research and Data Extraction Techniques](web_research_scraping.ipynb)

This informative notebook showcases a Local Operator session (using `qwen/qwen-2.5-72b-instruct` via openrouter) that implements web scraping techniques to extract the Canadian sanctions list. Given a single prompt from the user, the agent proceeds with a structured planning approach to retrieval of data from government sources with SERP API tool usage, processing semi-structured information into a clean CSV format, and implementation of verification steps to ensure data completeness and accuracy. The use case demonstrated in the notebook is particularly valuable for data collection, transformation, and validation tasks involving public web resources.
