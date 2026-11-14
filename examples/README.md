# Examples

Self-contained samples that run Local Operator in other environments. They are
not part of the `local-operator` package: each example has its own README,
dependencies and tests, and pins a released `local-operator` from PyPI.

| Example | What it shows |
| --- | --- |
| [`agentcore/`](agentcore/README.md) | lop running headless on Amazon Bedrock AgentCore Runtime. Each session gets its own microVM, and Bedrock calls are signed with SigV4 by the runtime's IAM role, with no API keys. The example deploys with CloudFormation and builds the image in CodeBuild. Its README includes transcripts from a real deployment. |
