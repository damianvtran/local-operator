# Security Policy

## Introduction

Thank you for helping us keep Local Operator secure. Local Operator strives to be a secure Python-based environment for on-device, interactive task execution by an AI agent. Your vigilance in reporting any potential security vulnerabilities is essential to maintaining a trustworthy and robust system. This policy outlines how to report issues and how we address them.

## Supported Versions

It is recommended to use the latest version of the Local Operator PyPI package.  The latest version can be found at [https://pypi.org/project/local-operator/](https://pypi.org/project/local-operator/).

Local Operator currently supports Python 3.12+ and is maintained with strict security practices including:

- Regular dependency updates and scanning with pip-audit
- Automated static analysis (using flake8, black, isort, and pyright).
- Comprehensive testing with pytest (including async tests).

Security updates and patches will be released for all actively maintained versions.

## Security Features

Local Operator implements several layers of security, including:

- **Code Safety Verification:**  
  Built-in safety checks analyze code before execution to detect potentially dangerous operations. The agent operates with a focus on validating safety and system impact prior to running any code.

- **User Confirmation for Risky Operations:**  
  Before executing dangerous or high-risk operations (such as file system changes), the system prompts the user for confirmation, thereby reducing the risk of accidental or malicious damage.

- **Continuous Integration Security Checks:**  
  Our CI pipeline includes linting, type checking, and testing to catch issues early and to enforce secure coding practices.  Security features should always be included in the test suites.

## Reporting a Vulnerability

If you identify a security-related vulnerability or security incident in Local Operator, we invite you to [report the vulnerability privately](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) and create a GitHub Security Advisory on our repository. Your advisory helps us quickly address potential risks and informs the community about the issue. When creating the advisory, please include:

1. **Detailed Description:**  
   Provide a clear explanation of the vulnerability or security incident, including any relevant context that can help us understand its nature.

2. **Reproduction Steps:**  
   List the steps to reproduce the problem, including any code snippets, logs, screenshots, or other pertinent details.

3. **Impact Assessment:**  
   Describe the potential impact, including what systems or data might be affected and any associated risks to end users.

4. **Additional Information:**  
   If applicable, include any mitigating factors, suggested fixes, or further observations that could assist in resolving the issue. Optionally, provide your contact information for follow-up questions.

Please ensure that sensitive details are managed securely. You can update your advisory with additional information as necessary. Our security team will review your submission promptly and work toward a timely resolution.

## Disclosure Policy

We follow responsible disclosure practices. Once a vulnerability or incident is confirmed and addressed, we will:

- Publish a public security advisory on GitHub detailing the issue, affected versions, and upgrade instructions.
- Provide a timeline for the release of patches or fixes.

## Contact

For any additional security-related inquiries or assistance with creating a GitHub Security Advisory, please contact:

- **Email:** [damianvtran@gmail.com](mailto:damianvtran@gmail.com)

## Additional Resources

For further guidance on creating GitHub Security Advisories and responsible disclosure practices, please refer to the [GitHub Security Advisories Documentation](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/creating-a-repository-security-advisory).

Thank you for your commitment to keeping Local Operator secure.
