# Security Policy

## Introduction

Thank you for helping us keep Local Operator secure. Local Operator strives to be a secure Python-based environment for on-device, interactive task execution by an AI agent. Your vigilance in reporting any potential security vulnerabilities is essential to maintaining a trustworthy and robust system. This policy outlines how to report issues and how we address them.

## Supported Versions

It is recommended to use the latest version of the Local Operator PyPI package.  The latest version can be found at [https://pypi.org/project/local-operator/](https://pypi.org/project/local-operator/).

Local Operator currently supports Python 3.12+ and is maintained with strict security practices including:

- Regular dependency updates and scanning with pip-audit
- Automated static analysis (using flake8, black, isort, and pyright).
- Comprehensive testing with pytest (including async tests).

Security fixes ship as the **next PyPI release** of `local-operator`; there are
no backported patch lines for older minor versions. When an advisory names a
fixed version, upgrade to it (or newer) and pin at least that version in your
own dependency spec, for example `local-operator>=0.47.5`.

Because a fix only protects the installs that pick it up, we also recommend
that consumers run their own dependency scanning — `pip-audit`, GitHub
Dependabot, or Snyk — in CI so the advisory reaches you even if you miss the
release notes. See "How advisories are published" for how we make sure those
tools actually know about a fix.

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

We follow responsible disclosure practices. Once a vulnerability or incident is confirmed, we will:

- Acknowledge the report on the private advisory within 48 hours.
- Reproduce the issue in an isolated environment and fix it on a branch that goes through the same review and QA gate as any other change.
- Ship the fix as a GitHub Release and PyPI release, with release notes linking the advisory.
- Publish the advisory only **after** the fixed version is available on PyPI, so that scanners can point to a safe version rather than only flagging the vulnerable one.
- Request a CVE, credit the reporter, and verify the advisory propagates to the databases downstream tooling reads (see below).

## How advisories are published

A published GitHub repository advisory is not, by itself, a warning that reaches
users: CI/CD scanners such as `pip-audit`, Dependabot, and Snyk read from
vulnerability databases, not from our repository page. For every advisory we
therefore commit to the full propagation chain, not just the first step:

1. **GitHub repository advisory** published with a precise affected range and
   the patched version, a "Verified fix" section (fixed version, fixing PR,
   before/after evidence), CWE and CVSS, and credit to the reporter.
2. **CVE request** through GitHub. Requesting a CVE is what puts the advisory
   into GitHub's curation queue; publishing alone does not.
3. **GitHub Advisory Database** — the advisory appears as a reviewed global
   entry at `https://github.com/advisories/<GHSA>`.
4. **OSV** ([osv.dev](https://osv.dev)) ingests the reviewed entry; from there
   it reaches **PyPI**'s per-release vulnerability data, **`pip-audit`** (both
   the `pypi` and `osv` sources), **Dependabot** alerts, and **Snyk**.
5. We check that chain a few days and again about two weeks after publication.
   If the advisory has not propagated by then, we submit it directly to the
   [PyPA advisory database](https://github.com/pypa/advisory-database) (which
   OSV, PyPI and `pip-audit` ingest) and to Snyk's
   [vulnerability disclosure](https://snyk.io/vulnerability-disclosure/) intake.

You can verify a fix landed in the tooling you use with, for example:

```sh
echo 'local-operator==<vulnerable-version>' > /tmp/req.txt
pip-audit --no-deps -s pypi -r /tmp/req.txt
pip-audit --no-deps -s osv  -r /tmp/req.txt
```

The maintainer-facing procedure behind this section, including the exact
commands and the follow-up schedule, is in
[`docs/security-advisory-runbook.md`](docs/security-advisory-runbook.md).

## Past advisories

| Advisory | Severity | Published | Fixed in | Reported by |
| --- | --- | --- | --- | --- |
| [GHSA-22mg-8gw7-636x](https://github.com/damianvtran/local-operator/security/advisories/GHSA-22mg-8gw7-636x) | Critical | 2026-09-05 | 0.47.5 | [@mhdgning131](https://github.com/mhdgning131) |
| [GHSA-3xjw-9qpc-53mh](https://github.com/damianvtran/local-operator/security/advisories/GHSA-3xjw-9qpc-53mh) | High | 2026-09-05 | 0.47.5 | [@mhdgning131](https://github.com/mhdgning131) |

If you are on a version earlier than 0.47.5, upgrade:

```sh
pip install --upgrade 'local-operator>=0.47.5'
```

## Contact

For any additional security-related inquiries or assistance with creating a GitHub Security Advisory, please contact:

- **Email:** [damian@gominerva.com](mailto:damian@gominerva.com)

## Additional Resources

For further guidance on creating GitHub Security Advisories and responsible disclosure practices, please refer to the [GitHub Security Advisories Documentation](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/creating-a-repository-security-advisory).

Thank you for your commitment to keeping Local Operator secure.
