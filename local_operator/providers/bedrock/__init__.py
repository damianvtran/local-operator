"""Amazon Bedrock provider (Claude via InvokeModelWithResponseStream).

Modules: ``connection`` (pure descriptor/Region/env-tier logic), ``auth``
(bearer or SigV4 via botocore, lazily imported), ``client`` (the
``AnthropicClient`` subclass), ``errors`` (AWS error table), ``discovery``,
``check`` (save-time validation) and ``store`` (the AuthStore seal hook).
Operator docs: ``docs/providers/BEDROCK.md``.

Deliberately empty of imports: ``providers.registry`` reaches
``bedrock.connection`` on every provider census, and that must not drag in
botocore or the client stack.
"""
