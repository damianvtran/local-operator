# zh-CN wire.errors.sdk_startup — staged translation packet

4/4 keys, pinned commit 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5. Translation only: not audited, not activated, no repo change.

## Values
- attach_hosting_unconfigured — 没有正在运行的运行时为{resume}提供服务，且此根目录无法启动一个：{error}为此根目录（其config.yml）配置托管，然后用spawn_session()/deliver()启动该ID，待运行时存在后再附接。
- attach_runtime_missing — 没有正在运行的运行时为{resume}提供服务；以mode=''own''打开它即可在当前进程中运行，或先用spawn_session()/deliver()启动它，待运行时存在后再附接。
- profile_attach_failed — 无法附接配置档{profile}
- tool_inventory_mismatch — spec.tools {spec_tools}与ApprovalPolicy.declared({policy_tools})声明的工具清单不同；本次运行的范围只能声明一次

## Contract
These four, unlike the 21 sdk_spec producers, render through startup_refusal_text with the resolved locale. Carrying the namespace therefore changes what `str(SessionSpecError)` contains for a zh-CN-resolved root — the RFC per-CARRIED design, and exactly why the context demands a migration boundary. No producer/str/args/attribute/API/runtime change is made or claimed here.

Known defect this packet cannot fix: {error} embeds opaque producer English (`session_factory.py:816`, "Hosting platform is not configured."), so a localized outer sentence still carries one English clause.

## Evidence
`node --max-old-space-size=256 validate.mjs` — exit 0, 0.7 s, eight negative controls rejected, actual log in mechanical.log. Rendered fit, pseudo, RTL, production renderer and live exception text: NOT RUN. No independent audit performed.
