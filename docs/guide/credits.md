<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Credits and Acknowledgements

Local Operator stands on the shoulders of the broader open-source agent
community. Several aspects of this harness's implementation were shaped by
studying and drawing inspiration from the projects below. We're grateful to
their authors and contributors for building in the open.

- **[opencode](https://github.com/anomalyco/opencode)**: created by
  [Dax Raad (`thdxr`)](https://github.com/thdxr) and the Anomaly (formerly SST)
  team. Its terminal-native, model-agnostic coding-agent design informed our
  thinking on the interactive CLI experience and provider-agnostic model
  handling.
- **[oh-my-pi](https://github.com/can1357/oh-my-pi)**: authored and maintained
  by [Can Bölük (`can1357`)](https://github.com/can1357), building on
  [Pi](https://github.com/earendil-works/pi) by
  [Mario Zechner (`mariozechner`)](https://github.com/mariozechner). Its
  approach to agent orchestration and harness ergonomics inspired aspects of our
  subagent and tooling implementation.
- **[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)**
  (`dsh`): an open-source agent harness by
  [DeepSeek AI](https://deepseek.com), built on an everything-is-a-plugin
  architecture over [Cordis](https://github.com/cordiverse/cordis). Its web
  client's UI and interaction design informed parts of our desktop app: a single
  sidebar carrying navigation alongside the agent, team, and session lists,
  section-level view controls, progressive disclosure for long lists, and a
  width-constrained conversation whose resize handles reveal themselves on
  hover. We studied the design and built our own implementation; DeepSeek AI has
  not reviewed, endorsed, or contributed to this project.
- **[pstack](https://github.com/cursor/plugins/tree/main/pstack)** (in
  [Cursor's plugins repository](https://github.com/cursor/plugins)): created by
  [Lauren Tan (`poteto`)](https://github.com/poteto). Its rigorous-engineering
  playbooks (verification, test-first development, smallest-change principles)
  informed several of our builtin skills.
- **[superpowers](https://github.com/obra/superpowers)**: created by
  [Jesse Vincent (`obra`)](https://github.com/obra). Its verification,
  test-driven development, systematic debugging and code-review skills informed
  our corresponding builtins.
- **[anthropics/skills](https://github.com/anthropics/skills)**: created by
  [Anthropic](https://github.com/anthropics). Its `frontend-design` and
  `webapp-testing` skills (Apache-2.0) informed our frontend-design and
  browser-qa builtins.
- **[ux-ui-agent-skills](https://github.com/plugin87/ux-ui-agent-skills)**:
  created by [plugin87](https://github.com/plugin87). Its design-review skills
  and objective quality gates informed our design-qa builtin.
- **[claude-skills](https://github.com/jezweb/claude-skills)**: created by
  [Jeremy Dawes (`jezweb`)](https://github.com/jezweb). Its design-review skill
  informed our design-qa builtin.
- **[agent-config](https://github.com/event4u-app/agent-config)**: created by
  the [event4u-app](https://github.com/event4u-app) team. Its governed-skill and
  design-review material informed our design-qa builtin.
- **[ehmo/platform-design-skills](https://github.com/ehmo/platform-design-skills)**:
  created by [ehmo](https://github.com/ehmo). Its condensed Apple HIG, Material 3
  and WCAG 2.2 rule sets informed our design-qa checks.
- **[microsoft/skills](https://github.com/microsoft/skills)**: created by
  [Microsoft](https://github.com/microsoft). Its frontend-design-review skill
  informed our frontend-design and design-qa builtins.
- **[mattpocock/skills](https://github.com/mattpocock/skills)**: created by
  [Matt Pocock](https://github.com/mattpocock). Its small, composable
  engineering skills (TDD, code review, debugging, writing for agents) informed
  the scope and tone of our catalog.
- **[awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)**:
  a community index maintained by [VoltAgent](https://github.com/VoltAgent). Its
  skill-quality standards informed our skill-authoring builtin.

Inspiration drawn from these projects informed our own independent
implementation; any mistakes here are our own.

## Session voice

Per-session voice (the hands-free mode in the desktop app) began as a local
rig built around the projects and models below, and its design record
([`docs/design/qwen-voice.md`](../design/qwen-voice.md)) is written against
them. **The voice that ships today does not run any of the rig.** The app
listens with Silero voice-activity detection in a hidden page, sends each
recording to this backend, and speaks through the speech provider the user
chose (Radient, or their own ElevenLabs key). The rig's launcher code stays in
the desktop app's repository, never started, and none of the projects below is
bundled or copied into this repository.

- **[qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)**: by the
  QwenAudio team (Apache-2.0). Its gateway and speech-provider contract shaped
  the original rig and our design decisions about it. Not reviewed or endorsed
  by them.
- **[speech-to-speech](https://github.com/huggingface/speech-to-speech)**: by
  [Hugging Face](https://huggingface.co) (Apache-2.0). The rig ran it as a
  separate program. Not reviewed or endorsed by them.
- **[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)**:
  by the Qwen team at Alibaba Cloud (Apache-2.0). The rig's speaking model; the
  weights were downloaded by the rig and are not part of this repository.
- **[Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)**: by NVIDIA
  (CC-BY-4.0). The rig's listening model, with the same arrangement. Not
  reviewed or endorsed by NVIDIA.
- **[Smart Turn](https://github.com/pipecat-ai/smart-turn)**: by the Pipecat
  team at Daily (BSD-2-Clause). The rig's turn detector; the shipped path
  detects turns from silence only.
- **[Silero VAD](https://github.com/snakers4/silero-vad)**: by the Silero Team
  (MIT), run in the desktop app through
  [vad-web](https://github.com/ricky0123/vad) by ricky0123 (ISC) on
  [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) by Microsoft
  (MIT). These are redistributed in the desktop app and carry their licence
  notices in that repository's `THIRD_PARTY_NOTICES.md`.

Speech itself is provided by services the user signs in to or brings a key for:
[Radient](https://radienthq.com) and [ElevenLabs](https://elevenlabs.io). They
are called over their APIs; nothing of theirs is redistributed here.

## A note on reuse and credit

The agent-harness projects above are open source under permissive licenses
(MIT; the Anthropic skills cited above are Apache-2.0), the session-voice
projects carry the licenses named beside each, and Local Operator itself is
MIT. Under that license, you are free to draw inspiration from or reuse code
from Local Operator in your own work. Verbatim reuse of the code requires
retaining the copyright notice and license text, as the license states. Beyond that legal
minimum, we appreciate credit where credit is due: an acknowledgement
of the projects and people whose work you build on, in the same spirit as the
credits above. It costs little and it keeps open source healthy.

Core contributor: Damian Tran &lt;[damian@gominerva.com](mailto:damian@gominerva.com)&gt;.
