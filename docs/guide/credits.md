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

## A note on reuse and credit

All of the projects above are open source under permissive licenses (MIT; the
Anthropic skills cited above are Apache-2.0), and Local Operator itself is
MIT. Under that license, you are free to draw inspiration from or reuse code
from Local Operator in your own work. Verbatim reuse of the code requires
retaining the copyright notice and license text, as the license states. Beyond that legal
minimum, we appreciate credit where credit is due: an acknowledgement
of the projects and people whose work you build on, in the same spirit as the
credits above. It costs little and it keeps open source healthy.

Core contributor: Damian Tran &lt;[damian@gominerva.com](mailto:damian@gominerva.com)&gt;.
