# Third-party notices
This kit's original analysis/scripts use the repository MIT license (LICENSE).
Public tool schemas are captured from the following pinned OSS packages.
The full upstream license texts are retained under licenses/. Source links,
package commands and per-schema hashes are also in public_loadout/public_mcp_schemas.json.
No MCP server binaries or transitive dependency trees are redistributed.

| Server | Package pin | License | Full text |
|---|---|---|---|
| [twilio](https://github.com/twilio-labs/mcp) | `@twilio-alpha/mcp@0.7.0` | MIT | [text](licenses/twilio.txt) |
| [grafana](https://github.com/grafana/mcp-grafana) | `mcp-grafana==2.2.0` | Apache-2.0 | [text](licenses/grafana.txt) |
| [aws-billing-cost-management](https://github.com/awslabs/mcp) | `awslabs.billing-cost-management-mcp-server==0.0.44` | Apache-2.0 | [text](licenses/aws-billing-cost-management.txt) |
| [supabase](https://github.com/supabase-community/supabase-mcp) | `@supabase/mcp-server-supabase@0.15.0` | Apache-2.0 | [text](licenses/supabase.txt) |
| [launchdarkly](https://github.com/launchdarkly/mcp-server) | `@launchdarkly/mcp-server@0.7.2` | MIT | [text](licenses/launchdarkly.txt) |
| [playwright](https://github.com/microsoft/playwright-mcp) | `@playwright/mcp@0.0.84` | Apache-2.0 | [text](licenses/playwright.txt) |
| [notion](https://github.com/makenotion/notion-mcp-server) | `@notionhq/notion-mcp-server@2.6.1` | MIT | [text](licenses/notion.txt) |
| [kubernetes](https://github.com/Flux159/mcp-server-kubernetes) | `mcp-server-kubernetes@4.2.1` | MIT | [text](licenses/kubernetes.txt) |
| [heroku](https://github.com/heroku/heroku-mcp-server) | `@heroku/mcp-server@1.2.13` | Apache-2.0 | [text](licenses/heroku.txt) |

Upstream license texts were retrieved 2026-11-03 from the documented source
repositories; exact download URLs and hashes are in licenses/sources.json.
Tool-schema texts are unmodified. Dummy capture credentials and generic
example email addresses are upstream examples, not private accounts.

Skills are **not redistributed**. fetch_public_skills.sh fetches upstream
commit ca45fc496a3436251f191354091d557da31c15b5 and preserves licenses.
Non-document skills are Apache-2.0 per upstream; docx/pdf/pptx/xlsx are
proprietary source-available (individual LICENSE.txt), not OSS. Review those
terms before fetching/use. Public neutral instructions are original kit text.

Matplotlib/numpy and their dependencies are installed separately, not bundled
with this kit. Their distributions carry their own third-party notices.
No proprietary office fixtures, internal schemas, private skills or raw model
request bodies are included. This is attribution/provenance, not legal advice.
