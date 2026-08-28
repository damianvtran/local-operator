# Privacy Policy for the Local Operator Browser Extension

**Effective date:** August 25, 2026

**Publisher:** Radient, Inc.

This is the source copy for a public, stable privacy-policy page. Host it at
`https://local-operator.com/browser-extension/privacy` before submitting the
extension. **Assumption:** the product and design documents do not reserve a
URL; this path is proposed because it names the product plainly. Replace the
URL consistently in the manifest, Chrome Web Store dashboard, and listing if
another route is chosen.

## The short version

The Local Operator browser extension connects the Local Operator app on this
computer to this browser. The extension does not send browser data to
Radient, Inc. or any other remote service. It does not run analytics, show
ads, or sell data.

The extension sends page content, accessibility information, screenshots,
and browser-action results only to the Local Operator browser bridge on
`127.0.0.1`; that extension-to-app connection never leaves the device. The
separate Local Operator app may send browser results needed for your task to
the AI model you choose, under the app privacy policy and that model
provider's terms. Choose a local model to keep that processing on-device too.

## What the extension accesses

The extension accesses a browser tab that Local Operator creates for the
agent. It does not take over the tab you are using.

On sites that you approve, the extension may:

- read the page URL and title;
- read visible page text and accessibility information;
- take screenshots of the delegated tab;
- click page controls and type into fields;
- navigate the delegated tab to a page; and
- observe navigation completion and redirects so it can report where the tab
  actually landed.

The extension asks before the agent uses a new website. You can allow the
site once, always allow it, or deny it. A site's normal subresources may load
from other domains; the approval controls where the delegated tab navigates,
not every subresource requested by the page.

## What is stored in the browser

The extension stores only the state needed to connect securely and remember
your choices:

- a long-lived pairing token issued by the browser bridge;
- the local bridge port if you changed it from the default;
- your per-site allow and deny choices; and
- temporary session state for the agent-owned tab, including its tab handle,
  a random nonce, page-element references, and any pending site prompt.

The pairing token, bridge port, and site list are stored in
`chrome.storage.local` in your browser profile. Temporary tab state and the bounded site-approval queue, including decisions
and requester-bound one-time grants, are stored in `chrome.storage.session` and
end with the browser session. Requester identifiers are never shown in the popup
or system notifications. The local
bridge stores only a SHA-256 hash of the pairing token and the paired
extension ID in a user-only file on your computer.

You can remove stored site choices and unpair the extension from its settings.
Removing the extension also removes its Chrome storage.

## What is transmitted

The extension makes one connection to the Local Operator browser bridge at
`127.0.0.1` on this computer. Through that local connection it receives
instructions such as navigate, read, click, type, and screenshot. It returns
the requested page content or action result through the same local
connection.

The extension itself transmits nothing off the device. It does not connect to
Radient, Inc. or any third-party analytics, advertising, telemetry, or cloud
browser service.

The Local Operator app is separate from this extension. The text of your
conversation, including browser results the agent needs in order to answer
you, may be sent by the app to the AI model you choose. You can instead choose
a model running locally. That processing is governed by the Local Operator
app's privacy policy and the terms of your chosen model provider. The
extension does not choose the model provider and does not contact it directly.

## Pairing and access controls

The extension pairs only after you enter a short code shown by Local Operator
on the same computer. It then accepts instructions only from the paired local
bridge. The bridge accepts extension connections only from the paired
extension ID.

Site access is denied by default. The extension asks in browser UI before the
agent navigates to a site you have not approved. Saved access is exact-origin
by default; literal loopback hosts may receive an explicit same-scheme all-port
grant. You can review and remove each site-access grant independently in the
extension settings and unpair the browser at any time.

## Analytics, advertising, and sale of data

The extension contains no analytics or advertising. It does not collect
usage metrics or crash reports. Radient, Inc. does not sell, rent, license, or
transfer data accessed by the extension because the extension does not send
that data to Radient, Inc.

The extension does not use or transfer browser data for creditworthiness,
lending, advertising, or any purpose unrelated to its single purpose of
connecting your own Local Operator app to your browser.

## Data retention

Radient, Inc. receives no data from the extension and therefore retains none.
Browser-local settings remain in your Chrome profile until you remove them,
unpair, clear extension storage, or uninstall the extension. Temporary
session state expires with the browser session.

## Children

The extension is a general-purpose productivity tool and is not directed to
children under 13. Because it sends no data to Radient, Inc., Radient, Inc.
does not knowingly collect personal information from children through the
extension.

## Changes to this policy

If the extension's data practices change, this page will be updated before a
new version using those practices is released. The effective date above will
also be updated. Material changes will be described in the extension's store
listing or release notes.

## Contact

Questions about this extension or this policy can be sent to:

**Radient, Inc.**

Email: **damian@radienthq.com**

Website: https://local-operator.com
