/* Pure decisions for the popup's origin-consent flow, DOM-free for the
 * pure-node test suite — the same pattern as pair-flow.ts, for the same class
 * of bug.
 *
 * The race: clicking Allow/Deny sends `origin_decision` to the worker, but
 * session storage's `pendingOrigin` and /health's `pending_origin` keep
 * reporting the prompt until that round-trip lands. A render() driven by
 * either source would leave the prompt on screen with three live buttons and
 * no sign the click took effect. The acknowledgement must therefore render
 * from the CLICK alone, and a stale prompt echo must hold the ack rather than
 * resurrect the buttons. */

import type { AccessQueueEntry, OriginDecision } from "../access-queue";
import type { BroadGrant } from "../origin-policy";

export type { OriginDecision } from "../access-queue";

/** The decision this popup already made. Keyed by origin AND by the
 * generation (entry id) it was made against.
 *
 * Origin alone is not enough: two live entries can share an origin (dedupe is
 * origin+requester), and `once`/`deny` resolve only the selected entry, so a
 * sibling survives the click and keeps the origin pending. An origin-only
 * latch reads that survivor as its own stale echo and strands the popup on
 * "Site allowed." over a live request the user cannot act on (A6/U7). A
 * `once` grant is spent on one navigation, so the same origin re-prompting is
 * the DESIGNED behaviour, not an edge case. */
export interface DecidedOrigin {
  origin: string;
  decision: OriginDecision;
  /** The entry id the decision answered. Empty on the /health-only fallback,
   * which has no generation to compare and falls back to origin equality. */
  entryId: string;
}

export interface DecisionAck {
  title: string;
  sub: string;
  tone: "success" | "neutral";
  check: boolean;
}

/** The acknowledgement each decision renders. Deny is a COMPLETED choice, not
 * a failure, so it takes the neutral register and no check — danger is
 * reserved for states the user must recover from (error/incompatible), and a
 * check over "denied" would read as the wrong verdict. `broadScope` names
 * which broad grant a `domain` decision wrote so the ack says exactly what
 * now stays allowed. */
export function ackForDecision(decision: OriginDecision, broadScope?: BroadGrant["scope"]): DecisionAck {
  if (decision === "deny") {
    return {
      title: "Site denied.",
      sub: "The agent won't use this site.",
      tone: "neutral",
      check: false,
    };
  }
  // "once" is a standing grant for the agent's NEXT navigation, not an
  // in-flight pass: in the async approval flow the navigation happens a turn
  // or two after the click, so the ack names what the grant actually covers
  // and its bound (10 min unconsumed — see ONCE_GRANT_TTL_MS). The button
  // option label stays "Just this once": it is the one-shot consent
  // vocabulary the store listing promises, and the ack carries the nuance (n2).
  if (decision === "once") {
    return {
      title: "Site allowed.",
      sub: "The agent's next visit to this site goes through (once, within 10 minutes).",
      tone: "success",
      check: true,
    };
  }
  if (decision === "domain") {
    // Both broad scopes share the wire value "domain", but a loopback grant is
    // a HOST grant: titling it "Domain allowed." names a scope the user was
    // never offered and contradicts its own body copy (D1).
    if (broadScope === "host") {
      return {
        title: "Host allowed.",
        sub: "The agent is continuing. Every port on this host stays allowed; take it back any time in Settings.",
        tone: "success",
        check: true,
      };
    }
    return {
      title: "Domain allowed.",
      sub: "The agent is continuing. Every page on this domain and its subdomains, on any port, stays allowed; take it back any time in Settings.",
      tone: "success",
      check: true,
    };
  }
  return {
    title: "Site allowed.",
    sub: "The agent is continuing. This exact site (address and port) stays allowed; take it back any time in Settings.",
    tone: "success",
    check: true,
  };
}

export interface ScopeOption {
  value: Exclude<OriginDecision, "deny">;
  label: string;
  /** What the option grants, rendered as data in the trough under the select. */
  detail: string;
}

export interface ScopeOptions {
  options: ScopeOption[];
  defaultValue: ScopeOption["value"];
}

/** The Allow dropdown's options for a prompt, derived from the ENTRY alone.
 * The broad (domain/host) option appears only when the worker stamped
 * `broad` on the entry, and is then the default: it is the grant most
 * approvals actually want (the whole product, not one subdomain and port).
 * An entry without it (an IP literal, a bare public suffix, an entry from
 * 0.1.7, or a /health-only render with no entry) falls back to "Only this
 * site" as the default, which is the fail-closed choice. */
export function scopeOptions(entry: Pick<AccessQueueEntry, "origin" | "broad"> | undefined): ScopeOptions {
  const options: ScopeOption[] = [];
  const broad = entry?.broad;
  if (broad) {
    options.push(
      broad.scope === "host"
        ? { value: "domain", label: "Any port on this host", detail: `${broad.key}, any port` }
        : {
            value: "domain",
            label: "All pages on this domain",
            detail: `${broad.key} and every subdomain, any port`,
          },
    );
  }
  const origin = entry?.origin;
  options.push(
    { value: "site", label: "Only this site", detail: origin ?? "this exact site" },
    {
      value: "once",
      label: "Just this once",
      // Every other detail line names the subject as data; a bare quantity
      // would make this the one option whose trough stops saying what is
      // being granted (U6).
      detail: origin
        ? `one navigation to ${origin}, within 10 minutes`
        : "one navigation within 10 minutes",
    },
  );
  return { options, defaultValue: broad ? "domain" : "site" };
}

/** Which origin view a render shows.
 *
 * The ack is held only while the pending entry IS the generation that was
 * decided. A different pending origin, or the SAME origin under a different
 * generation, is a genuinely new prompt and must be shown: holding it strands
 * a live request behind a confident "Site allowed." (A6/U7). No pending
 * origin means the round-trip landed and the caller should clear its latch so
 * a future prompt for the same origin is not swallowed.
 *
 * The /health-only fallback carries no entry id. There is no generation to
 * compare there, so both sides being empty falls back to origin equality,
 * which is exactly the U1 behaviour that path had before. */
export function originPromptView(
  pendingOrigin: string | undefined,
  decided: DecidedOrigin | null,
  pendingEntryId = "",
): "prompt" | "ack" | "none" {
  if (!pendingOrigin) return "none";
  if (!decided || decided.origin !== pendingOrigin) return "prompt";
  // Normalize both sides: a missing id and an empty id are the same "no
  // generation to compare" state, and they must not read as a mismatch, or
  // the /health-only path would prompt over its own ack and reopen U1.
  return (decided.entryId ?? "") === (pendingEntryId ?? "") ? "ack" : "prompt";
}

/** The identity the scope select's option list is cached against.
 *
 * The entry id alone collides on the /health-only fallback, where it is the
 * empty string for EVERY origin: two successive fallback renders for
 * different sites then reuse one option list, so the trough names the
 * previous origin while Allow grants the current one (A7). That defeats the
 * trough's whole purpose, which is letting the user verify the authority as
 * data before committing. Falling back to the origin keeps the key
 * non-empty and unique per site. */
export function scopeLatchKey(entryId: string, pendingOrigin: string | undefined): string {
  return entryId || (pendingOrigin ? `origin:${pendingOrigin}` : "");
}

export interface OriginNotice {
  title: string;
  sub: string;
}

/** The notice a REJECTED decision shows, or null when none should appear.
 *
 * A rejection with an EMPTY shown prompt id came from a /health-fallback
 * render (worker restarted, queue storage empty, or the entry expired while
 * the daemon still echoed it) — the request was never "replaced", the popup
 * simply had no generation to aim at. resolveOrigin's origin fallback and the
 * daemon's cleared-event reconciliation retry or retire that render, so a
 * scary "Request changed." interstitial there was pure noise — and looping it
 * every click was the reported bug.
 *
 * A rejection WITH a shown id is a real miss: the generation the buttons were
 * drawn for is gone. If the origin is still pending under a NEWER generation
 * the prompt was replaced; if the live queue holds no entry for the origin at
 * all, the request expired or was cancelled. */
export function noticeForRejectedDecision(
  shownPromptId: string,
  originStillPending: boolean,
): OriginNotice | null {
  if (!shownPromptId) return null;
  if (originStillPending) {
    return {
      title: "Request changed.",
      sub: "The site request was replaced while this window was open. Review the new request.",
    };
  }
  // D1/D2 (design round 1): state the CONSEQUENCE (nothing was granted or
  // denied) instead of restating the title's cause, so a user who just
  // clicked Allow knows the click had no effect.
  return {
    title: "Request expired.",
    sub: "It timed out or was cancelled, so nothing was granted or denied.",
  };
}
