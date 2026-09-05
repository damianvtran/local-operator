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

/** The decision this popup already made, keyed by ORIGIN — finding A6's rule:
 * a redirect chain resolves each hop independently, so a DIFFERENT pending
 * origin is a new prompt even while this one's ack is still settling. */
export interface DecidedOrigin {
  origin: string;
  decision: OriginDecision;
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
  // label stays "Allow once": it is the one-shot consent vocabulary the
  // store listing already promises, and the ack carries the nuance (n2).
  if (decision === "once") {
    return {
      title: "Site allowed.",
      sub: "The agent's next visit to this site goes through (once, within 10 minutes).",
      tone: "success",
      check: true,
    };
  }
  if (decision === "domain") {
    const standing =
      broadScope === "host"
        ? "Every port on this loopback host stays allowed"
        : "Every page on this domain and its subdomains, on any port, stays allowed";
    return {
      title: "Domain allowed.",
      sub: `The agent is continuing. ${standing}; take it back any time in Settings.`,
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
  options.push(
    { value: "site", label: "Only this site", detail: entry?.origin ?? "this exact site" },
    { value: "once", label: "Just this once", detail: "one navigation within 10 minutes" },
  );
  return { options, defaultValue: broad ? "domain" : "site" };
}

/** Which origin view a render shows. A pending origin equal to the one just
 * decided is the stale echo of the race above — hold the ack. A different
 * pending origin is a genuinely new prompt (A6). No pending origin means the
 * round-trip landed and the caller should clear its latch so a future prompt
 * for the SAME origin (e.g. a retry after deny) is not swallowed. */
export function originPromptView(
  pendingOrigin: string | undefined,
  decided: DecidedOrigin | null,
): "prompt" | "ack" | "none" {
  if (!pendingOrigin) return "none";
  return decided && decided.origin === pendingOrigin ? "ack" : "prompt";
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
