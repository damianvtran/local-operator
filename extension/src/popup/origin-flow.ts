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

export type OriginDecision = "once" | "always" | "deny";

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
 * check over "denied" would read as the wrong verdict. */
export function ackForDecision(decision: OriginDecision): DecisionAck {
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
  const standing =
    decision === "always" ? " Always-allowed sites can be taken back any time in Settings." : "";
  return {
    title: "Site allowed.",
    sub: `The agent is continuing.${standing}`,
    tone: "success",
    check: true,
  };
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
