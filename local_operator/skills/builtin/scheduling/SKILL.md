---
name: scheduling
description: "Scheduling across time zones: confirm times in the attendee's own zone, use IANA zone names, mind DST edges, buffers and calendar etiquette. Use when booking meetings or calls with people in other zones."
---

# Scheduling

Times for other people: zones named exactly, ambiguity engineered out, confirmations in the frame the attendee reads.

## Zones

- Use IANA zone names (America/New_York, Europe/Berlin, Asia/Kolkata), never abbreviations. "ET" is ambiguous, "CST" means several different zones, "IST" at least three.
- Keep the zone ID attached to the time; a bare "14:00" is not schedulable.
- When converting, show the basis: "14:00 UTC = 10:00 New York (EDT)".

## DST edges

- Transitions move a zone by an hour. Check the UTC offset for the DATE in question, not for today.
- Fall-back repeats an hour: 01:30 can occur twice. If a proposed time lands there, name the offset (01:30 EDT versus EST).
- Spring-forward skips an hour: a time in the gap does not exist. Never propose one.
- Recurring meetings: keep recurrences zoned (with the zone ID), so they stay at the local wall time. A fixed-UTC recurrence drifts an hour twice a year.

## Restate in the recipient's zone

- Every proposal and confirmation gives the time in the OTHER person's zone, plus yours, plus the zone IDs: "10:00-10:30 Europe/London (11:00 for you, Europe/Berlin)".
- Avoid relative-only phrasing: "3pm my time" makes the reader do the conversion, and they will get it wrong when tired.

## Working hours and buffers

- Judge the time against the recipient's working hours and week boundaries, not yours. A slot outside their 9-18 needs a stated reason.
- Leave buffers where the recipients' culture expects them; back-to-back chains across zones are a smell.
- Respect stated focus blocks and known out-of-hours patterns.

## Confirm, then book

- A proposed slot is not a booking. Restate date, time, zone, and duration; get explicit confirmation before creating the invitation.
- The confirmation restates all four fields. Typos and DST errors are caught here or not at all.
- Change any field and it is a new proposal: reconfirm the whole thing.

## Calendar etiquette

- No surprise short-notice meetings where avoidable; mark urgency only when it is real.
- Public holidays differ by country and region; check the attendee's region, not the country's default.

## Red flags

- "3pm ET" in a proposal or invitation.
- Converting only one side of the pair.
- A proposed time inside a DST transition window.
- Weekend or holiday slots chosen because the zone check stopped at the clock.
- Booking on a proposal that was never confirmed.
