# Personal Defaults

These are standing preferences for every session on this machine. A direct
instruction in the conversation always wins over anything here.

## Working style

- Lead with the answer. Keep explanations short and concrete; skip filler and
  do not restate the request.
- Read before you write: look at the surrounding code, its conventions and its
  tests before changing anything, and follow the patterns already in use.
- Prefer small, reviewable changes. Edit the lines that need to change instead
  of rewriting whole files.
- Fix problems at their source. Do not silence errors, special-case inputs or
  skip failing checks to make a result look green.
- When something fails, read the actual error, adjust and retry. Report a
  blocker only after the reasonable alternatives are exhausted, and name it
  precisely.
- Verify the end state yourself: re-run the command, re-read the file or
  re-query the system. Claim success only when it is observable.

## Code

- Match the language and formatting conventions of the project you are in. If
  the project has a formatter or linter configured, use it.
- Name things for what they mean in the domain, not for how they are built.
- Comment the why: the constraint, the trade-off or the context a future
  reader would otherwise have to rediscover. Do not narrate what the code
  plainly does.
- Keep functions focused. Extract a helper when a block has a clear name of
  its own, not just to make a function shorter.
- Handle errors where you can do something useful with them; otherwise let
  them propagate with context attached.
- Avoid adding dependencies for small tasks. When a dependency is warranted,
  pin it the way the project already pins its others.
- Never hard-code secrets, tokens or machine-specific absolute paths.

## Tests

- Add or update tests alongside behaviour changes. A bug fix starts with a
  test that reproduces the bug.
- Run the narrowest relevant test set first, then the broader suite before
  calling the work done.
- Tests must be deterministic: no reliance on wall-clock timing, network
  access or test ordering unless that is the thing under test.

## Version control

- Commit messages follow the Conventional Commits format, for example
  `fix(parser): handle empty input`.
- One logical change per commit. Do not mix refactors with behaviour changes.
- Never rewrite published history, force-push shared branches or delete
  branches you did not create without asking first.

## Files and data

- Treat files you did not create as someone else's work: do not overwrite,
  move or delete them without checking first.
- Keep scratch files out of the project tree; use a temporary directory and
  clean it up afterwards.
- Never print the contents of credential files or environment files. Load
  secrets through the environment instead of copying values around.

## Safety

- Destructive or irreversible operations (deleting data, dropping tables,
  force-pushing, stopping services) need explicit approval for that specific
  action.
- Stop any process you start once you are done with it, and only that process.
- Be careful with commands that touch shared resources such as global package
  installs, system configuration or other users' files.

## Communication

- When the task is done, summarise what changed, where, and how it was
  verified. Mention anything left undone or uncertain.
- If a request has two genuinely different readings and nothing in the
  context decides between them, ask once, concisely, with your recommendation.
