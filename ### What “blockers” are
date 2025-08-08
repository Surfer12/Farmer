### What “blockers” are
- **Definition**: Anything that prevents safe, high-confidence progress right now. In my flow, blockers are conditions that push Ψ below the action gates or make an action unsafe/irreversible without more info.

### Common blocker categories
- **Access/permissions**: Missing repo, env, secret, or API scopes.
- **Ambiguity/missing decisions**: Unclear requirements, acceptance criteria, or user preferences (e.g., library choice).
- **Conflicting/weak evidence**: Internal vs. external signals disagree (low α-fit, high authority/verifiability risk), no canonical source.
- **Build/test instability**: Failing or flaky tests, red CI, broken build.
- **Environment/tooling**: Version mismatches, missing dependencies, incompatible runtimes.
- **Destructive risk**: Changes that could lose data or disrupt prod without backup/rollback.
- **External dependencies**: Waiting on API quotas, vendor outage, PR review, third-party schema.
- **Policy/compliance**: Licensing, data handling, security constraints.
- **Performance/resources**: Operation would exceed limits or require long interactive sessions.
- **Interactivity required**: Command needs TTY or prompts (I run non-interactively by default).

### How I detect and react
- **Detection**: Low Ψ from high \(R_a/R_v\), failing lints/tests/build, missing canonical URLs, or unresolved choices.
- **Action**:
  - Gather evidence (prefer canonical), lower α toward N when sources appear.
  - Use dry-runs, non-interactive flags, and safe defaults.
  - If still blocked, ask a precise, minimal question or propose 1–2 clear options.

### Examples
- Need a choice: “ORM: Prisma vs. Drizzle?” → I propose tradeoffs; proceed after your pick.
- Secrets missing: “DATABASE_URL not set” → Request value or suggest .env setup.
- Red tests: I triage failing specs, isolate regressions, and fix before new work.
- Destructive migration: I gate until backup/rollback plan is confirmed.

### What unblocks me fastest
- Links to canonical specs/issue threads, your preference on key choices, required secrets/permissions, and acceptance criteria.

- Defined blockers as conditions preventing safe, high-confidence progress.
- Listed practical categories, detection via Ψ and risk signals, and how I resolve or escalate.
- Noted what inputs from you remove blockers quickest.