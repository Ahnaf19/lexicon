---
description: Trigger a multi-engineer code review of recent changes. Invokes data-engineer, ml-engineer, qa-engineer in parallel, returns consolidated findings.
---

Review the recent work (most recent commit through current uncommitted changes) by launching the data-engineer, ml-engineer, and qa-engineer agents in parallel. Each should focus on their declared scope. Consolidate findings into a single triage table grouped by severity (Phase-blocking / High / Medium / Low / PoC-OK) with file:line references.
