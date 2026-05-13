---
name: commit
description: Commit staged/unstaged changes in logical groups with concise messages. TRIGGER when user says 'commit', 'let's commit', 'commit my changes', 'commit the changes', 'make a commit'. DO NOT TRIGGER when user is discussing what to change, reviewing diffs without intent to commit, or asking about git status only.
argument-hint: (no args needed)
---

## Step 1: Inspect

```bash
git status
git diff --stat
```

## Step 2: Assess and propose

**Batching:** group related changes. Don't split per-file unless topics genuinely diverge (different module, different concern).

**Hold suggestion:** if changes look incomplete (whitespace-only, partial feature, obvious WIP), suggest holding:

> "These changes look incomplete — want to hold off until <X> is also done?"

User can override.

**Propose the plan before doing anything:**

```
Proposed commits:

Group 1: [file1, file2, file3] — "feat: <message>"
Group 2: [file4]               — "test: <message>"

Confirm? (or edit groupings/messages)
```

Wait for confirmation.

## Step 3: Execute

For each group in order:
1. `git add <files>`
2. `git commit -m "<type>: <message>"`
3. Confirm success before the next group.

## Message format

`<type>: <short description>`

Types: `feat | fix | refactor | chore | test | docs | perf`

- 1 line preferred.
- 2–3 bullets in body only when genuinely needed for clarity.
- No Claude authorship signature.
- No verbose body unless context truly requires it.
- Reference PRD section when relevant: `feat: ingest layer per PRD §5a` is encouraged but not required.

**do not add claude signature in commit message.**

## Project-specific notes

- This repo has **no submodules**. Skip the submodule rules.
- `docs/planning/` is gitignored — it will never appear in `git status`. Ignore concerns there.
- The PRD itself lives in `docs/planning/` and is never committed.
