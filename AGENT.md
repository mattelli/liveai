# AGENT.md — Coding AI Agent Rules

## Startup

1. On startup, get the current OS.
2. Get the current working directory.

## Goals

3. Ask the user for one or more goals.

## Planning

4. Provide a plan to achieve the goals. Always ask questions to the user whenever
   clarification is needed — do not guess.

5. Ask the user whether they need a written plan.

   **If YES**, the written plan must include:
   - Goals
   - Current behavior
   - Expected behavior
   - Verification of the implementation with unit tests

   In the plan, split the implementation into simple tasks, and specify that
   **one commit shall be done per task**.
   **proceed one task at the time and commit, ask confirmation before start with the next task**.
   **detail what has been done in each commit message**.

   **If NO**, always ask the user whether the changes can be committed.

## Processes

6. Always ask the user before spawning any process.
