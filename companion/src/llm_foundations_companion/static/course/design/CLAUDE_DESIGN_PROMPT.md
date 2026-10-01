# Prompt for Claude Design

Redesign the visual experience and navigation for my new **LLM Foundations: build, measure, explain** course. I want an adult learning product that makes a difficult subject understandable, engaging, and easy to resume in short sessions.

Codex has rebuilt the curriculum, lesson text, exercises, answer explanations, and Python labs from scratch. Your responsibility is the learner-facing design, navigation, diagrams, and interactions. Use the supplied `llm-foundations-v2` package as the content authority. The original Grok-generated tutorial is reference material; do not copy its content defects or simply restyle its long pages.

## Read these inputs first

1. `START_HERE.md` and `curriculum.json` for the course sequence and outcomes.
2. `lessons/` for the 13 fully written lessons.
3. `labs/WORKBOOK.md` and the Python files for what actually runs.
4. `REVIEW.md` for the original course's strengths and problems.
5. `design/INTERACTION_CONTRACT.md`, `examples/`, and `VALIDATION.md` for real data and boundaries.

If the uploaded archive cannot be read in your environment, identify the specific required file and ask for it. Do not silently replace the course with invented material.

## The learning goal

The learner should be able to trace and demonstrate:

**text → tokenizer → token IDs → embeddings → attention and model → next-token loss → optimizer update → validation → checkpoint → generation.**

The crucial distinction is between training a tokenizer and training neural-network weights. The course also separates starting from random weights, resuming a checkpoint, fine-tuning, and inference. Preserve these distinctions in every visual and interaction.

## Experience and navigation

- Give the home screen one obvious “Start” or “Continue learning” action. Show the next lesson, why it matters, and a small progress summary. Put the complete roadmap behind a secondary action rather than making it the first task.
- Use a stable course outline with the current lesson visible. Support direct lesson links, previous/next navigation, search or glossary access, and a clear route back from labs.
- Present each lesson as a short sequence: outcome → explanation → worked example → predict/try → feedback → stopping point. Preserve deeper detail in optional expansions.
- Separate “read,” “practiced,” and “checked understanding.” Let me review freely; do not trap navigation behind quiz gates or confuse clicking complete with demonstrated skill.
- Include a personal experiment notebook with the workbook's fields. Save progress/notes locally and provide export/import so browser storage is not the only copy. Explain storage scope plainly.
- Give copyable commands an OS/environment selector where needed, show their working folder, and keep advanced settings out of the first-run path.

## Visual direction

Use a calm, precise technical-workshop aesthetic: excellent typography, comfortable line lengths, generous spacing, disciplined color, and diagrams that carry meaning. Make it feel like a thoughtful educational application rather than a marketing landing page. Avoid excessive badges, giant hero sections, endless card grids, neon decoration, and reward effects that interrupt learning.

Establish consistent visual identities for text, token IDs, vectors, model weights, predictions, targets, and measured results. Explain that color key once and reuse it. Support light and dark themes, narrow screens, keyboard operation, readable focus states, and reduced motion. Use explicit labels as well as color. Any animation should have a pause/step control and a static equivalent.

## Essential screens and interactions

1. Course home and roadmap with a clear continuation state.
2. A reusable lesson reader that renders all 13 supplied lessons and their answer explanations.
3. Tokenizer explorer: text, UTF-8 bytes, token IDs, learned merges, exact reconstructed text, and a visible round-trip check. Include newline and emoji examples. Label the built-in byte view and the supplied teaching BPE accurately.
4. Next-token exercise: align inputs and shifted targets, step through the causal mask, and explain why future tokens are hidden.
5. Model diagram: expose tensor shapes and let me connect vocabulary size, width, heads, and context to the relevant components.
6. Training/evaluation workspace: measured train and validation curves, a textual results table, the run identity, sample output, and a concise explanation of what the result supports.
7. Checkpoint journey: clearly show new training, continued training, and generation as different actions.
8. Capstone evidence checklist and rubric, with unresolved items visible.

## Truthful interaction states

There is no live training HTTP service in the new package. A browser prototype may use the supplied recorded metrics or labelled simulations. Do not show a fake “Start training” control as though it ran Python. Offer real local commands, import recorded results, or identify a future backend integration explicitly. Keep “Simulation,” “Recorded run,” and “Connected live run” visually distinct; the last requires a real verified connection.

Use the supplied measured example to build the results screen. Its weak repetitive output is useful teaching evidence; do not substitute a fluent invented sample or claim that the short run created a capable chatbot. Never compare raw perplexity across different tokenizers as a quality ranking.

Handle missing files, malformed imports, empty history, failed training, interrupted runs, stale data, and existing output directories. Do not convert failures into success-looking placeholders. Avoid displaying implementation internals in the main learning flow unless they help the learner decide what to do.

## Deliver

Build the working responsive interface using the capabilities available in your environment. Include the design system, navigation, all supplied lesson content, the key interactive examples, recorded-run view, and capstone. Keep the content separate from presentation so I can update lessons without redesigning screens. Preserve the Python lab behavior; identify any desired backend changes separately.

Verify the main learner journeys: first visit → first lesson → exercise feedback → saved progress → return and continue; tokenizer round trip; interpreting an imported run; and recording capstone evidence. Check desktop and narrow layouts, keyboard navigation, focus, contrast, and reduced motion. Report what you actually tested and what remains a prototype. Do not call the result finished based only on attractive screenshots.

Begin with a concise design direction and then produce the design and working prototype. Make reasonable reversible design choices; ask only for missing information that blocks progress.
