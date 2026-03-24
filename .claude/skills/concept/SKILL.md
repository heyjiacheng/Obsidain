---
name: concept
description: Use this skill when the user points out a concept in a paper note and asks to explain it as a standalone Obsidian note. Triggers on phrases like "explain this concept", "what is X", "create a concept note for X".
version: 1.0.0
---

# Concept Note Creator

When the user identifies a concept from a paper note, create a standalone explanation note and link it back.

## Inputs

- **Concept name**: The term or technique the user wants explained (e.g., "FiLM conditioning")
- **Source paper note** (optional): The paper note where the concept appears (e.g., `PhD-Research/Papers/OpenVlaOft.md`)

## Steps

1. **Read the source paper note** to understand how the concept is used in context.

2. **Research the concept**: Use web search or fetch the original paper/resource to get an accurate, thorough understanding of the concept.

3. **Read an existing concept note** from `PhD-Research/Concepts/` for style and structure reference.

4. **Derive the filename** from the concept name using PascalCase with hyphens between words (e.g., `FiLM-Conditioning.md`, `MLP-MaxPool-PointCloud-Encoder.md`).

5. **Write the concept note** to `PhD-Research/Concepts/<filename>.md` following this structure:

   ```markdown
   ---
   title: "<Concept Name>"
   tags:
     - concept
     - <relevant-domain-tags>
   aliases:
     - <common alternative names>
   ---

   # <Concept Name>

   > [!abstract] One-line summary
   > <One clear sentence explaining what this concept does.>

   ## The Problem It Solves
   <What gap or challenge does this concept address?>

   ## How It Works (Step by Step)
   <Clear explanation with diagrams/code blocks if helpful. Use math notation where appropriate.>

   ## In Practice
   <Concrete usage details, parameter values, or implementation notes.>

   ## Related
   - [[<Related concepts and papers>]]
   ```

   Adapt sections as needed — not every concept needs all sections. Add sections like "Why It Works", "Trade-offs", or "Analogy" when they help understanding.

6. **Link the concept in the source paper note**: If a source paper note was provided, add a `[[ConceptName]]` wikilink in the relevant location within that paper note (typically where the concept is mentioned or in the Connections section). Only add the link if it doesn't already exist.

## Obsidian Markdown Formatting

Apply the `obsidian:obsidian-markdown` skill when writing the note.
