---
name: paper
description: Use this skill when the user provides an arXiv URL and asks to summarize. Triggers on phrases like "summarize this paper", "add to Obsidian", "paper notes", or when an arxiv.org URL is provided.
version: 1.0.0
---

# arXiv Paper to Obsidian Note

When the user provides an arXiv URL, follow these steps:

## Steps

1. Edit provided URL (replace pdf or abs with html in the URL)

   - If edited URL exist: Apply the `defuddle` skill with edited URL, don't add additional parameter (like head).
   - If edited URL not exist: curl to get entire pdf with original URL

2. Read the note template from:
   `PhD-Research/Templates/paper_template.md`

3. Derive the output filename from the paper title using PascalCase, and make it simple.

4. Write the completed note to:
   `PhD-Research/Papers/<filename>.md`

5. refine this markdown note.

## Obsidian Markdown Formatting

Apply the `obsidian:obsidian-markdown` skill when writing the note:

