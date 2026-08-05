This is a new instruction from me through the Local Operator UI using the inline edit feature.  Please stop your current task and pay attention to this new instruction.

Please make edits in the following file based on the edit prompt and selection hint from the file content provided.

File path:
<file_path>
{{file_path}}
</file_path>

Edit instruction:
<edit_prompt>
{{edit_prompt}}
</edit_prompt>

Selection from file that I'm referring to:
<selection>
{{selection}}
</selection>

Current file content:
<file_content>
{{file_content}}
</file_content>

The selection is the text that I want to edit in the file and/or the text that I am referring to in the edit_prompt. Review the text against the actual file_content above and find it, then use the matching content in the SEARCH block for your EDIT action. Make sure the content of the SEARCH block exactly matches some text in the file_content, it does not need to match the selection if the selection is different (this can be because of formatting issues from the UI or other input source).

CRITICAL: When creating SEARCH blocks, you must copy the text EXACTLY as it appears in the file_content, including:
- All whitespace characters (spaces, tabs, newlines)
- All punctuation and special characters
- All formatting (markdown, HTML, etc.)
- Line breaks and paragraph spacing
- Bullet points, dashes, and list formatting

If you need to edit multiple instances of the same text, then be specific about the surrounding selection to deduplicate or otherwise provide multiple identical search queries to replace each in order. Focus carefully on the selection that I have provided and make sure to be precise, only applying changes that are relevant to the selection and not the entire file, unless the selection encompasses the entire file.

If the selection is empty, then it means that I have not selected any text and have just placed my cursor at a particular position and asked you to do an edit instruction. Often this means that an insertion is required at the cursor position.

Here are the guidelines for the EDIT action for your reference:

#### Example for EDIT:

<example>
Editing a file to update or add content.

<action_response>
<action>EDIT</action>

<learnings>
I learned about this new content that I found from the web.  It will be useful for the user to know this because of x reason.
</learnings>

<file_path>
existing_file.txt
</file_path>

<replacements>
<<<<<<< SEARCH
original text 1

original text 2
=======
text to replace 1

text to replace 2
>>>>>>> REPLACE

<<<<<<< SEARCH
original text 3
=======
text to replace 3
>>>>>>> REPLACE
</replacements>
</action_response>
</example>

EDIT usage guidelines:
- MOST IMPORTANT: The SEARCH blocks must contain EXACT and PRECISE text that exists in the file_content. Copy the text character-for-character, including all whitespace, line breaks, markdown formatting, HTML tags, and special characters. Even a single missing space or newline will cause the edit to fail.
- Before creating a SEARCH block, locate the exact text in the file_content and copy it precisely. Pay special attention to:
  * Line breaks and paragraph spacing
  * Indentation (spaces vs tabs)
  * Bullet points and list formatting (-, *, numbers)
  * Markdown formatting (**bold**, *italic*, headers ##)
  * Punctuation and special characters
- You can make multiple edits in one response by using multiple SEARCH/REPLACE blocks. Each edit will apply to the first instance of the search text in the file.
- If you need to edit multiple instances of the same text, then be specific about the surrounding selection to deduplicate or otherwise provide multiple identical search queries to replace each in order.
- Make sure that you include the replacements in the "replacements" field or you will run into parsing errors.
- Do not duplicate headers in the replacements. If you are replacing placeholders, make sure that you pay attention to the other text around the placeholder and don't repeat content that is already present in the file.
- If you have enough information to make multiple edits in a file at a time, you should do so instead of editing one at a time to be less expensive and more efficient.
- Be careful and precise with edits. You can edit multiple times in one action, but if you are running into errors then you may need to break it up into separate edit actions, one section at a time. If you keep running into errors, then you may need to rewrite the file from scratch instead with a WRITE action.

Please provide your EDIT action response with SEARCH and REPLACE blocks to make the necessary changes. You MUST use only the EDIT action in response to this message.
