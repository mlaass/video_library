import os
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv


def load_openrouter_key():
    """
    Loads the OpenRouter API key from .env file.
    Returns the API key or None if not found.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        print("Please create a .env file with your OpenRouter API key:")
        print("OPENROUTER_API_KEY=your_api_key_here")
        return None

    return api_key


def generate_summary_and_title(transcript_content, api_key, model="anthropic/claude-3.5-sonnet"):
    """
    Generates a title and summary for the transcript using OpenRouter API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",  # Optional: for analytics
        "X-Title": "Transcript Summarizer",  # Optional: for analytics
    }

    prompt = f"""Please analyze the following transcript and provide:
1. A concise, descriptive title (max 80 characters)
2. A brief summary (2-3 sentences) highlighting the main topics and key points

Transcript:
{transcript_content}

Please format your response as:
TITLE: [your title here]
SUMMARY: [your summary here]"""

    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.7}

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse the response to extract title and summary
        lines = content.strip().split("\n")
        title = ""
        summary = ""

        for line in lines:
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()

        return title, summary

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        return None, None
    except KeyError as e:
        print(f"Error parsing API response: {e}")
        return None, None


def process_transcript_file(transcript_path, api_key, model):
    """
    Processes a single transcript file by adding title and summary to the top.
    """
    if not transcript_path.is_file():
        print(f"Error: Transcript file not found at {transcript_path}")
        return

    print(f"\nProcessing: {transcript_path.name}")

    try:
        # Read the original transcript content
        with open(transcript_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Check if the file already has a title and summary
        if original_content.startswith("TITLE:") or original_content.startswith("# "):
            print(f"File {transcript_path.name} already appears to have a title/summary. Skipping.")
            return

        # Generate title and summary
        print("Generating title and summary...")
        title, summary = generate_summary_and_title(original_content, api_key, model)

        if title and summary:
            # Create the header content
            header = f"""# {title}

## Summary
{summary}

---

"""

            # Write the new content with header + original content
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(header + original_content)

            print(f"✅ Successfully processed {transcript_path.name}")
            print(f"   Title: {title}")
            print(f"   Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
        else:
            print(f"❌ Failed to generate title and summary for {transcript_path.name}")

    except Exception as e:
        print(f"An unexpected error occurred while processing {transcript_path.name}: {e}")


def main():
    """
    Main function to parse arguments and process transcript files.
    """
    parser = argparse.ArgumentParser(
        description="Add AI-generated titles and summaries to transcript files using OpenRouter."
    )
    parser.add_argument(
        "--input_folder", type=str, required=True, help="Path to the folder containing transcript files."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="OpenRouter model to use (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4).",
    )

    args = parser.parse_args()

    input_path = Path(args.input_folder)

    # Validate input directory
    if not input_path.is_dir():
        print(f"Error: Input folder not found at {input_path}")
        return

    # Load OpenRouter API key
    api_key = load_openrouter_key()
    if api_key is None:
        return

    print(f"Using model: {args.model}")

    # Find all transcript files
    transcript_files = list(input_path.glob("*_transcript.md"))

    if not transcript_files:
        print(f"No transcript files (*_transcript.md) found in {input_path}")
        return

    print(f"\nFound {len(transcript_files)} transcript file(s) to process:")
    for file_path in transcript_files:
        print(f"  - {file_path.name}")

    # Process each transcript file
    print(f"\nStarting summarization process for transcripts in: {input_path}")
    for transcript_path in transcript_files:
        process_transcript_file(transcript_path, api_key, args.model)

    print("\n✅ All transcript files have been processed.")


if __name__ == "__main__":
    main()
