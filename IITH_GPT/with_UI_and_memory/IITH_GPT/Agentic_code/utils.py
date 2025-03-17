import requests
from llama_index.core.llms import ChatMessage, MessageRole

# Define Gemini API endpoint and API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_API_KEY = "YOUR KEY"

def classify_query_with_gemini(query):
    """
    Classify a user query as 'summarization' or 'question_answering' using Google Gemini API.
    """
    # Construct the API URL with the key
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    # Create the request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"""
                        Classify the following query into one of these types:
                        - 'summarization'
                        - 'question_answering'
                        - 'search'
                        - 'fact_verification'
                        - 'exploration'
                        - 'math'

                        Query: {query}

                        Examples:
                        1. What is the capital of India?
                           Output: question_answering
                        2. Summarize the given paragraph.
                           Output: summarization
                        3. Find documents on climate change policies.
                           Output: search
                        4. Verify if the claim 'Earth is flat' is true.
                           Output: fact_verification
                        5. Explore the history of space exploration.
                           Output: exploration
                        6. 3+5*2
                           Output: math
                        7. add 3 and 5
                           Output: math
                        """
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make a POST request to the Gemini API
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Extract the classification result
        classification = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()
        # print(f"Predicted class: {classification}")
        return classification

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return "error"


def clean_latex_query_with_gemini(query):
    """
    Extracts only the relevant content from a user query for LaTeX conversion.
    Uses Google Gemini API to process the query and return the cleaned content.
    """
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    prompt = f"""
    Extract only the relevant content from the following query that needs to be converted into LaTeX.

    - Remove unnecessary words like 'Convert', 'Give me', 'Write this in LaTeX'.
    - Keep only mathematical expressions, equations, lists, tables, or structured text.
    - If the query contains a sentence, extract the main content that should be formatted.
    - Do NOT add extra words, just return the clean content.

    **Examples:**
    1. **Query:** "Convert a^2 + b^2 = c^2 into LaTeX"
       **Output:** "a^2 + b^2 = c^2"

    2. **Query:** "Give me the LaTeX code for a matrix: [[1,2],[3,4]]"
       **Output:** "[[1,2],[3,4]]"

    3. **Query:** "How to write an equation x^2 - 4x + 4 = 0 in LaTeX?"
       **Output:** "x^2 - 4x + 4 = 0"

    4. **Query:** "Write a table with columns Name, Age, Country and rows John, 25, USA; Alice, 30, UK"
       **Output:** "Name, Age, Country\\nJohn, 25, USA\\nAlice, 30, UK"

    5. **Query:** "List items: - Apple - Banana - Orange"
       **Output:** "- Apple\\n- Banana\\n- Orange"

    **User Query:** {query}

    Return only the cleaned content without any extra text.
    """

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract cleaned text safely
        cleaned_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        return cleaned_text  # Returns the relevant extracted text

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return "error"
     
def convert_text_to_markdown_with_gemini(plain_text):
    """
    Converts plain text into GitHub-flavored Markdown using the Gemini API.
    """
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    prompt = f"""
    Convert the following plain text into well-formatted GitHub-flavored Markdown.

    - Use headings (##, ###) for sections if the text contains clear section breaks.
    - Use **bold** for important terms if they stand out.
    - Use bullet points for lists if there are enumerations.
    - Use numbered lists if it's a sequence of steps.
    - Format code blocks using triple backticks (```) if code snippets are detected.
    - Maintain the meaning and logical flow of the text.

    **Plain Text Input:** 
    {plain_text}

    Return only the formatted Markdown without extra explanations or introductions.
    """

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract Markdown output safely
        markdown_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        return markdown_text  # Final Markdown response

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return "error"


def chunk_text(text, chunk_size=1000):
    """Split the text into smaller chunks if it's too long."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def gemini_summarize(text):
    """
    Summarize the given text using the Gemini API.
    The payload instructs the API to provide a concise summary of the provided text.
    """
    # Construct the API URL with your key
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    # Create the payload with the summarization instruction
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Please provide a concise summary of the following document:\n\n{text}"
                    }
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Extract the summary from the response
        summary = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        return summary
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return "error"

def summarize_text(text):
    """
    Use the Gemini API to summarize the provided text.
    For long texts, break them into chunks, summarize each, and join the results.
    """
    if len(text) <= 1000:
        return gemini_summarize(text)
    
    # Handle longer texts by splitting them into chunks
    chunks = chunk_text(text, chunk_size=1000)
    summaries = [gemini_summarize(chunk) for chunk in chunks]
    return " ".join(summaries)

import re

def do_math(expression: str) -> str:
    """
    Evaluates a mathematical expression following BODMAS rule.
    
    Parameters:
        expression (str): A string representing the math expression (e.g., '3 + 5 * 2')
    
    Returns:
        str: The result of the evaluation or an error message.
    """
    try:
        # Remove any unwanted characters (allow only numbers, operators, and spaces)
        sanitized_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression)
        
        # Evaluate the expression following BODMAS rules using Python's eval
        result = eval(sanitized_expr)
        
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Invalid expression ({str(e)})"


# JSON Schema for Ollama Integration
function_schema = {
    "type": "function",
    "function": {
        "name": "do_math",
        "description": "Evaluates mathematical expressions following BODMAS rules",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '3 + 5 * 2')"
                }
            },
            "required": ["expression"]
        }
    }
}

def execute_user_code(str):
    user_code = str

    # Optionally, prepare a restricted environment:
    safe_globals = {
        "__builtins__": {
            # You can selectively allow safe built-ins here.
            "print": print,
            # Add other safe functions as needed.
        }
    }
    
    try:
        # Execute the code in the restricted global environment.
        exec(user_code, safe_globals)
    except Exception as e:
        print("An error occurred:", e)