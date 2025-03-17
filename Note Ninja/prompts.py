from llama_index.core.llms import ChatMessage, MessageRole
from datetime import datetime

agent_prompt = """
            You are an intelligent AI assistant helping the user analyze an uploaded document.
            The user has uploaded a document, and relevant information has been extracted as follows:
            ---
            Extracted Context:
            {context}
            ---
            Your priority is to:
            1. Use the extracted context as the ground truth for answering.
            2. Do not speculate or assume information outside the provided context.
            3. If the context is insufficient, then make use of your tools, search the internet and find an answer if available.
            4. You have also been provided with a code execution tool, so you can execute Python code written as a string.

            User Query:
            {recontextualized_query}
            """

flowchart_prompt = """
You are a highly intelligent assistant specializing in analyzing flowcharts and providing meaningful insights from structured information. You will be given a textual representation of flowcharts extracted from multiple pages of a PDF. Each flowchart consists of:

Nodes: Representing key elements or steps in a process. Each node has a unique ID and descriptive text.
Edges: Representing directional relationships or transitions between nodes (e.g., from one step to another).
Annotations: Additional notes or context relevant to the page.
Your task is to:

Summarize the overall purpose of the flowcharts, based on the nodes and annotations.
Identify key processes or decision points, and explain their significance.
Highlight any dependencies or critical paths within the flowcharts by analyzing the edges.
Detect any ambiguities, missing steps, or logical inconsistencies (e.g., disconnected nodes or missing transitions).
If the flowcharts represent multiple processes or workflows, summarize each one separately and describe how they might be interconnected.

Below are the flowcharts:
{flowchart_str}
"""

reformulation_type_prompt = """You are an intelligent AI assistant. The current date in YYYY-MM-DD format is {current_date}. You will be provided with:
1. Chat history between the user and the chatbot.
2. A user-uploaded document, which might be a summary of a larger document rather than the complete document.
3. A user query.

Your task is to determine the query type and generate an output according to the following rules:
- If the query is chit-chat or conversational in nature (e.g., greetings like 'hello', 'hi', or expressions of gratitude like 'thanks') and doesn't require any particular information to be answered, respond with query type as 'general' and output as a polite response to the query.
- Queries that can be answered with a bit of humor or logic and don't need information also come under the above category.
- If the document or chat history is sufficiently relevant to the query (not necessarily complete), respond with query type as 'direct' and output as the reformulated query based on the chat history and/or document.
- If the query requires additional context not sufficiently present in the chat history or document, respond with query type as 'context' and output as the reformulated query.
- If the query is completely unrelated to the chat history and document, respond with query type as 'context' and output as the grammatically corrected version of the query.

**Note:**
- The reformulated query should always be grammatically correct and contextually rich.
- Only provide the Query Type and Output. Do not provide any other explanation or response.

### Examples

**Example 1:**
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Query: Tell me more abt him
User Uploaded Document:

Query Type: context
Output: Tell me more about Isaac Newton who discovered the laws of motion.

**Example 2:**
Chat History:

User Uploaded Document:

User Query: How are you?
Query Type: general
Output: I am doing well, thank you for asking. How can I help you today?

**Example 3:**
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Uploaded Document:

User Query: Waht is the capital of France?
Query Type: context
Output: What is the capital of France?

**Example 4:**
Chat History:

User Uploaded Document:
The document is a financial report of Google Inc.
User Query: What is the revenue in last quarter?
Query Type: direct
Output: What is the revenue of Google Inc. in the last quarter?

**Example 5:**
Chat History:

User Uploaded Document:
The Super Bowl is the annual league championship game of the National Football League (NFL) of the United States.
User Query: When is the next super bowl happening?
Query Type: context
Output: When is the next Super Bowl annual league championship happening in United States?

Chat History:
{chat_history}
User Uploaded Document:
{document}
User Query: {user_query}
"""

def recontextualize_query(model, user_query, conversation_memory, extracted_text=''):
    # Prepare context from history and retrieved documents
    history_context = "\n".join([f"{role.capitalize()}: {content}" for message in conversation_memory for role, content in message.items()])
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = reformulation_type_prompt.format(chat_history=history_context, document=extracted_text if extracted_text.strip() else "No content extracted or uploaded.", user_query=user_query, current_date=current_date)
    
    response = model.chat([ChatMessage(content=prompt, role=MessageRole.USER)]) # Generate recontextualized query
    recontextualized_query = response.message.content

    return recontextualized_query

def summarize_flowcharts(model, flowchart_str):
    response = model.chat([ChatMessage(content=flowchart_prompt.format(flowchart_str=flowchart_str), role=MessageRole.USER)])

    return response.message.content