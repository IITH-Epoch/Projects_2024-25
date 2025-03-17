from llama_index.core.tools import FunctionTool
from tavily import TavilyClient
import io
import math
import os
import random
import contextlib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tavily_api_key = "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"
tavily_cli = TavilyClient(api_key=tavily_api_key)

model_name = "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def web_search(query: str) -> str:
    """Function to search the web and obtain information using a search query"""
    results = tavily_cli.search(query=query)
    return results

def mul_integers(a: int, b: int) -> int:
    """Function to multiply 2 integers and return an integer"""
    return a * b

def add_integers(a: int, b: int) -> int:
    """Function to add 2 integers and return an integer"""
    return a + b

def div_integers(a: int, b: int) -> float:
    """Function to add 2 integers and return a float"""
    return a / b

def execute_code(code: str) -> dict:
    """
    Executes Python code and returns the result or error.
    
    Parameters:
        code (str): The Python code to execute.
    
    Returns:
        dict: A dictionary with 'success', 'output', and 'error' keys.
    """
    # Sandbox for executing the code
    safe_globals = {"__builtins__": {"print": print, "math": math, "random": random, "os": os}}  # Restrict built-ins
    safe_locals = {}

    # Capture the output
    output_buffer = io.StringIO()
    result = {"success": False, "output": None, "error": None}

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals, safe_locals)
        result["success"] = True
        result["output"] = output_buffer.getvalue()
    except Exception as e:
        result["error"] = str(e)
    finally:
        output_buffer.close()

    return result

def code_explainer(code_str: str) -> str:
    """
    Takes in a string of code as input, and explains the logic of the code.
    Accuracy is not the best so verify generated explanation.
    """
    # Tokenize input
    tokenized_inputs = tokenizer(code_str, return_tensors='pt')

    # Generate explanation
    output_ids = model.generate(**tokenized_inputs, max_new_tokens=40)

    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

add_tool = FunctionTool.from_defaults(fn=add_integers)
mul_tool = FunctionTool.from_defaults(fn=mul_integers)
div_tool = FunctionTool.from_defaults(fn=div_integers)
search_tool = FunctionTool.from_defaults(fn=web_search)
code_exec_tool = FunctionTool.from_defaults(fn=execute_code)
code_explainer_tool = FunctionTool.from_defaults(fn=code_explainer)

def return_tool_list():
    return [add_tool, mul_tool, div_tool, search_tool, code_exec_tool, code_explainer_tool]