# Classify the user query into one of these categories: code_execution, summary, search, analysis, comparison
query_classification_prompt = """You are an intelligent AI assistant. You will be provided with a user query. You will have to classify the query into one of the following categories:
- **code_execution**: The query specifically requires a code to be **executed**, such as plotting, performing numerical calculations, or verifying the output of a code. **Note:** If the user only asks to write the code and not execute it, do not classify it as code_execution.
- **summary**: The query is either asking for a summary or it requires a large amount of information to be retrieved and summarized.
- **search**: The query is asking for a specific information which can be answered with a single piece of information.
- **analysis**: The query is asking for a thorough analysis of every part of some document or text, which may require reasoning and understanding of the text.
- **comparison**: The query is asking for a comparison between two or more entities, which may require multiple sources of information.

**Note:**
- If you are not confident about the classification, respond with 'other'.
- Only provide your answer in lowercase. Do not provide any other explanation or response.

User Query: {user_query}
Answer:"""

# Don't classify into analysis if document is not given
query_classification_prompt_no_doc = """
You are an intelligent AI assistant. Classify the user query into one of the following categories:
- code_execution
- summary
- search
- comparison

Your response must be one of these categories and nothing else. Do not include any other text or explanation.

User Query: {user_query}
Answer:"""


summarize = """
You are an intelligent AI assistant. You will be provided with a user query and the subqueries generated from the user query. You will have to **summarize** the answer with the context provided with the subqueries.
Remeber, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence the persons or the queries are related to IITH only.

"Guidelines:\n"
"1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
"2. Ensure that the answer is coherent and relevant to the context provided.\n"
"3. The answer should be a summary of the context and the user query.\n"
"4. The answer should be in complete sentences and should be grammatically correct.\n"
"5. The answer should be concise and to the point.\n"
"6. Do not include any irrelevant information in the answer.\n"
"7. Do not include any personal opinions in the answer.\n"
"8. Do not include any information that is not present in the context.\n"
"9. Do not include any information that is not relevant to the user query.\n"


User Query: {user_query}
Context: {context}
Subqueries: {subqueries}
Answer:"""



question_answering = """
You are an intelligent AI assistant. You will be provided with a user query and the subqueries generated from the user query. You will have to **answer shortly** the subqueries with the context provided with the subqueries.
Remeber, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence the persons or the queries are related to IITH only.

"Guidelines:\n"
"1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
"2. Ensure that the answer is coherent and relevant to the context provided.\n"
"3. The answer should be a short answer of the context and the user query.\n"
"4. The answer should be in complete sentences and should be grammatically correct.\n"
"5. The answer should be concise and to the point.\n"
"6. Do not include any irrelevant information in the answer.\n"
"7. Do not include any personal opinions in the answer.\n"
"8. Do not include any information that is not present in the context.\n"
"9. Do not include any information that is not relevant to the user query.\n"


User Query: {user_query}
Context: {context}
Subqueries: {subqueries}
Answer:"""

search = """
You are an intelligent AI assistant. You will be provided with a user query and the subqueries generated from the user query. You will have to **detail the answer** for the subqueries with the context provided with the subqueries.
Remeber, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence the persons or the queries are related to IITH only.

"Guidelines:\n"
"1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
"2. Ensure that the answer is coherent and relevant to the context provided.\n"
"3. The answer should be a *detailed* answer of the context and the user query.\n"
"4. The answer should be in complete sentences and should be grammatically correct.\n"
"5. The answer should be concise and to the point.\n"
"6. Do not include any irrelevant information in the answer.\n"
"7. Do not include any personal opinions in the answer.\n"
"8. Do not include any information that is not present in the context.\n"
"9. Do not include any information that is not relevant to the user query.\n"


User Query: {user_query}
Context: {context}
Subqueries: {subqueries}
Answer:"""


exploration = """
You are an intelligent AI assistant. You will be provided with a user query and the subqueries generated from the user query. You will have to **detail the answer** for the subqueries with the context provided with the subqueries.
Remeber, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence the persons or the queries are related to IITH only.

"Guidelines:\n"
"1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
"2. Ensure that the answer is coherent and relevant to the context provided.\n"
"3. The answer should be a *detailed* answer of the context and the user query.\n"
"4. The answer should be in complete sentences and should be grammatically correct.\n"
"5. The answer should be concise and to the point.\n"
"6. Do not include any irrelevant information in the answer.\n"
"7. Do not include any personal opinions in the answer.\n"
"8. Do not include any information that is not present in the context.\n"
"9. Do not include any information that is not relevant to the user query.\n"


User Query: {user_query}
Context: {context}
Subqueries: {subqueries}
Answer:"""


fact_verification = """
You are an intelligent AI assistant. You will be provided with a user query and the subqueries generated from the user query. You will have to **support the fact** for the subqueries with the context provided with the subqueries.
Remeber, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence the persons or the queries are related to IITH only.

"Guidelines:\n"
"1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
"2. Ensure that the answer is coherent and relevant to the context provided.\n"
"3. The answer should be a *supporting the fact* in the context and the user query.\n"
"4. The answer should be in complete sentences and should be grammatically correct.\n"
"5. The answer should be concise and to the point.\n"
"6. Do not include any irrelevant information in the answer.\n"
"7. Do not include any personal opinions in the answer.\n"
"8. Do not include any information that is not present in the context.\n"
"9. Do not include any information that is not relevant to the user query.\n"


User Query: {user_query}
Context: {context}
Subqueries: {subqueries}
Answer:"""