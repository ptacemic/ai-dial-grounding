from enum import StrEnum
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from openai import BaseModel
from pydantic import SecretStr, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `api_based_grounding.png` to see the flow of app

QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts search parameters from user questions about users.

## Available Search Fields:
- **name**: User's first name (e.g., "John", "Mary")
- **surname**: User's last name (e.g., "Smith", "Johnson") 
- **email**: User's email address (e.g., "john@example.com")

## Instructions:
1. Analyze the user's question and identify what they're looking for
2. Extract specific search values mentioned in the query
3. Map them to the appropriate search fields
4. If multiple search criteria are mentioned, include all of them
5. Only extract explicit values - don't infer or assume values not mentioned

## Examples:
- "Who is John?" → name: "John"
- "Find users with surname Smith" → surname: "Smith" 
- "Look for john@example.com" → email: "john@example.com"
- "Find John Smith" → name: "John", surname: "Smith"
- "I need user emails that filled with hiking" → No clear search parameters (return empty list)

## Response Format:
{format_instructions}
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


#TODO:
# 1. Create AzureChatOpenAI client
# 2. Create UserClient

llm_client = AzureChatOpenAI(
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_endpoint=DIAL_URL,
    model="gpt-4"
)
user_client = UserClient()
token_tracker = TokenTracker()


#TODO:
# Now we need to create pydentic models that will be user for search and their JSON schema will be passed to LLM by
# langchain. In response from LLM we expect to get response in such format (JSON by JSON Schema)
# 1. SearchField class, extend StrEnum and has constants: name, surname, email
# 2. Create SearchRequest, extends pydentic BaseModel and has such fields:
#       - search_field (enum from above), also you can provide its `description` that will be provided with JSON Schema
#         to LLM that model will be better understand what you expect there
#       - search_value, its string, sample what we expect here is some name, surname or email to make search
# 3. Create SearchRequests, extends pydentic BaseModel and has such fields:
#       - search_request_parameters, list of SearchRequest, by default empty list

class SearchField(StrEnum):
    name = "name"
    surname = "surname"
    email = "email"


class SearchRequest(BaseModel):
    search_field: SearchField = Field(description="The field to search by (name, surname, or email)")
    search_value: str = Field(description="The value to search for in the specified field")


class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(default_factory=list, description="List of search parameters extracted from the user query")


def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    """Extract search parameters from user query and retrieve matching users."""
    #TODO:
    # 1. Create PydanticOutputParser with `pydantic_object=SearchRequests` as `parser`
    # 2. Create messages array with:
    #       - use SystemMessagePromptTemplate and from template generate system message from QUERY_ANALYSIS_PROMPT
    #       - user message
    # 3. Generate `prompt`: `ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())`
    # 4. Invoke it: `(prompt | llm_client | parser).invoke({})` as `search_requests: SearchRequests` (you are using LCEL)
    # 5. If `search_requests` has `search_request_parameters`:
    #       - create `requests_dict`
    #       - iterate through searched parameters and:
    #           - add to `requests_dict` the `search_request.search_field.value` as key and `search_request.search_value` as value
    #       - print `requests_dict`
    #       - search users (**requests_dict) with `user_client`
    #       - return users that you found
    # 6. Otherwise print 'No specific search parameters found!' and return empty array
    
    # 1. Create parser
    parser = PydanticOutputParser(pydantic_object=SearchRequests)
    
    # 2. Create messages
    messages = [
        SystemMessagePromptTemplate.from_template(QUERY_ANALYSIS_PROMPT),
        HumanMessage(content=user_question)
    ]
    
    # 3. Generate prompt
    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())
    
    # 4. Invoke using LCEL
    search_requests: SearchRequests = (prompt | llm_client | parser).invoke({})
    
    # 5. If search parameters found
    if search_requests.search_request_parameters:
        requests_dict = {}
        for search_request in search_requests.search_request_parameters:
            requests_dict[search_request.search_field.value] = search_request.search_value
        
        print(f"Search parameters: {requests_dict}")
        users = user_client.search_users(**requests_dict)
        return users
    
    # 6. No parameters found
    print('No specific search parameters found!')
    return []


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    """Combine user query with retrieved context into a formatted prompt."""
    #TODO:
    # 1. Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    # 2. Make augmentation for USER_PROMPT
    # 3. print augmented prompt
    # 3. return augmented prompt
    
    # 1. Prepare context
    result = []
    for user in context:
        user_str = "User:\n"
        for key, value in user.items():
            user_str += f"  {key}: {value}\n"
        result.append(user_str)
    context_str = "\n".join(result)
    
    # 2. Make augmentation
    augmented_prompt = USER_PROMPT.format(context=context_str, query=user_question)
    
    # 3. Print augmented prompt
    print(f"Augmented prompt:\n{augmented_prompt}")
    
    # 3. Return augmented prompt
    return augmented_prompt


def generate_answer(augmented_prompt: str) -> str:
    """Generate final answer using the augmented prompt."""
    #TODO:
    # 1. Create messages array with:
    #       - SYSTEM_PROMPT
    #       - augmented_prompt
    # 2. Generate response, use invoke method with llm_client
    # 3. Return response content
    
    # 1. Create messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=augmented_prompt)
    ]
    
    # 2. Generate response
    response = llm_client.invoke(messages)
    
    # Track token usage
    usage = response.response_metadata.get('token_usage', {})
    total_tokens = usage.get('total_tokens', 0)
    token_tracker.add_tokens(total_tokens)
    
    # 3. Return response content
    return response.content


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    #TODO:
    # 1. Create infinite loop
    # 2. Get input from console as `user_question`
    # 3. retrieve context
    # 4. if context is present:
    #       - make augmentation
    #       - generate answer with augmented prompt
    # 5. Otherwise print `No relevant information found`
    
    # 1. Create infinite loop
    while True:
        # 2. Get input from console
        user_question = input("> ").strip()
        if user_question.lower() in ['quit', 'exit']:
            # Print token usage summary before exiting
            summary = token_tracker.get_summary()
            print(f"\n=== Token Usage Summary ===")
            print(f"Total tokens used: {summary['total_tokens']}")
            print(f"Number of queries: {summary['batch_count']}")
            print(f"Tokens per query: {summary['batch_tokens']}")
            break
        
        # 3. Retrieve context
        context = retrieve_context(user_question)
        
        # 4. If context is present
        if context:
            augmented_prompt = augment_prompt(user_question, context)
            answer = generate_answer(augmented_prompt)
            print(f"\nAnswer: {answer}\n")
        # 5. Otherwise print no info found
        else:
            print("No relevant information found\n")


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce