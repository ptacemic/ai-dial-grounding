import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# HOBBIES SEARCHING WIZARD
# Implements input/output grounding with adaptive vector store and Named Entity Extraction


NEE_SYSTEM_PROMPT = """You are a Named Entity Extraction system that analyzes user descriptions and extracts hobbies.

## Instructions:
1. Analyze the provided user information (id and about_me sections)
2. Identify hobbies and interests mentioned in the about_me text
3. Group users by their hobbies
4. Return ONLY user IDs grouped by hobby - DO NOT include any other user information
5. Only include hobbies that are explicitly mentioned or clearly implied in the about_me text

## Response Format:
{format_instructions}

## Context:
{context}

## User Query:
{query}
"""


class HobbyUsers(BaseModel):
    """Maps hobbies to lists of user IDs"""
    hobbies: dict[str, list[int]] = Field(
        description="Dictionary mapping hobby names to lists of user IDs who have that hobby"
    )


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


# Initialize clients
embeddings = AzureOpenAIEmbeddings(
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_endpoint=DIAL_URL,
    model="text-embedding-3-small-1",
    dimensions=384
)

llm_client = AzureChatOpenAI(
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_endpoint=DIAL_URL,
    model="gpt-4"
)

user_client = UserClient()
token_tracker = TokenTracker()


class HobbySearchWizard:
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.known_user_ids: set[int] = set()
        
    async def __aenter__(self):
        print("🔎 Initializing vector store with users...")
        # Get all users
        users = user_client.get_all_users()
        
        # Create documents with only id and about_me to reduce context window
        documents = []
        for user in users:
            user_id = user.get('id')
            about_me = user.get('about_me', '')
            self.known_user_ids.add(user_id)
            
            doc = Document(
                page_content=f"User ID: {user_id}\nAbout: {about_me}",
                metadata={"user_id": user_id},
                id=str(user_id)
            )
            documents.append(doc)
        
        # Create vector store
        self.vectorstore = await Chroma.afrom_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="user_hobbies"
        )
        
        print(f"✅ Vector store ready with {len(documents)} users")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.vectorstore:
            # Cleanup - delete collection
            try:
                self.vectorstore.delete_collection()
            except:
                pass
    
    async def update_vectorstore(self):
        """Update vector store with new/deleted users"""
        print("🔄 Updating vector store...")
        
        # Get current users from service
        current_users = user_client.get_all_users()
        current_user_ids = {user['id'] for user in current_users}
        
        # Find deleted users
        deleted_ids = self.known_user_ids - current_user_ids
        if deleted_ids:
            print(f"  Removing {len(deleted_ids)} deleted users")
            await self.vectorstore.adelete(ids=[str(uid) for uid in deleted_ids])
            self.known_user_ids -= deleted_ids
        
        # Find new users
        new_user_ids = current_user_ids - self.known_user_ids
        if new_user_ids:
            print(f"  Adding {len(new_user_ids)} new users")
            new_users = [u for u in current_users if u['id'] in new_user_ids]
            new_docs = []
            for user in new_users:
                user_id = user['id']
                about_me = user.get('about_me', '')
                doc = Document(
                    page_content=f"User ID: {user_id}\nAbout: {about_me}",
                    metadata={"user_id": user_id},
                    id=str(user_id)
                )
                new_docs.append(doc)
            
            await self.vectorstore.aadd_documents(new_docs)
            self.known_user_ids.update(new_user_ids)
        
        if not deleted_ids and not new_user_ids:
            print("  No changes detected")
    
    async def search_by_hobbies(self, query: str, k: int = 20) -> dict[str, list[dict[str, Any]]]:
        """Search users by hobby query and return grouped by hobby with full info"""
        
        # Update vector store with latest users
        await self.update_vectorstore()
        
        # Retrieve relevant user documents
        print(f"\n🔍 Searching for: {query}")
        results = await self.vectorstore.asimilarity_search(query, k=k)
        
        if not results:
            print("No matching users found")
            return {}
        
        print(f"Found {len(results)} potentially matching users")
        
        # Prepare context for NEE
        context_parts = []
        for doc in results:
            context_parts.append(doc.page_content)
        context = "\n\n".join(context_parts)
        
        # Perform Named Entity Extraction
        print("🤖 Extracting hobbies with Named Entity Extraction...")
        parser = PydanticOutputParser(pydantic_object=HobbyUsers)
        
        messages = [
            SystemMessagePromptTemplate.from_template(NEE_SYSTEM_PROMPT)
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
            format_instructions=parser.get_format_instructions(),
            context=context,
            query=query
        )
        
        response = await llm_client.ainvoke((await prompt.aformat_messages())[0].content)
        
        # Track tokens
        usage = response.response_metadata.get('token_usage', {})
        total_tokens = usage.get('total_tokens', 0)
        token_tracker.add_tokens(total_tokens)
        
        # Parse response
        hobby_users: HobbyUsers = parser.parse(response.content)
        
        # Output grounding - verify user IDs and fetch full info
        print("✅ Performing output grounding (verifying user IDs)...")
        grounded_results = {}
        
        for hobby, user_ids in hobby_users.hobbies.items():
            verified_users = []
            for user_id in user_ids:
                try:
                    # Fetch full user info (output grounding)
                    user = await user_client.get_user(user_id)
                    verified_users.append(user)
                    print(f"  ✓ User {user_id} verified for hobby: {hobby}")
                except Exception as e:
                    print(f"  ✗ User {user_id} not found (hallucination detected)")
            
            if verified_users:
                grounded_results[hobby] = verified_users
        
        return grounded_results


async def main():
    print("=" * 60)
    print("HOBBIES SEARCHING WIZARD")
    print("Input/Output Grounding with Named Entity Extraction")
    print("=" * 60)
    print()
    print("Query samples:")
    print(" - I need people who love to go to mountains")
    print(" - Find users interested in technology")
    print(" - Show me people who enjoy outdoor activities")
    print()
    
    async with HobbySearchWizard() as wizard:
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                # Print token usage summary
                summary = token_tracker.get_summary()
                print(f"\n=== Token Usage Summary ===")
                print(f"Total tokens used: {summary['total_tokens']}")
                print(f"Number of queries: {summary['batch_count']}")
                print(f"Tokens per query: {summary['batch_tokens']}")
                break
            
            if not user_question:
                continue
            
            # Search by hobbies
            results = await wizard.search_by_hobbies(user_question)
            
            # Display results in JSON format
            print("\n📊 Results (grouped by hobby):")
            if results:
                import json
                print(json.dumps(results, indent=2, default=str))
            else:
                print("No users found matching the query")
            print()


if __name__ == "__main__":
    asyncio.run(main())



