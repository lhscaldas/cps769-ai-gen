
import getpass
import os
from env import OPENAI_API_KEY # key armazenada em um arquivo que está no .gitignore para manter o sigilo

# os.environ['OPENAI_API_KEY'] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    max_tokens=200,
    temperature=1,
)

from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


structured_llm = llm.with_structured_output(Joke)

print(structured_llm.invoke("Tell me a joke about cats"))