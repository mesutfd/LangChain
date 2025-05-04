from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from dotenv import load_dotenv

import requests
import os

load_dotenv()

class DeepSeekChat(BaseChatModel):
    model: str = "deepseek-chat"
    temperature: float = 0.7
    api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    system_prompt: str = "your response is being used directly in an automated system, DO NOT return any extra description, SQL-QUERY-ONLY"

    def _call(
            self,
            messages: List[HumanMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        def map_message(m):
            if isinstance(m, HumanMessage):
                return {"role": "user", "content": m.content}
            elif isinstance(m, AIMessage):
                return {"role": "assistant", "content": m.content}
            else:
                return {"role": "user", "content": m.content}  # fallback

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}] + [map_message(m) for m in messages],
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = "https://api.deepseek.com/v1/chat/completions"
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _generate(
            self,
            messages: List[HumanMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        response_text = self._call(messages, stop=stop, run_manager=run_manager)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response_text))]
        )

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"


# Instantiate the custom DeepSeek LLM
llm = DeepSeekChat(model="deepseek-chat", temperature=0)

# from langchain.chains import SQLDatabaseChain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI  # or any chat model you use

# Create a SQLAlchemy-compatible connection string
db = SQLDatabase.from_uri("postgresql+psycopg2://admin:P%40ssw0rd@localhost:5432/baroline")

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

from main import enrich_prompt

en_pr = enrich_prompt(query="Give me report of my cargoes per month.")

response = db_chain.run(en_pr)

print(response)
