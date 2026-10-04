"""RAGAS adapter: separate Gemini requests for n samples; no OpenAI n parameter."""
import asyncio
from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM
from config import EVAL_MODEL
from src.llm import completion


class GeminiJudge(BaseRagasLLM):
    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        temperature = self.get_temperature(n) if temperature is None else temperature
        results = [Generation(text=completion([{'role': 'user', 'content': prompt.to_string()}],
                                               model=EVAL_MODEL, json_output=True, temperature=temperature,
                                               cache_tag=f'ragas-sample-{i}')) for i in range(n)]
        return LLMResult(generations=[results])

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)
