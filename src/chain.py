from src.chains.claude_chain import ClaudeMessageChain
from src.chains.gemini_chain import GeminiMessageChain
from src.chains.oai_chain import OpenAIMessageChain




class MessageChain:
    @staticmethod
    def get_chain( model: str, **kwargs):
        if "claude" in model:
            return ClaudeMessageChain(**kwargs)
        elif "gemini" in model:
            return GeminiMessageChain(**kwargs)
        elif "gpt" in model:
            return OpenAIMessageChain(**kwargs)
        else:
            raise ValueError(f"Model {model} not supported")
