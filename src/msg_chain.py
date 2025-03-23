



class MessageChain:
    @staticmethod
    def get_chain( model: str, **kwargs):
        if "claude" in model:
            from src.msg_chains.claude_msg_chain import ClaudeMessageChain
            return ClaudeMessageChain(**kwargs)
        elif "gemini" in model:
            from src.msg_chains.gemini_msg_chain import GeminiMessageChain
            return GeminiMessageChain(**kwargs)
        elif "gpt" in model:
            from src.msg_chains.oai_msg_chain import OpenAIMessageChain
            return OpenAIMessageChain(**kwargs)
        else:
            raise ValueError(f"Model {model} not supported")
