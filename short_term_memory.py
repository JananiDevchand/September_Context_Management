from langchain.memory import ConversationBufferWindowMemory

class ShortTermMemory:
    def __init__(self, k=2):
        self.memory = ConversationBufferWindowMemory(
            k=k,
            return_messages=True
        )

    def load(self):
        return self.memory.load_memory_variables({}).get("history", [])

    def add(self, user_input, assistant_output):
        self.memory.save_context(
            {"input": user_input},
            {"output": assistant_output}
        )
