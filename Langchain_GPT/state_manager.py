class GlobalState:
    def __init__(self):
        self.text_input = ""
        self.llm_function = None
        self.agent_output_str = ""
        self.log_output_str = ""
        self.bot_message = ""
        self.chat_history = []
        self.finish_answer = ""
        self.log_output_str = ""
        self.streaming_active = False
        self.module_template = ""
global_state = GlobalState()

def get_state():
    return global_state

def update_state(attr, value):
    setattr(global_state, attr, value)
