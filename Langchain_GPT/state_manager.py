import threading
import queue


class GlobalState:
    def __init__(self):
        self.text_input = ""
        self.llm_function = None
        self.agent_output_str = ""
        self.log_output_str = ""
        self.bot_message = ""
        self.chat_history = []
        self.finish_answer = ""
        self.streaming_active = True
        self.module_template = ""
        self.streaming_buffer = queue.Queue()
        self.buffer_lock = threading.Lock()
        self.reset_flag = False
        self.glm_model_info = ""
        self.openai_model_info = ""

global_state = GlobalState()
shared_output = queue.Queue()
Images_path = []
glm_chat_history = []

def get_state():
    global_state.thread_stop_event = threading.Event()
    return global_state


def update_state(attr, value):
    setattr(global_state, attr, value)
