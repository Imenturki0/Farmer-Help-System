from collections import defaultdict

MAX_HISTORY = 10

class ConversationMemory:

    def __init__(self):
        self.sessions =defaultdict(list)

    def add(self,session_id,role,content):

        self.sessions[session_id].append({
            "role": role,
            "content": content
        })

        # Keep only recent messages
        if len(self.sessions[session_id]) > MAX_HISTORY:
            self.sessions[session_id] = self.sessions[session_id][-MAX_HISTORY:]

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def clear(self, session_id):
        self.sessions.pop(session_id, None)
    
    def get(self,session_id):
        if session_id not in self.sessions:
            self.sessions[session_id]
        return self.sessions[session_id]
    
memory = ConversationMemory()