class Conversation:
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id
        self.turns = []
        self.session_values = {}

    def add_turn(self, question, answer_dict):
        self.turns.append({"question": question, "answer": answer_dict})

    def history_text(self):
        lines = []
        for turn in self.turns:
            lines.append("User: {}".format(turn["question"]))
            lines.append("Assistant: {}".format(turn["answer"]["answer"]))
        return "\n".join(lines)

    def set_session_value(self, key, value):
        self.session_values[key] = value

    def has_active_override(self):
        return bool(self.session_values)

    def get_session_values_text(self):
        if not self.session_values:
            return ""
        parts = list(self.session_values.values())
        return ("The user has made this assumption for THIS conversation only (never update the "
                "documents themselves): " + " | ".join(parts))


class ConversationStore:
    def __init__(self):
        self.conversations = {}

    def new_conversation(self, conversation_id):
        # Callers invoke this once per golden-set/submission item, passing the item's
        # conversation_id. Several items can share one conversation_id (a multi-turn
        # exchange), so this must return the existing Conversation for that id rather
        # than overwrite it, or every multi-turn follow-up loses all prior context.
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(conversation_id)
        return self.conversations[conversation_id]

    def get(self, conversation_id):
        return self.conversations.get(conversation_id)
