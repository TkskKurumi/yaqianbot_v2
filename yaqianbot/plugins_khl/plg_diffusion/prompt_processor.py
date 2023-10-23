import re
from ...backend.kook import KHLMessage
from .tag_storage import get_user_tag
from typing import Dict
class PromptProcessor:
    def _process_replace(entries: Dict[str, str], prompt: str):
        replaced = prompt
        for k, v in entries.items():
            if(k in replaced):
                replaced = replaced.replace(k, v+", ")
        return replaced
    def _process_comma(prompt: str):
        # full_comma = "\uff0c"
        pattern = r" *[,\uff0c] *"
        spl = re.split(pattern, prompt)
        spl = [i.strip() for i in spl if i.strip()]
        return ", ".join(spl)
    def __init__(self, message, prompt: str):
        tags = get_user_tag(message)
        replaced = PromptProcessor._process_replace(tags, prompt)
        comma_d = PromptProcessor._process_comma(replaced)
        self.result = comma_d




