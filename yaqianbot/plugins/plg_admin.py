from ..backend import *
from ..backend.receiver_decos import *
from ..utils import after_match
@receiver
@threading_run
@startswith("/exec")
@is_su
def cmd_exec(message: CQMessage):
    text = message.plain_text
    commands = after_match("/exec", text).strip()
    exec(commands)