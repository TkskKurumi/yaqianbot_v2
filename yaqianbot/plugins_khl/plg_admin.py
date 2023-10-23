from ..backend import receiver
from ..backend.kook import KHLMessage
from ..backend.receiver_decos import command, threading_run, is_su, on_exception_response
from ..backend import log
import html

@receiver
@threading_run
@on_exception_response
@is_su
@command("/exec", {})
def cmd_exec(message: KHLMessage, *args):
    content = message.raw.extra.get("kmarkdown").get("raw_content")
    content = content[5:].strip(" ")
    exec(content)