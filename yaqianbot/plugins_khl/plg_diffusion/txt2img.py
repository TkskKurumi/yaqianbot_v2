from ...utils.candy import simple_send
from ...backend import receiver
from ...backend.kook import KHLMessage
from ...backend.receiver_decos import command, threading_run, is_su, on_exception_response
from ...backend import log
from .client import Client
from .prompt_processor import PromptProcessor

@receiver
@threading_run
@on_exception_response
@command("/画图", {})
def cmd_txt2img(message: KHLMessage, *args, **kwargs):
    prompt = "/*nsfw, naked, nude, nipples, pussy*/"+" ".join(args)
    prompt = PromptProcessor(message, prompt).result
    t = Client("txt2img")
    t.param(prompt=prompt)
    simple_send(t.get_image())