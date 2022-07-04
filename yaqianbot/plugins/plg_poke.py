from ..backend import poke_receiver
from ..backend.receiver_decos import on_exception_response
from ..utils.candy import simple_send
from .plg_chatbot import response_when
import traceback
@poke_receiver
# @on_exception_response
def on_poke(message):
    try:
        if(message.raw["target_id"] == message.raw["self_id"]):
            response_when(message, "被戳的反应")
    except Exception:
        traceback.print_exc()