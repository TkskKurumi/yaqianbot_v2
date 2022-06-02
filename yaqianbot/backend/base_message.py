from dataclasses import dataclass
from typing import Any
from .requests import get_image



@dataclass
class User:
    id: str
    # id
    name: str
    # username
    from_group: str
    # group id/private message

    def __hash__(self):
        return hash((self.from_group, self.id))

    def __eq__(self, other):
        return (self.from_group == other.from_group) and (self.id == other.id)
    def get_avatar(self):
        url = "https://q.qlogo.cn/headimg_dl?dst_uin=%s&img_type=jpg&spec=640"%self.id
        return get_image(url)


rpics = dict()


@dataclass
class Message:
    sender: User
    pics: Any = None
    recent_pics: Any = None
    ated: Any = None
    plain_text: str = ""
    group: str = ""
    raw: Any = None

    def update_rpics(self):
        
        rpics[self.sender] = self.pics or rpics.get(self.sender)
        self.recent_pics = rpics[self.sender]
        return self.recent_pics

    async def response_async(self, *args, **kwargs):
        raise NotImplementedError()

    def response_sync(self, *args, **kwargs):
        raise NotImplementedError()
