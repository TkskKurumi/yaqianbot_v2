from dataclasses import dataclass
from typing import Any


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
        rpics[self.sender] = rpics.get(self.sender, self.pics)
        self.recent_pics = rpics[self.sender]
        return self.recent_pics

    async def response_async(self, *args, **kwargs):
        raise NotImplementedError()

    def response_sync(self, *args, **kwargs):
        raise NotImplementedError()
