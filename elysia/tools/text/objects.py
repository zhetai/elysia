from elysia.objects import Text


class Summary(Text):
    def __init__(self, text: str, title: str):
        Text.__init__(self, "summary", [{"text": text, "title": title}])
