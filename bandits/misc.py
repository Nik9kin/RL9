class Const:
    def __init__(self, x):
        self.x = x

    def __call__(self, *args, **kwargs):
        return self.x
