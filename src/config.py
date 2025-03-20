class DotDict(dict):

    def __getattr__(self, name):

        try:
            value = self[name]

            if isinstance(value, dict):
                value = DotDict(value)
                self[name] = value
            return value

        except KeyError:
            raise AttributeError(f"'dotdict' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]