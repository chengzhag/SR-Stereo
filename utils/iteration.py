
class NameValues:
    def __init__(self, prefix, suffixes, values):
        self.pairs = []
        for suffix, value in zip(suffixes, values):
            if value is not None:
                self.pairs.append((prefix + suffix, value))

    def str(self, unit=''):
        scale = 1
        if unit == '%':
            scale = 100
        str = ''
        for name, value in self.pairs:
            str += '%s: %.2f%s, ' % (name, value * scale, unit)
        return str

    def dic(self):
        dic={}
        for name, value in self.pairs:
            dic[name] = value
        return dic
