class Test():
    def __init__(self):
        self.__dict__['test_param'] = 1

    def display_param(self):
        print(self.test_param)

    def set_param(self):
        test = self.__dict__['test_param']
        test = 5

t = Test()
t.display_param()
t.set_param()
t.display_param()