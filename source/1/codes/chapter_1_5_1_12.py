class Cat(object):
    def __init__(self, age1, color1, nickName1):
        self.age = age1
        self.color = color1
        self.nickName = nickName1

    def showCatInfo(self):
        print("This cat is ", self.nickName)

    def sound(self):
        print("Meow…", "--by cat:", self.nickName)


cat1 = Cat(3, "White", "DuduMiao")
