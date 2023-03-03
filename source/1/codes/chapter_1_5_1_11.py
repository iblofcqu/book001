def showText():
    print("***************")
    print("This is first text line")
    print("***************")


showText()


def add(a, b):
    sum = a + b
    print(a, "+", b, "=", sum)
    return sum


rtnValue = add(3, 5)
print("add(3,5) return:", rtnValue)
