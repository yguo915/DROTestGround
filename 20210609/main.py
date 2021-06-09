import DRO as dro

def main():
    dir = "/Users/user/PycharmProjects/yijieguo/tissue/data2.txt"
    obj = dro.DRO(dir)
    print(obj.getTissue())



if __name__ == '__main__':
    main()
