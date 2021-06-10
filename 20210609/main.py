import DRO as dro

def main():
    dir = "/Users/user/PycharmProjects/yijieDRO/tissue/tissueVpVcMets040857_65_65_65.txt"
    tissue = dro.DRO(dir)
    print(tissue.getTissue())
    print(tissue.getTissueSize())



if __name__ == '__main__':
    main()
