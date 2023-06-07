import pickle


def saveclass(name, obj):
    obj=pickle.dumps(obj)
    with open(name,"ab")as f:
        f.write(obj)


def loadclass(name):
    f=open(name,"rb")
    while True:
        try:
            obj = pickle.load(f)
        except:
            break
    f.close()
    return obj

