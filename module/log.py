import datetime

def printLog(log):
    print("[", getTimeNow(), "]", end = " ")
    print(log)

def getTimeNow():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")