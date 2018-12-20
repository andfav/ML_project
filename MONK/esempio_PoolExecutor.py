from concurrent.futures import ProcessPoolExecutor as Pool
import time

def wait(s,id):
    time.sleep(s)
    return "Fatto: "+str(id)

def wait2(t):
    (s, id) = t
    time.sleep(s)
    return "Fatto: "+str(id)

def main():
    p = Pool(5)

    #Esempio con submit.
    print('***Esempio con ProcessPoolExecutor.submit.***')
    futureList = []
    for i in range(5):
        futureList.append(p.submit(wait,i+6,i))

    while(sum([futureList[i].done() for i in range(5)]) != 5):
        time.sleep(0.1)
    
    for i in range(5):
        print(str(futureList[i].result()))

    #Esempio con map.
    print('***Esempio con ProcessPoolExecutor.map.***')
    l = list(p.map(wait2,[(i+6,i) for i in range(5)]))

    for i in range(5):
        print(l[i])

if __name__ == '__main__':
    main()
