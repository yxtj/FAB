import random
import timeit
import sys

def one_dp(dim):
    return [ random.random()*2 - 1 for _ in range(dim)]

def one_dpc(center, dist):
    return [ v + random.random()*2*dist-dist for v in center]

def genCentroids(k, dim):
    centroids = []
    for i in range(k):
        centroids.append(one_dp(dim))
    return centroids

def genNum4Each(k, num):
    portion = [ random.random() for _ in range(k)]
    sp = sum(portion)
    cluster_num = [ portion[i]*num/sp for i in range(k)]
    sn = sum(cluster_num)
    d = num - sn
    if d != 0:
        cluster_num[-1] += d;

def dumpParam(fout, k, dim, centroids):
    fout.write('%d,%d\n' % (k,dim))
    for c in centroids:
        fout.write(','.join('%.8s' % v for v in c))
        fout.write('\n')


def KMgen(fnd, fnp, k, dim, num, seed):

    fout_d = open(fnd, 'w')
    fout_p = open(fnp, 'w')

    random.seed(seed)

    # generate centroids
    centroids = genCentroids(k, dim)
    dumpParam(fout_p, k, dim, centroids);
    fout_p.close()

    cluster_num = genNum4Each(k, num)
    clist = [ i for i in range(k) for _ in range(cluster_num[i])]
    random.shuffle(clist)
    # generate cluster data
    cnt = 0
    for ci in clist:
        c = centroids[ci]
        dist=random.random()*0.5+0.01
        point = one_dpc(c, dist)
        fout_d.write(','.join('%.8s' % v for v in point))
        fout_d.write('\n')
        cnt+=1
        if cnt % 10000 == 0:
            print("generate %s data points" % (cnt))
    fout_d.close()

def stoiKMG(s):
    s=s.lower()
    f = 1
    if s[-1] == 'k':
        f=1000
    elif s[-1] == 'm':
        f=1000*1000
    elif s[-1] == 'g':
        f=1000*1000*1000
    if s[-1] in ['k','m','g']:
        s=s[:-1]
    return int(s) * f


if __name__ == '__main__':
    argc = len(sys.argv)
    print (sys.argv)
    if argc <= 5:
        print("Usage: <k> <dim> <n> <data-file> <param-file> [seed=123]")
    else:
        k = int(sys.argv[1])
        dim = int(sys.argv[2])
        num = stoiKMG(sys.argv[3])
        fnd = sys.argv[4]
        fnp = sys.argv[5]
        seed = int(sys.argv[6]) if argc > 6 else 123

        print("KMgen write to file " + fnd)

        start = timeit.default_timer()
        KMgen(fnd, fnp, k, dim, num, seed)
        end = timeit.default_timer()

        print("KMgen finish in " + str((end - start)) + " s")
