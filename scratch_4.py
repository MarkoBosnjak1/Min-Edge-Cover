from timeit import default_timer as timer
import numpy as np
import math
import pandas as pd
from scipy.spatial import KDTree
from collections import defaultdict
from math import sqrt
from operator import itemgetter
from collections import OrderedDict


def lineFromPoints(P, Q, a, b, c):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])
    return a, b, c


# Function which converts the input line to its
# perpendicular bisector. It also inputs the points
# whose mid-point lies on the bisector
def perpendicularBisectorFromLine(P, Q, a, b, c):
    mid_point = [(P[0] + Q[0]) // 2, (P[1] + Q[1]) // 2]

    # c = -bx + ay
    c = -b * (mid_point[0]) + a * (mid_point[1])
    temp = a
    a = -b
    b = temp
    return a, b, c


# Returns the intersection point of two lines
def lineLineIntersection(a1, b1, c1, a2, b2, c2):
    determinant = a1 * b2 - a2 * b1
    if (determinant == 0):

        # The lines are parallel. This is simplified
        # by returning a pair of (10.0)**19
        return [(10.0) ** 19, (10.0) ** 19]
    else:
        x = (b2 * c1 - b1 * c2) // determinant
        y = (a1 * c2 - a2 * c1) // determinant
        return [x, y]


def findCircumCenter(P, Q, R):
    # Line PQ is represented as ax + by = c
    a, b, c = 0.0, 0.0, 0.0
    a, b, c = lineFromPoints(P, Q, a, b, c)

    # Line QR is represented as ex + fy = g
    e, f, g = 0.0, 0.0, 0.0
    e, f, g = lineFromPoints(Q, R, e, f, g)

    # Converting lines PQ and QR to perpendicular
    # vbisectors. After this, L = ax + by = c
    # M = ex + fy = g
    a, b, c = perpendicularBisectorFromLine(P, Q, a, b, c)
    e, f, g = perpendicularBisectorFromLine(Q, R, e, f, g)

    # The point of intersection of L and M gives
    # the circumcenter
    circumcenter = lineLineIntersection(a, b, c, e, f, g)

    if (circumcenter[0] == (10.0) ** 19 and circumcenter[1] == (10.0) ** 19):
        return -1,-1
    else:
        return circumcenter[0],circumcenter[1]


#
#    findCircumCenter(P, Q, R)






    #OUR CODE
arr = np.genfromtxt("izlaz.csv", delimiter=",")
arr_ulaz = np.genfromtxt("ulaz.csv", delimiter=",")
f = open("suncobrani.txt",'w')
#x1,y1,x2,y2,x3,y3 indeks_1,indeks_2 ,indeks_3, distance_1,distance_2, obiljezen
#0, 1, 2 ,3 , 4      ,5       ,6
arr = arr[np.lexsort((arr[:,9] ,arr[:,10]))]
arr = arr[::-1]

rezultat = 0
print(arr[:10])
visited = defaultdict(int)
suncobrani = dict()
tocke_suncobrana = dict()
novi_suncobrani = set()

brojac = 0
def distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

for i in range(1000000):
    # 11x2,21 indeks_3,22_indeks4,23_indeks5,24_indeks6_25_indeks7
    x1, y1 = arr[i][0], arr[i][1]
    x2, y2 = arr[i][2], arr[i][3]
    x3, y3 = arr[i][4], arr[i][5]
    x4, y4 = arr[i][11], arr[i][12]
    r = arr[i][9]

    if (x1,y1) not in visited and (x2,y2) in visited and (x3,y3) not in visited and arr[i][9] == arr[i][10]:
        x2 = x3
        y2 = y3
        visited[(x1,y1)] +=1
        visited[(x2,y2)] +=1

        xs = (x1 + x2) / 2
        ys = (y1 + y2) / 2

        rs = arr[i][9] / 2

        suncobrani[(xs,ys,rs)] = [(x1,y1), (x2,y2)]
        if (x1,y1) in tocke_suncobrana:
            tocke_suncobrana[(x1,y1)].append((x2,y2))
        else:
            tocke_suncobrana[(x1,y1)] = [(x2,y2)]
        if (x2,y2) in tocke_suncobrana:
            tocke_suncobrana[(x3,y3)].append((x1,y1))
        else:
            tocke_suncobrana[(x2,y2)] = [(x1,y1)]
    elif (x1,y1) not in visited and (x2,y2) in visited and (x3,y3)  in visited and (x4,y4) not in visited and arr[i][9] == arr[i][13]:
        x2 = x4
        y2 = y4
        visited[(x1,y1)] +=1
        visited[(x2,y2)] +=1

        xs = (x1 + x2) / 2
        ys = (y1 + y2) / 2

        rs = arr[i][9] / 2

        suncobrani[(xs,ys,rs)] = [(x1,y1), (x2,y2)]
        if (x1,y1) in tocke_suncobrana:
            tocke_suncobrana[(x1,y1)].append((x2,y2))
        else:
            tocke_suncobrana[(x1,y1)] = [(x2,y2)]
        if (x2,y2) in tocke_suncobrana:
            tocke_suncobrana[(x3,y3)].append((x1,y1))
        else:
            tocke_suncobrana[(x2,y2)] = [(x1,y1)]
    elif (x1,y1) not in visited:
        visited[(x1,y1)] +=1
        visited[(x2,y2)] +=1

        xs = (x1 + x2) / 2
        ys = (y1 + y2) / 2

        rs = arr[i][9] / 2

        suncobrani[(xs,ys,rs)] = [(x1,y1), (x2,y2)]
        if (x1,y1) in tocke_suncobrana:
            tocke_suncobrana[(x1,y1)].append((x2,y2))
        else:
            tocke_suncobrana[(x1,y1)] = [(x2,y2)]
        if (x2,y2) in tocke_suncobrana:
            tocke_suncobrana[(x2,y2)].append((x1,y1))
        else:
            tocke_suncobrana[(x2,y2)] = [(x1,y1)]

#suncobrani = OrderedDict(sorted(suncobrani.items(), reverse = True, key=lambda t: t[0][2]))

for key, value in suncobrani.items():
    breaked = True
    for point in value:
        if visited[point] == 1:
            breaked=False

    if breaked:
        for point in value:
            visited[point]-=1
    else:
        novi_suncobrani.add(key)
print(len(novi_suncobrani))

def is_it_worth(a,b,c,d):
    x1,y1 = a[0],a[1]
    x2,y2 = b[0],b[1]
    x3,y3 = c[0],c[1]
    A_B = distance(a,b)
    A_C = distance(a,c)
    B_C = distance(b,c)
    s = A_B + A_C + B_C
    s = s/2
    if s!=A_B and s!=A_C and s!=B_C and s>A_B and s>A_C and s>B_C:
        if a == d:
            radijus_dvi = A_B**2 + A_C**2
        elif b == d:
            radijus_dvi = A_B**2 + B_C**2
        elif c==d:
            radijus_dvi = A_C**2 + B_C**2
        R = (A_B*A_C*B_C*A_B*A_C*B_C)/(s*(s-A_B)*(s-A_C)*(s-B_C))
        if R < radijus_dvi:
            return True
        else:
            return False
    else:
        return False


for key,value in suncobrani.items():
    pas = False
    num_2 = 0
    num_1 = 0
    for point in value:
        if visited[point] == 2:
            tocka_1 = point
            pas = True
            num_2 += 1
        elif visited[point]==1:
            num_1 += 1
        else:
            pas = False
    if pas==True and num_2 == 1:
        points = tocke_suncobrana[tocka_1]
        if len(points)==2:
            tocka_2 = points[0]
            tocka_3 = points[1]
            x1,y1 = tocka_1[0],tocka_1[1]
            x2,y2 = tocka_2[0],tocka_2[1]
            x3,y3 = tocka_3[0],tocka_3[1]
            xs1 = (x1+x2)/2
            ys1 = (y1+y2)/2
            xs2 = (x1+x3)/2
            ys2 = (y1+y3)/2
            a = distance(tocka_1, tocka_2)
            b = distance(tocka_1, tocka_3)
            c = distance(tocka_2, tocka_3)
            r1 = a/2
            r2 = b/2
            povrsina_dvi = r1**2 + r2**2
            s = (a+b+c)/2
            if s!=a and s!=b and s!=c:
                R_na_kvadrat = ((a*b*c)**2)/(16*s*(s-a)*(s-b)*(s-c))
                povrsina_velike = R_na_kvadrat
                if povrsina_velike < povrsina_dvi:
                    xsredista,ysredista = findCircumCenter(tocka_2,tocka_1,tocka_3)
                    if xsredista!=-1:
                        R = R_na_kvadrat**(0.5)
                        novi_suncobrani.remove((xs1,ys1,r1))
                        novi_suncobrani.remove((xs2, ys2, r2))
                        novi_suncobrani.add((xsredista,ysredista,R))
                        visited[tocka_1] = 1

for key, value in suncobrani.items():
    pas = False
    num_1 = 0
    num_2 = 0
    for point in value:
        if visited[point] == 3:
            tocka_1 = point
            pas = True
            num_2 += 1
        elif visited[point] == 1:
            num_1 += 1
        else:
            pas = False
    if pas==True and num_2 == 1:
        points = tocke_suncobrana[tocka_1]
        if len(points)==3:
            tocka_2 = points[0]
            tocka_3 = points[1]
            tocka_4 = points[2]
            x1,y1 = tocka_1[0],tocka_1[1]
            x2,y2 = tocka_2[0],tocka_2[1]
            x3,y3 = tocka_3[0],tocka_3[1]
            x4,y4 = tocka_4[0],tocka_4[1]
            xs1 = (x1+x2)/2
            ys1 = (y1+y2)/2
            xs2 = (x1+x3)/2
            ys2 = (y1+y3)/2
            xs3 = (x1+x4)/2
            ys3 = (y1+y4)/2
            a = distance(tocka_2, tocka_3)
            b = distance(tocka_3,tocka_4)
            c = distance(tocka_2,tocka_4)
            R_1 = distance(tocka_1,tocka_2)/2
            R_2 = distance(tocka_1,tocka_3)/2
            R_3 = distance(tocka_1,tocka_4)/2
            r1 = a/2
            r2 = b/2
            r3 = c/2
            povrsina_tri = R_1**2 +R_2**2 + R_3**2
            s = (a+b+c)/2
            if s!=a and s!=b and s!=c:
                R_na_kvadrat = ((a*b*c)**2)/(16*s*(s-a)*(s-b)*(s-c))
                povrsina_velike = R_na_kvadrat
                if povrsina_velike < povrsina_tri:
                    xsredista,ysredista = findCircumCenter(tocka_2,tocka_3,tocka_4)
                    if xsredista!=-1:
                        R = sqrt(R_na_kvadrat)
                        novi_suncobrani.remove((xs1,ys1,R_1))
                        novi_suncobrani.remove((xs2, ys2, R_2))
                        novi_suncobrani.remove((xs3,ys3,R_3))
                        novi_suncobrani.add((xsredista,ysredista,R))
                        visited[tocka_1] = 1
                elif is_it_worth(tocka_1,tocka_2,tocka_3,tocka_1) == True:
                    a = distance(tocka_1,tocka_2)
                    b = distance(tocka_1,tocka_3)
                    c = distance(tocka_2,tocka_3)
                    xs1 = (tocka_1[0]+tocka_2[0])/2
                    ys1 = (tocka_1[1]+tocka_2[1])/2
                    R_1 = a/2
                    xs2 = (tocka_1[0] + tocka_3[0])/2
                    ys2 = (tocka_1[1] + tocka_3[1])/2
                    R_2 = b/2
                    s = (a+b+c)/2
                    R_na_kvadrat = ((a*b*c)**2)/(16*s*(s-a)*(s-b)*(s-c))
                    R = sqrt(R_na_kvadrat)
                    xsredista, ysredista = findCircumCenter(tocka_1, tocka_2, tocka_3)
                    novi_suncobrani.remove((xs1,ys1,R_1))
                    novi_suncobrani.remove((xs2,ys2,R_2))
                    novi_suncobrani.add((xsredista,ysredista,R))
                elif is_it_worth(tocka_1,tocka_2,tocka_4,tocka_1) == True:
                    a = distance(tocka_1, tocka_2)
                    b = distance(tocka_1, tocka_4)
                    c = distance(tocka_2, tocka_4)
                    xs1 = (tocka_1[0] + tocka_2[0]) / 2
                    ys1 = (tocka_1[1] + tocka_2[1]) / 2
                    R_1 = a / 2
                    xs2 = (tocka_1[0] + tocka_4[0]) / 2
                    ys2 = (tocka_1[1] + tocka_4[1]) / 2
                    R_2 = b / 2
                    s = (a + b + c) / 2
                    R_na_kvadrat = ((a * b * c) ** 2) / (16 * s * (s - a) * (s - b) * (s - c))
                    R = sqrt(R_na_kvadrat)
                    xsredista, ysredista = findCircumCenter(tocka_1, tocka_2, tocka_4)
                    novi_suncobrani.remove((xs1, ys1, R_1))
                    novi_suncobrani.remove((xs2, ys2, R_2))
                    novi_suncobrani.add((xsredista, ysredista, R))
                elif is_it_worth(tocka_1,tocka_3,tocka_4,tocka_1) == True:
                    a = distance(tocka_1, tocka_3)
                    b = distance(tocka_1, tocka_4)
                    c = distance(tocka_3, tocka_4)
                    xs1 = (tocka_1[0] + tocka_3[0]) / 2
                    ys1 = (tocka_1[1] + tocka_3[1]) / 2
                    R_1 = a / 2
                    xs2 = (tocka_1[0] + tocka_4[0]) / 2
                    ys2 = (tocka_1[1] + tocka_4[1]) / 2
                    R_2 = b / 2
                    s = (a + b + c) / 2
                    R_na_kvadrat = ((a * b * c) ** 2) / (16 * s * (s - a) * (s - b) * (s - c))
                    R = sqrt(R_na_kvadrat)
                    xsredista, ysredista = findCircumCenter(tocka_1, tocka_3, tocka_4)
                    novi_suncobrani.remove((xs1, ys1, R_1))
                    novi_suncobrani.remove((xs2, ys2, R_2))
                    novi_suncobrani.add((xsredista, ysredista, R))
for k in novi_suncobrani:
    r = k[2]
    rezultat += r*r*math.pi  # edit
    new = ''
    new += str(k[0]) + ',' + str(k[1]) +',' +str(k[2]) + '\n'
    f.write(new)
print (rezultat)