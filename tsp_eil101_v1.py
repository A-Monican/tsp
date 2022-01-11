import random
import sys
import pandas as pd
import numpy as np
from pandas.core.accessor import DirNamesMixin

np.set_printoptions(threshold=sys.maxsize)

def ct_distance(l1):
        #计算2城市之间的欧氏距离，向下取整
        d = np.sqrt((l1[0][0]-l1[1][0]) ** 2 + (l1[0][1] - l1[1][1]) ** 2)
        dis = int(0.5 + d)
        return dis

def get_dab(a, b, city_location):#获得ab之间距离
        a = city_location[a]
        b = city_location[b]
        dab = [a,b]
        return ct_distance(dab)

def get_tour_cost(x, cost_tsp):#计算x的回路长度
        dab = np.zeros(101)
        tour_total_distance = 0
        for i in range(0,101):
                if i == 100:
                        a = x[i]
                        b = x[0]
                else:
                        a = x[i]
                        b = x[i+1]
                dab[i] = cost_tsp[a-1][b-1]#while cost_tsp[a-1][b-1]
                tour_total_distance = tour_total_distance + dab[i]
        return tour_total_distance

def rev(dx):       # i j from cost_tsp 交换x中的ij得到新的x2 并计算x2的回路长度 返回新dx
        i=random.randint(0,100)
        j=random.randint(0,100)
        if(i>j):
                c=i
                i=j
                j=c
        if((i==j)or(i==0 and j==100)or(i==1 and j==100)or(i==0 and j==99)):
                dx['tour_total_distance'] = dx['tour_total_distance']
        else:        
                dx['tour_total_distance'] = dx['tour_total_distance'] - cost_tsp[x[((i-1)+101)%101]-1][x[i]-1] - cost_tsp[x[j]-1][x[(j+1)%101]-1] + cost_tsp[x[((i-1)+101)%101]-1][x[j]-1] + cost_tsp[x[i]-1][x[(j+1)%101]-1]           
        xa =x[:i]
        xb =x[j+1:]
        xij = x[i:j+1]
        xji=xij[::-1]
        x2 = xa + xji +xb
        dx['x'] = x2
        return dx

df = pd.read_csv('eil101.tsp', sep=" ", skiprows=6, header=None)   
city = np.array(df[0][0:len(df)-1])
city_name = city.tolist()
city_x = np.array(df[1][0:len(df)-1])
city_y = np.array(df[2][0:len(df)-1])
city_location = list(zip(city_x,city_y))
#print(city_location)

cost_tsp = np.zeros((101,101))
for i in range(0,101):
        for j in range(0,101):
               cost_tsp[i][j]=get_dab(i , j, city_location)

x = [ k for k in range(1,102)]
random.shuffle(x)
tour_total_distance = get_tour_cost(x,cost_tsp)
dx = {'x':x,'tour_total_distance':tour_total_distance}

#bestx = [1,69,27,101,53,28,26,12,80,68,29,24,54,55,25,4,39,67,23,56,75,41,22,74,72,73,21,40,58,13,94,95,97,87,2,57,15,43,42,14,44,38,86,16,61,85,91,100,98,37,92,59,93,99,96,6,89,52,18,83,60,5,84,17,45,8,46,47,36,49,64,63,90,32,10,62,11,19,48,82,7,88,31,70,30,20,66,71,65,35,34,78,81,9,51,33,79,3,77,76,50]
#bestdistance = get_tour_cost(bestx,cost_tsp)
#print(bestdistance)

rev(dx)
print(dx['tour_total_distance'])

xc = dx['x']
yc = dx['tour_total_distance']
dxc={'xc':xc,'yc':yc}

FEs = 1
while (FEs < 10000):
        xn = dx['x']
        yn = dx['tour_total_distance']   
        dxn = {'xn':xn,'tour_total_distance':yn}
        rev(dxn)
        if(dxn['tour_total_distance'] <= dxc['yc']):    
                dxc['xc']=dxn['xn']
                dxc['yc']=dxn['tour_total_distance']
                print(dxc['yc'])   
        FEs = FEs + 1
   