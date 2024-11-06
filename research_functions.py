from collections import Counter
from typing import List

def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return Counter(x)==Counter(y)

def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    res=-1
    for i in range(1, len(x)):
        if x[i-1]%3==0 or x[i]%3==0:
            res=max(res, x[i-1]*x[i])
    return res

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    # Создаем пустую матрицу для результата
    result = [[0] * len(image[0]) for _ in range(len(image))]

    # Проходим по каждому пикселю
    for i in range(len(image)): #height
        for j in range(len(image[0])):  #width
            weighted_sum = 0
            for k in range(len(image[0][0])):   #amount of channels
                weighted_sum += image[i][j][k] * weights[k]
            result[i][j] = weighted_sum
    return result

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    newx=[]
    newy=[]
    for i in x:
        newx.extend([i[0]]*i[1])
    for i in y:
        newy.extend([i[0]]*i[1])
    if len(newx)!=len(newy): return -1
    res=0
    for i in range(len(newx)): res+=newx[i]*newy[i]
    return res

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    #X - (nxd), Y - (mxd), M - (nxm), Mij=cos(Xi, Yj)
    res=[[0 for j in range(len(Y))] for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            xzero=yzero=True
            sum=xlen=ylen=0
            for k in range(len(X[i])):
                if X[i][k]!=0: xzero=False
                if Y[j][k]!=0: yzero=False
                sum+=X[i][k]*Y[j][k]
                xlen+=X[i][k]**2
                ylen+=Y[j][k]**2
            res[i][j]=1 if (xzero or yzero) else sum/(xlen**0.5)/(ylen**0.5)
    return res
