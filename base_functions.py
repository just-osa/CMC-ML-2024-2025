from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    res=[]
    for i in range(0, len(X), 4):
        new_row=X[i][120:500:5]
        res.append(new_row)
    return res


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    res=0
    flag=False
    for i in range(min(len(X), len(X[0]))):
        if X[i][i]>=0:
            res+=X[i][i]
            flag=True
    return res if flag else -1

def replace_values(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    newX=deepcopy(X)
    for i in range(len(newX[0])):
        average=0.0
        for j in range(len(newX)):
            average+=newX[j][i]
        average/=len(newX)
        for j in range(len(newX)):
            if newX[j][i]<0.25*average or newX[j][i]>1.5*average:
                newX[j][i]=-1
    return newX
