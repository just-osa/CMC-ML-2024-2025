import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    xunique, xcount = np.unique(x, return_counts=True)
    yunique, ycount = np.unique(y, return_counts=True)
    return np.array_equal(xunique, yunique) and np.array_equal(xcount, ycount)

def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    prod=x[:-1]*x[1:]
    condition=(x[:-1]%3==0)|(x[1:]%3==0)
    res=prod[condition]
    if res.size==0:
        return -1
    return np.max(res)



def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    newimage=image*weights[np.newaxis, np.newaxis, :]
    res=np.sum(newimage, axis=-1)
    return res


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    xnormal=np.repeat(x[:, 0], x[:, 1])
    ynormal=np.repeat(y[:, 0], y[:, 1])
    return -1 if len(xnormal)!=len(ynormal) else np.dot(xnormal, ynormal)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    xnorm=np.linalg.norm(X, axis=1, keepdims=True)
    ynorm=np.linalg.norm(Y, axis=1, keepdims=True)
    prod=np.dot(X, Y.T)
    cos=prod/(xnorm * ynorm.T)
    cos=np.where((xnorm==0)|(ynorm.T==0), 1, cos)
    return cos