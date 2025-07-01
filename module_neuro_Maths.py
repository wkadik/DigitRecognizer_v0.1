import json
import pprint
import random
import numpy


def ele_vec_mul(number: int, vector: list) -> list[int]:
    """
    число перемножается на вектор
    :param number:
    :param vector:
    :return:
    """
    output = [0] * len(vector)
    ind = 0
    while ind < len(output):
        output[ind] = number * vector[ind]
        ind = ind + 1
    return output


def weighted_amount(vec1: list, vec2: list) -> int:
    """
    поэлементно перемножаем элементы и сладываем произвидения
    """
    assert len(vec1) == len(vec2), "длинна векторов не совпfдает"
    ind = 0
    sum_ = 0
    while ind < len(vec1):
        sum_ += vec1[ind] * vec2[ind]
        ind += 1
    return sum_


def vec_mat_mul(vec: list, mat: list[list]) -> list:
    assert len(vec) == len(mat[0])
    result = []
    for v in mat:
        result.append(weighted_amount(vec, v))
    return result


def vec_vec_sub(vec1: list, vec2: list) -> list:
    """
    поэлементное вычитание векторов
    :param vec1:
    :param vec2:
    :return:
    """
    results = []
    for i, j in zip(vec1, vec2):
        results.append(i - j)
    return results


def vec_vec_mul(vec1: list, vec2: list) -> list:
    """
    поэлементное умножение векторов
    :param vec1:
    :param vec2:
    :return:
    """
    results = []
    for i, j in zip(vec1, vec2):
        results.append(i * j)
    return results


# --------------------------------------------------------------------------------
def vec_vec_in_mat(vec1: list, vec2: list) -> list:
    """
    поэлементное умножение векторов
    :param vec1:
    :param vec2:
    :return:
    """
    mat = []
    for i in vec1:
        mat.append(ele_vec_mul(i, vec2))
    return mat


def mat_mat_sub(mat1: list[list], mat2: list[list], alpha: float = 1.0) -> list[list]:
    """
     поэлементное вычитание матриц
    :param mat1:
    :param mat2:
    :param alpha:
    :return:
    """
    assert len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0])
    result_mat = numpy.zeros((len(mat1), len(mat1[0])))
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            result_mat[i][j] = mat1[i][j] - mat2[i][j] * alpha
    return result_mat


def get_random_mat(rows: int, columns: int) -> list[list]:
    """
    создание матриц
    :param rows:
    :param columns:
    :return:
    """
    mat = []
    for i in range(rows):  # отвечает за кол-во строчек
        mat.append([])
        for j in range(columns):  # отвечает за кол-во колонн
            mat[i].append(random.random())
    return mat


def get_zero_mat(rows: int, columns: int) -> list[list]:
    """
    создание матриц
    :param rows:
    :param columns:
    :return:
    """
    mat = []
    for i in range(rows):  # отвечает за кол-во строчек
        mat.append([])
        for j in range(columns):  # отвечает за кол-во колонн
            mat[i].append(0)
    return mat


def transform_mat_in_vec(mat: list[list]) -> list:
    """
    трансформирует матрицу в вектор
    :param mat:
    :return:
    """
    vec = []  # пустой список,в него мы будем закидывать числа из матрицы
    for i in mat:
        vec.extend(i)
    return vec


def transform_vec_in_mat(vec: list, columns: int) -> list[list]:
    """
    трансформирует вектор в матрицу
    :param vec:
    :param columns:
    :return:
    """
    assert len(vec) % columns == 0
    mat = []  # пустая матрица,в неё мы будем закидывать числа из списка
    left_ind = 0
    right_ind = columns
    while right_ind <= len(vec):
        mat.append(vec[left_ind:right_ind])
        left_ind = right_ind
        right_ind += columns
    return mat


def paste_num_in_vec(num: int, max_num: int) -> list:
    """%
    генерирует вектор нулей,по max_num и так скажем и по индексу num прибавляет единичку
    :param num:
    :param max_num:
    :return:
    """
    assert num < max_num
    vec = [0] * max_num
    vec[num] = 1
    return vec


def vec_ele_div(vec: list, num: int) -> list:
    """
    делит все элементы списка на num которе мы указываем
    :param vec:
    :param num:
    :return:
    """
    assert len(vec) > 0  # в списке нет элементов
    result = []
    for i in vec:
        result.append(i / num)
    return result


def vec_ele_mul(vec: list, num: int) -> list:
    """
    умножает все элементы списка на num которое мы указываем
    :param vec:
    :param num:
    :return:
    """
    assert len(vec) > 0  # в списке нет элементов
    result = []
    for i in vec:
        result.append(i * num)
    return result


def top(vec: list, max_top):
    vec = vec.copy()
    sort_vec = vec.copy()
    sort_vec.sort(reverse=True)
    result = []
    for i in sort_vec[:max_top]:
        ind = vec.index(i)
        result.append((ind, i))
    return result


def decor_top(vec, target):
    result = f"target: {target}\n"
    for i in range(len(vec)):
        result += f"{i + 1}) {vec[i][0]} - {round(vec[i][1] , 2)}%\n"
    result += "=" * 30
    return result


def shift_min_vector(vec):
    min_num = min(vec)
    min_num = abs(min_num)
    print(min_num)
    result = []
    for i in vec:
        result.append(i + min_num)
    return result


def sigmoid(x):
    return 1/(1+numpy.e ** -x)


def relu(x):
    return (x>0) * x
def relu2(x):
    return x > 0
