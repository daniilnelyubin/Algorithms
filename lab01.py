import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import random
import matplotlib.lines as mlines


def const(arr):
    return 1


def sum(arr):
    return np.sum(arr)


def multiplication(arr):
    result = 1
    for value in arr:
        result *= value
    return result


def naive_polinom(arr, x):
    result = 0
    for i in range(len(arr)):
        result += arr[i] * x ** i
    return result


def horner_polinom(arr, x):
    result = 0
    for i in arr[-1:: -1]:
        result = result * x + i
    return result


def bubble(arr):
    if len(arr) <= 1:
        return arr
    N = len(arr)
    for i in range(N - 1):
        for j in range(N - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def quick(arr):
    if len(arr) <= 1:
        return arr

    q = random.choice(arr)
    s_arr = []
    m_arr = []
    e_arr = []
    for n in arr:
        if n < q:
            s_arr.append(n)
        elif n > q:
            m_arr.append(n)
        else:
            e_arr.append(n)
    return quick(s_arr) + e_arr + quick(m_arr)


def insertion(arr):
    for i in range(len(arr)):
        j = i - 1
        key = arr[i]
        while arr[j] > key and j >= 0:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge(A, B):
    i = 0
    j = 0
    k = 0

    C = [0] * (len(A) + len(B))

    while i < len(A) and k < len(B):
        if A[i] <= B[k]:
            C[j] = A[i]
            i += 1
        else:
            C[j] = B[k]
            k += 1
        j += 1

    while i < len(A):
        C[j] = A[i]
        i += 1
        j += 1
    while k < len(B):
        C[j] = B[k]
        k += 1
        j += 1
    return C


def tim(arr, run=32):
    for x in range(0, len(arr), run):
        arr[x: x + run] = insertion(arr[x: x + run])

    while run < len(arr):
        for x in range(0, len(arr), 2 * run):
            arr[x: x + 2 * run] = merge(arr[x: x + run], arr[x + run: x + 2 * run])
        run = run * 2
    return arr


def create_arr(N, max_int):
    temp_matrix = []
    final_matrix = []

    for i in range(0, N):
        for j in range(0, N):
            temp_matrix.append(random.randint(1, max_int))
        final_matrix.append(temp_matrix)
        temp_matrix = []

    return final_matrix


def matrix_multiplication(first_matrix, second_matrix):
    final_matrix = []

    for z in range(0, N):
        temp_matrix = []
        for j in range(0, N):
            sum = 0
            for i in range(0, N):
                sum += (first_matrix[z][i] * second_matrix[i][j])
            temp_matrix.append(sum)
        final_matrix.append(temp_matrix)

    return final_matrix


def plot_data(data, column, pow_of_polinom=1 ,size=-1,title=""):
    if size==-1:
        size = data.shape[0]

    ax1 = sns.lineplot(data=data[:size], x='id', y=column)
    column_line = mlines.Line2D([], [], color='blue', label=title)

    approx = np.poly1d(np.polyfit(range(1, size + 1), data[column][:size], pow_of_polinom))
    values = approx(range(1, size + 1))

    ax2 = sns.lineplot(x=data.id[:size], y=values)
    approximation_line = mlines.Line2D([], [], color='red', label='Approximation')

    plt.legend(handles=[approximation_line,column_line])


    plt.xlabel("Vector size")
    plt.ylabel("Time in seconds")
    plt.title(title)

    plt.show()


def plot_graphics(data):
    for column in data.columns:
        if column== "id":
            continue
        if column == "const":
            plot_data(data, column,title="Constant")
            continue
        if column == "sum":
            plot_data(data, column,title="Sum")
            continue
        if column == "multiplication":
            plot_data(data, column,title="Multiplication")
            continue
        if column == "naive_polinom":
            plot_data(data, column, size=1750,title="Naive Polinom")
            continue
        if column == "horner_polinom":
            plot_data(data, column,pow_of_polinom=1,title="Horner Polinom")
            continue
        if column == "bubble":
            plot_data(data, column, pow_of_polinom=2,title="Bubble sort")
            continue
        if column == "quick":
            plot_data(data, column, pow_of_polinom=2,title="Quick sort")
            continue
        if column == "tim":
            plot_data(data, column, pow_of_polinom=2, title="Tim sort")

            continue




if __name__ == "__main__":
    part = 2
    if part == 1:
        data = pd.DataFrame()

        functions = [const, sum, multiplication, naive_polinom, horner_polinom, bubble, quick, tim]
        count_of_runs = 5

        for function in functions:
            print(function.__name__ + " function execution")
            execution_time_arr = list()
            for n in range(1, 2001):
                vector = list(np.random.randint(1, 100, n))
                execution_time = 0

                for _ in range(count_of_runs):
                    try:
                        start_time = time()
                        if function == naive_polinom or function == horner_polinom:
                            function(vector, 1.5)
                        else:
                            function(vector)
                        finish_time = time()
                        execution_time += (finish_time - start_time)
                    except OverflowError:
                        # print("Stack was overflowed")
                        continue

                execution_time_arr.append(execution_time / count_of_runs)
            data[function.__name__] = execution_time_arr

        data.to_csv('data.csv')

    if part == 2:
        N = 3
        max_int = 5
        first = create_arr(N, max_int)
        second = create_arr(N, max_int)

        print(np.matrix(np.array(first)))
        print(np.matrix(np.array(second)))

        multiplication_matrix = matrix_multiplication(first, second)

        print(np.matrix(np.array(multiplication_matrix)))
    if part == 3:
        data = pd.read_csv("data.csv")
        plot_graphics(data)
