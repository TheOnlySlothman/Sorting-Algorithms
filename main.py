import random
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import seaborn as sns


def shuffle(array):
    n = len(array)
    temp_array = array.copy()
    while n:
        i = random.randrange(n)
        n -= 1
        temp_array[i], temp_array[n] = temp_array[n], temp_array[i]
    return temp_array


def bad_shuffle(array):
    temp_array = array.copy()
    temp_array.sort(key=lambda a: random.random() - .5)
    return temp_array


def bubble_sort(array):
    for i in range(1, len(array)):
        for j in range(len(array) - i):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array


def merge_sort(array):
    n = len(array)
    temp_array = [0] * n

    def merge(left, right, end):
        middle = right
        for x in range(left, end):
            if right >= end or array[left] <= array[right] and not left >= middle:
                temp_array[x] = array[left]
                left = left + 1
            elif left >= middle or array[left] > array[right]:
                temp_array[x] = array[right]
                right = right + 1
            else:
                print(f'error left: {left}, right: {right}, middle: {middle}'
                      f' array[left]: {array[left]}, array[right]: {array[right]}')

    group_size = 1
    while group_size < n:
        for i in range(0, n, group_size << 1):
            merge(i, min(i + group_size, n), min((i + (group_size << 1)), n))
        array = temp_array
        temp_array = [0] * n
        group_size <<= 1
    return array


def partition(array, left, right, pivot):
    array[pivot], array[right - 1] = array[right - 1], array[pivot]
    pivot = right - 1
    temp = left
    for i in range(left, right):
        if array[i] < array[pivot]:
            array[i], array[temp] = array[temp], array[i]
            temp += 1
    array[temp], array[pivot] = array[pivot], array[temp]
    return temp


def quicksort(array, left, right):
    # n = len(array) / 2
    # median = (array[n] + array[n]) / 2 if len(array) % 2 == 0 else array[n]
    if not (left < right - 1):
        return array
    pivot = (left + right) >> 1
    pivot = partition(array, left, right, pivot)

    quicksort(array, left, pivot)
    quicksort(array, pivot, right)
    return array


def quicksort_start(array):
    quicksort(array, 0, len(array))
    return array


def selection_sort(array):
    length = len(array)
    for i in range(length - 1):
        min_index = i
        for j in range(i + 1, length):
            if array[j] < array[min_index]:
                min_index = j
        array[i], array[min_index] = array[min_index], array[i]
    return array


def insertion_sort(array):
    for i in range(1, len(array)):
        for j in range(i - 1, -1, -1):
            if array[j] > array[i]:
                array[j], array[i] = array[i], array[j]
                i -= 1
            else:
                break
    return array


def heapsort(array):
    if array:
        n = len(array)
        for i in range(n // 2 - 1, -1, -1):
            heapify(array, n, i)
        for i in range(n-1, -1, -1):
            array[i], array[0] = array[0], array[i]
            heapify(array, i, 0)

        return array


def heapify(array, n, i):
    largest = i
    left = i * 2 + 1
    right = i * 2 + 2
    if left < n and array[left] > array[largest]:
        largest = left
    if right < n and array[right] > array[largest]:
        largest = right
    if i != largest:
        array[i], array[largest] = array[largest], array[i]
        heapify(array, n, largest)


def test_all():
    algorithms = [bubble_sort, merge_sort, quicksort_start, selection_sort, insertion_sort, heapsort]
    for a in algorithms:
        print(a.__name__)
        fresh_array = [x for x in range(120)]

        print('10 iterations:')
        all_sorted = True
        for i in range(11):
            array = shuffle(fresh_array)
            array = a(array)
            if fresh_array != array:
                print(f'sorted: {fresh_array == array}')
                print(f'array: {array}, fresh array: {fresh_array}')
                all_sorted = False
                break
        print('1 empty:')
        array = shuffle([])
        array = a(array)
        if array:
            print(f'sorted: {[] == array}')
            print(f'array: {array}, original array: {[]}')
            all_sorted = False
            break
        if all_sorted:
            print('all sorted')
        print()


def main():
    print('1 iteration')
    fresh_array = [x for x in range(10)]
    print(f'fresh array: {fresh_array}')
    array = shuffle(fresh_array)
    print(f'shuffled array: {array}')
    array = heapsort(array)
    print(f'sorted array: {array}')
    print(f'sorted: {fresh_array == array}')

    print('9 iterations:')
    all_sorted = True
    for i in range(10):
        array = shuffle(fresh_array)
        array = heapsort(array)
        if fresh_array != array:
            print(f'sorted: {fresh_array == array}')
            print(f'array: {array}')
            all_sorted = False
            break
    if all_sorted:
        print('all sorted')


def count_values_in_columns(df, order):
    arrays = []
    for i in range(df.shape[1]):
        array = []
        for j in order:
            array.append(df[df[i] == j].shape[0])
        arrays.append(array)
    return pd.DataFrame(arrays)


# The column (horizontal position) of the matrix represents the index of the element prior to shuffling,
# while the row (vertical position) represents the index of the element after shuffling.
def test():
    max_num = 60
    iterations = 1000
    array = [x for x in range(max_num)]
    arrays = []
    for x in range(iterations):
        arrays.append(bad_shuffle(array))

    df = pd.DataFrame(arrays)
    df = count_values_in_columns(df, array)
    # df.count(axis=0, numeric_only=True)
    # df.to_csv(path_or_buf="shuffled.csv", sep=",", index=False)
    # plt.figure(figsize=(df.shape[0], df.shape[1]))
    color_map = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(df, center=iterations / max_num, square=True, vmax=max_num, vmin=0)
    plt.show()


def test2():
    array = [x for x in range(10)]
    arrays = []
    for x in range(20):
        arrays.append(shuffle(array))

    df = pd.DataFrame(arrays)
    temp = df[df[0] == 0].shape[0]
    print(temp)


if __name__ == '__main__':
    test()