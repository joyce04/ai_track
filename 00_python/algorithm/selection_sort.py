def selection_sort(arr):
    """
    selection sort function
    :param arr: array of numbers
    :return: ascending order sorted number array

    >>> selection_sort([1, 16, 3, 32, 5, 2, 6, 10])
    [1, 2, 3, 5, 6, 10, 16, 32]
    """

    for i in range(0, len(arr) - 1):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[min_index] > arr[j]:
                min_index = j
        if min_index != i:
            origin_value = arr[i]
            arr[i] = arr[min_index]
            arr[min_index] = origin_value
    return arr


print(selection_sort([1, 16, 3, 32, 5, 2, 6, 10]))
