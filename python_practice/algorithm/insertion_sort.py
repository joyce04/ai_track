def insertion_sort(arr):
    """
    insertion sort function
    :param arr: array of numbers
    :return: ascending order sorted number array

    >>>insertion_sort([1, 16, 3, 32, 5, 2, 6, 10])
    [1, 2, 3, 5, 6, 10, 16, 32]
    """
    for i in range(1, len(arr)):
        key = i
        j = i - 1
        while j >= 0 and arr[j] > arr[key]:
            comparision = arr[j]
            arr[j] = arr[key]
            arr[key] = comparision
            j -= 1
            key -= 1
    return arr


print(insertion_sort([1, 16, 3, 5, 12, 32, 5, 2, 6, 10, 42]))
