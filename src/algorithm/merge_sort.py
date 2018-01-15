def merge(arr1, arr2, arr):
    i = j = c = 0

    while i < len(arr1):
        if j < len(arr2):
            if arr1[i] < arr2[j]:
                arr[c] = arr1[i]
                i += 1
            else:
                arr[c] = arr2[j]
                j += 1
            c += 1
        else:
            while i < len(arr1):
                arr[c] = arr1[i]
                i += 1
                c += 1

    while j < len(arr2):
        arr[c] = arr2[j]
        j += 1
        c += 1

    return arr


def merge_practice(arr):
    """
    merge sort method
    :param arr: number of array
    :return: sorted number array
    >>>merge_practice([23, 1, 11, 2, 45, 3, 7, 4, 5, 22, 6, 7])
    [1, 2, 3, 4, 5, 6, 7, 7, 11, 22, 23, 45]
    """
    arr_len = len(arr)
    if arr_len == 1:
        return arr
    else:
        arr1 = merge_practice(arr[0:arr_len / 2])
        arr2 = merge_practice(arr[arr_len / 2:arr_len])
        return merge(arr1, arr2, arr[0:arr_len])


result = merge_practice([23, 1, 11, 2, 45, 3, 7, 4, 5, 22, 6, 7])
print(result)
