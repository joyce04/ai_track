#doctest 연습
#각 모듈과 그 컴포넌트의 주된 사용법을 알려주는 문서로써는 유용

def fun(x):
    """Return increment of x
    >>> fun(2)
    3
    >>> fun(3)
    4
    """
    return x +1

#if __name__ == '__main__':
    #import doctest
    #doctest.testmod()