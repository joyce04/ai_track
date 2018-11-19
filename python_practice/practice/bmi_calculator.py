import math


def calculate_bmi(weight_pounds, height_inches):
    weight_kg = weight_pounds * 0.45359237
    height_meters = height_inches * 0.0254
    #bmi 는 키(m)의 제곱을 몸무게(kg)로 나눈값
    return str(round(weight_kg /math.pow(height_meters, 2), 2))


def calculate():
    weight = input("enter weight in pounds: ")
    height = input("enter height in inches: ")
    print('for given weight : {} pounds & height : {} inches'.format(weight, height))
    print('calculated bmi is {}'.format(calculate_bmi(float(weight), float(height))))
