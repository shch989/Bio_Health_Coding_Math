import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import math
import sympy as sym
from itertools import permutations
from itertools import combinations
from itertools import product
from itertools import combinations_with_replacement
import datetime
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def time_check(): #수행시간 확인 함수입니다. 내용 수정하지 말아주세요
    print("중간고사 시험 수행 인증시간:", "%s년 %s월 %s일 %s시 %s분" %(datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day,datetime.datetime.now().hour,datetime.datetime.now().minute))

maze = {
    'a': ['b'],
    'b': ['a', 'c'],
    'c': ['b', 'd', 'g'],
    'd': ['d'],
    'e': ['f', 'i'],
    'f': ['e', 'g'],
    'g': ['f', 'h', 'k'],
    'h': ['g', 'l'],
    'i': ['e', 'j'],
    'j': ['i', 'n'],
    'k': ['g'],
    'l': ['h'],
    'm': ['n'],
    'n': ['j', 'm', 'o'],
    'o': ['n', 'p'],
    'p': ['o']
}

#미로찾기 함수입니다. 내용 수정하지 말아주세요
def my_maze(maze, start, end):
    qu = []
    done = set()
    qu.append(start)
    done.add(start)

    print("시작")
    while qu:
        p = qu.pop(0)
        v = p[-1]
        
        if v == end:
            print("종료")
            return p

        for x in maze[v]:
            if x not in done:
                qu.append(p+x)
                done.add(x)

print("답: ", my_maze(maze, "a", "p"))



def exam_1():
    print("문제1번")
    #코딩은 여기부터 시작해주세요
    x = np.arange(-10, 11)

    y1 = np.exp2(x)
    y2 = np.power(1/2, x)

    plt.plot(x, y1, label='(2^x, x)')
    plt.plot(x, y2, label='((1/2)^x, x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('문제 1번')
    plt.legend()
    plt.grid(True)

    plt.show()
    
    #수행시간 확인 함수입니다. 함수 호출을 지우지 말아주세요
    time_check()

def exam_2():
    print("문제2번")
    #코딩은 여기부터 시작해주세요
    x = np.linspace(0, 10 * np.pi, 500)

    a = (x + 1) ** 2
    b = 2 * (x + 1)
    
    sin_a = np.sin(a)
    cos_b = np.cos(b)
    sin_a_plus_b = np.sin(a + b)

    # sin(a) 그래프
    plt.plot(x, sin_a, label='sin(a)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin(a)')
    plt.legend()
    plt.grid(True)

    plt.show()

    # cos(b) 그래프
    plt.plot(x, cos_b, label='cos(b)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('cos(b)')
    plt.legend()
    plt.grid(True)

    plt.show()

    # sin(a+b) 그래프
    plt.plot(x, sin_a_plus_b, label='sin(a+b)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin(a+b)')
    plt.legend()
    plt.grid(True)

    plt.show()
    
    #수행시간 확인 함수입니다. 함수 호출을 지우지 말아주세요
    time_check()

def exam_3():
    print("문제3번")
    #코딩은 여기부터 시작해주세요
    x = int(input("숫자를 입력하세요: "))

    A = set(range(1, x + 1, 2))
    B = set(range(1, x + 1, 3))
    C = set(2 ** i for i in range(x.bit_length()) if 2 ** i <= x)

    print("집합 A =", A)
    print("집합 B =", B)
    print("집합 C =", C)
    D = set()
    
    for value1 in B:
        for value2 in C:
            D.add((value1, value2))

    print("곱집합 BxC =", D)

    #수행시간 확인 함수입니다. 함수 호출을 지우지 말아주세요
    time_check()
    
def exam_4():
    print("문제4번")
    #코딩은 여기부터 시작해주세요
    postitive_resp = [14540, 14250]
    negative_resp = [1120, 170090]

    print("\t A보균자 \t A미보균자")
    print("양성반응 \t{}".format(postitive_resp[0],postitive_resp[1]))
    print("음성반응 \t{}".format(negative_resp[0],negative_resp[1]))
    print("계 \t{} \t{}".format(postitive_resp[0] + negative_resp[0],postitive_resp[1] + negative_resp[1]))

    print("")
    print("(1) 선정한 사람이 A균 미보균자일 때, 이 사람이 양성 반응을 보일 확률")
    prob_non_a = (postitive_resp[1] + negative_resp[1]) / 200000
    prob_non_a_positive = postitive_resp[1] / 200000
    print("(1)번 답: {:.4f}".format(prob_non_a_positive / prob_non_a))

    print("")
    print("(2) 선정한 사람이 A균 보균자 일 때, 이 사람이 음성 반응을 보일 확률 구하여 화면에 출력하기")
    prob_a = (postitive_resp[0] + negative_resp[0]) / 200000
    prob_a_negative = negative_resp[0] / 200000
    print("(2)번 답: {:.4f}".format(prob_a_negative / prob_a))

    #수행시간 확인 함수입니다. 함수 호출을 지우지 말아주세요
    time_check()
    
def exam_5():
    print("문제5번")
    #코딩은 여기부터 시작해주세요
    A = np.array([[1, -1, 2],
                  [2, 1, -3],
                  [4, 1, 1]])

    B = np.array([[3, -1, -2],
                  [-4, 2, 1],
                  [1, 4, -3]])

    print("(1) A+B를 계산하여 출력합니다.")
    print(A + B)

    AB = np.dot(A, B)
    AB_transposed = np.transpose(AB)
    print("(2) (A x B)T를 계산하여 출력합니다.")
    print(AB_transposed)

    AB_inverse = np.linalg.inv(AB)
    print("(3) (A x B)의 역행렬을 계산하여 출력합니다.")
    print(AB_inverse)


    #수행시간 확인 함수입니다. 함수 호출을 지우지 말아주세요
    time_check()


def final_exam(): #기말고사 문제 메뉴 입니다
    
    exam_1()
    exam_2()
    exam_3()
    exam_4()
    exam_5()
    
final_exam()
    
    
