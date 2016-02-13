import matplotlib.pyplot as plt
import numpy as np
import sys

class Matrix:
    epsilon = 1e-9
    
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        # self.matrix = [[0] * cols] * rows
        self.matrix = []
        for i in range(rows):
            self.matrix.append([0] * cols)

    def check_bounds(self, i, j):
        if i >= self.rows or j >= self.cols:
            raise Exception('Invalid index ({2},{3})! Matrix dimensions \
                             are {0} rows and {1} cols'.format(self.rows, self.cols, i, j))
        else:
            return True

    def set(self, i, j, val):
        self.check_bounds(i, j)
        self.matrix[i][j] = float(val)

    def get(self, i, j):
        self.check_bounds(i, j)
        return self.matrix[i][j]

    @staticmethod
    def input_from_file(filename):
        f = open(filename, 'r')
        rows, cols = 0, 0
        cols = len(f.readline().split())
        rows = sum(1 for line in f)

        new_m = Matrix(rows, cols)
        new_m.matrix = [map(float, line.split()) for line in open(filename, 'r')]
        f.close()
        return new_m

    @staticmethod
    def input_to_file(matrix, filename):
        f = open(filename, 'w')
        f.write(str(matrix))
        f.close()

    def __str__(self):
        s = ""
        for i in range(self.rows):
            for j in range(self.cols):
                s += str(self.get(i,j)) + " "
            if i + 1 < self.rows:
                s += " | "
        return s

    @staticmethod
    def copy(old):
        new_m = Matrix(old.rows, old.cols)
        for i in range(old.rows):
            for j in range(old.cols):
                new_m.set(i, j, old.get(i,j))
                # print "({0}, {1}) -> {2}".format(i, j, new_m.get(i,j))

        return new_m

    def multiply(self, factor):
        new_m = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_m.set(i, j, self.get(i, j) * factor)
        return new_m
        
        # for i in range(self.rows):
        #     for j in range(self.cols):
        #         self.set(i, j, self.get(i, j) * factor)
        # return self

    def inverse(self):
        inv = Matrix(self.cols, self.rows)
        A = Matrix.copy(self)
        A, P = A.LUP()

        for i in range(A.rows):
            b = Matrix(A.rows, 1)
            b.matrix[i][0] = 1
            b = Matrix.permute(b, P)
            y = A.substitution_forward(b)
            x = A.substitution_backward(y)
            for j in range(A.cols):
                inv.set(j, i, x.get(j, 0))

        return inv

    def __add__(self, other):
        new_m = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                new_m.set(i, j, self.get(i, j) + other.get(i, j))
        return new_m

        # for i in range(self.rows):
        #     for j in range(self.cols):
        #         self.set(i, j, self.get(i, j) + other.get(i, j))
        # return self

    def __sub__(self, other):
        new_m = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                new_m.set(i, j, self.get(i, j) - other.get(i, j))
        return new_m

        # for i in range(self.rows):
        #     for j in range(self.cols):
        #         self.set(i, j, self.get(i, j) - other.get(i, j))
        # return self

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return self.multiply(float(other))
        
        if (self.cols != other.rows):
            raise Exception('Matrices not compatible for multiplication!')

        new_m = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                new_m.set(i, j, 0)
                for k in range(self.cols):
                    new_m.set(i, j, new_m.get(i, j) + self.get(i, k)*other.get(k,j))
        return new_m

    def __div__(self, other):
        if other == 0:
            raise Exception('Cannot divide by zero!')
        return self.__mul__(1/(float(other)))

    def __invert__(self):
        new_m = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                new_m.set(j, i, self.get(i, j))
        return new_m

    def __eq__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.get(i, j) - other.get(i, j)) > self.epsilon:
                    return False
        return True

    def substitution_forward(self, v):
        # Solve Ly = Pb
        vector = Matrix.copy(v)
        if (vector.cols > 1):
            raise Exception('Input parameter must have dimensions n x 1!')
        if (vector.rows != self.rows):
            raise Exception('Invalid dimensions for input vector!')

        for i in range(self.rows-1):
            for j in range(i+1, self.rows):
                val = vector.get(j, 0) - (self.get(j, i) * vector.get(i, 0))
                vector.set(j, 0, val)
        return vector

    def substitution_backward(self, v):
        # Solve Ux = y
        vector = Matrix.copy(v)
        if (vector.cols > 1):
            raise Exception('Input parameter must have dimensions n x 1!')
        if (vector.rows != self.rows):
            raise Exception('Invalid dimensions for input vector!')

        for i in reversed(xrange(self.rows)):
            if abs(self.get(i, i)) < self.epsilon:
                raise Exception('Aborted, matrix is singular!')

            vector.set(i, 0, vector.get(i, 0) / self.get(i, i))
            for j in range(i):
                val = vector.get(j, 0) - (self.get(j, i) * vector.get(i, 0))
                vector.set(j, 0, val)
        return vector

    def LU(self):
        if self.rows != self.cols:
            raise Exception('Matrix must be square to perform LU decomposition!')

        for i in range(self.rows-1):
            for j in range(i+1, self.rows):
                if abs(self.get(i, i)) < self.epsilon:
                    raise Exception('Aborted, matrix is singular!')

                self.set(j, i, self.get(j, i) / self.get(i, i))
                for k in range(i+1, self.rows):
                    val = self.get(j, k) - (self.get(j, i) * self.get(i, k))
                    self.set(j, k, val)
        return self

    def LUP(self):
        if self.rows != self.cols:
            raise Exception('Matrix must be square to perform LUP decomposition!')

        vector = Matrix(1, self.rows)
        for i in range(self.rows):
            vector.set(0, i, i)

        for i in range(self.rows-1):
            pivot = i
            v_i = int(vector.get(0, i))

            for j in range(i+1, self.rows):
                v_pivot = int(vector.get(0, pivot))
                v_j = int(vector.get(0, j))

                if abs(self.get(v_j, i)) > abs(self.get(v_pivot, i)):
                    pivot = j

            tmp = int(vector.get(0, pivot))
            vector.set(0, pivot, v_i)
            vector.set(0, i, tmp)

            v_i = int(vector.get(0, i))
            v_pivot = int(vector.get(0, pivot))

            for j in range(i+1, self.rows):
                v_j = int(vector.get(0, j))
                if abs(self.get(v_i, i)) < self.epsilon:
                    raise Exception('Aborted, matrix is singular!')

                val = self.get(v_j,i) / self.get(v_i,i)
                self.set(v_j, i, val)
                for k in range(i+1, self.rows):
                    val = self.get(v_j, k) - (self.get(v_j, i) * self.get(v_i, k))
                    self.set(v_j, k, val)

        # permute matrix
        self = Matrix.permute(self, vector)
        return self, vector

    @staticmethod
    def permute(b, P):
        new_B = Matrix(b.rows, b.cols)
        cnt = 0
        for index in P.matrix[0]:
            new_B.matrix[cnt] = b.matrix[int(index)]
            cnt += 1

        return new_B


def runge_kutta(A, x, T, t_max):
    x_k = Matrix.copy(x)
    t = 0
    ts = []
    x1s = []
    x2s = []
    while t <= t_max:
        m1 = A * x_k
        m2 = A * (x_k + m1 * T * 0.5)
        m3 = A * (x_k + m2 * T * 0.5)
        m4 = A * (x_k + m3 * T)
        x_k = x_k + (m1 + m2*2 + m3*2 + m4) * T / 6
        t += T
        print t, x_k
        ts.append(t)
        x1s.append(x_k.matrix[0][0])
        x2s.append(x_k.matrix[1][0])

    plt.plot(ts, x1s, label='x1')
    plt.plot(ts, x2s, label='x2')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return x_k


def trapese(A, x, T, t_max):
    U = Matrix(2,2)
    U.matrix = [[1, 0], [0, 1]]
    R = (U - A * T * 0.5).inverse() * (U + A * T * 0.5)

    x_k = Matrix.copy(x)
    t = 0
    ts = []
    x1s = []
    x2s = []
    while t <= t_max:
        x_k = R * x_k
        t += T
        print t, x_k
        ts.append(t)
        x1s.append(x_k.matrix[0][0])
        x2s.append(x_k.matrix[1][0])

    plt.plot(ts, x1s, label='x1')
    plt.plot(ts, x2s, label='x2')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return x_k


def task1(trap=True):
    A = Matrix(2,2)
    A.matrix = [[0, 1],[-1, 0]]
    x = Matrix(2,1)
    x.matrix = [[0],[1]]
    if trapese:
        print trapese(A, x, 0.1, 10)
    else:
        print runge_kutta(A, x, 0.1, 10)


def task2(trap=True):
    A = Matrix(2,2)
    A.matrix = [[0, 1],[-200, -102]]
    x = Matrix(2,1)
    x.matrix = [[1],[-2]]
    if trapese:
        print trapese(A, x, 0.1, 10)
    else:
        print runge_kutta(A, x, 0.1, 10)


N = sys.argv[1]
trap = len(sys.argv) > 2

locals()["task" + str(N)](trap)
