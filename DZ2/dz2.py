import copy
import math
import numpy
import random
import sys

class GoalFunction():

    def __init__(self, f, init=None):
        self.f = f
        self.init  = init
        self.counter = 0
        self.y = dict()

    def eval(self, x):
        self.counter += 1
        if not self.y.get(str(x)):
            self.y[str(x)] = self.f(x)

        return self.f(x)

    def count(self):
        return self.counter

    def reset(self):
        self.counter = 0


def f1(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def f2(x):
    return (x[0]-4)**2 + 4*(x[1]-2)**2

def f3(xs):
    sol = 0
    for x,i in enumerate(xs):
        sol += (x-i)**2
    return sol

def f4(x):
    return abs((x[0]-x[1])*(x[0]+x[1])) + math.sqrt(x[0]**2 + x[1]**2)

def f6(xs):
    tmp_sum = sum(map(lambda x: x**2, xs))
    return 0.5 + (math.sin(math.sqrt(tmp_sum))**2-0.5)/((1+0.001*tmp_sum)**2)


def unimodal_interval(gf, xs, h):
    point = xs[0]
    l = point - h
    r = point + h;
    m = point
    step = 1

    fm = gf.eval(point)
    fl = gf.eval(l)
    fr = gf.eval(r)

    if fm < fr and fm < fl:
        return l, r

    elif fm > fr:
        while True:
            l = m
            m = r
            fm = fr
            step *= 2
            r = point + h * step
            fr = gf.eval(r)
            if not fm > fr:
                break
    else:
        while True:
            r = m
            m = l
            fm = fl
            step *= 2
            l = point - h * step
            fl = gf.eval(l)
            if not fm > fl:
                break

    return l, r


def golden_section_search(gf, a, b, e=10e-6):
    k = 0.5 * (math.sqrt(5) - 1)
    c = b - k * (b - a)
    d = a + k * (b - a)
    fc = gf.eval(c)
    fd = gf.eval(d)

    while (b - a) > e:
        if fc < fd:
            b = d
            d = c
            c = b - k * (b - a)
            fd = fc
            fc = gf.eval(c)
        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            fd = gf.eval(d)

        # print a,b,c,d
        # print gf.eval(a), gf.eval(b), gf.eval(c), gf.eval(d)

    return (b + a)/2


def simplex_nelder_mead(gf, x0, mov_simplex=1, alpha=1, beta=0.5, gamma=2, sigma=0.5, e=10e-6):
    simplex = [x0]
    for i in range(len(x0)):
        tmp_x = copy.copy(x0)
        tmp_x[i] = x0[i] + mov_simplex
        simplex.append(tmp_x)

    cnt = 0
    while True:
        h, l = get_max_min_indices(gf, simplex)
        xc = get_centroid(simplex, h)
        xr = reflection(simplex[h], xc, alpha)

        if cnt > 10000:
            return xc
        else:
            cnt += 1

        if gf.eval(xr) < gf.eval(simplex[l]):
            xe = expansion(xr, xc, gamma)
            if gf.eval(xe) < gf.eval(simplex[l]):
                simplex[h] = xe
            else:
                simplex[h] = xr
        else:
            if check_condition(gf, simplex, xr, h):
                if gf.eval(xr) < gf.eval(simplex[h]):
                    simplex[h] = xr

                xk = contraction(simplex[h], xc, beta)

                if gf.eval(xk) < gf.eval(simplex[h]):
                    simplex[h] = xk
                else:
                    simplex = move_points(simplex, simplex[l], sigma)
            else:
                simplex[h] = xr

        # print xc
        # print gf.eval(xc)

        if stop_value(gf, simplex, xc) <= e:
            return xc


def reflection(xh, xc, alpha):
    tmp_xh = numpy.array(xh)
    tmp_c = numpy.array(xc)
    tmp_sol = (1 + alpha) * tmp_c - alpha * tmp_xh
    return tmp_sol.tolist()


def expansion(xr, xc, gamma):
    tmp_xr = numpy.array(xr)
    tmp_c = numpy.array(xc)
    tmp_sol = (1 - gamma) * tmp_c + gamma * tmp_xr  
    return tmp_sol.tolist()


def contraction(xh, xc, beta):
    tmp_xh = numpy.array(xh)
    tmp_c = numpy.array(xc)
    tmp_sol = (1 - beta) * tmp_c + beta * tmp_xh
    return tmp_sol.tolist()


def get_centroid(xs, h):
    xc = [0] * len(xs[0])
    for i in range(len(xs)):
        if i == h:
            continue
        for j in range(len(xs[i])):
            xc[j] += xs[i][j]
    xc = map(lambda x: x/float(len(xc)), xc)
    return xc


def stop_value(gf, simplex, xc):
    sol = 0
    for i in range(len(simplex)):
        sol += (gf.eval(simplex[i]) - gf.eval(xc))**2

    return math.sqrt(sol / float(len(simplex)))


def move_points(xs, mov, sigma):
    new_xs = []
    tmp_mov = numpy.array(mov)
    for x in xs:
        tmp_x = numpy.array(x)
        tmp_sol = sigma * (tmp_x + tmp_mov)
        new_xs.append(tmp_sol.tolist())

    return new_xs


def check_condition(gf, simplex, xr, h):
    for i in range(len(simplex)):
        if i == h:
            continue
        if gf.eval(xr) <= gf.eval(simplex[i]):
            return False
    return True


def get_max_min_indices(gf, xs, e=10e-6):
    tmp_max = gf.eval(xs[0])
    tmp_min = gf.eval(xs[0])
    max_i = min_i = 0
    for i in range(len(xs)):
        x = xs[i]
        if gf.eval(x) - tmp_max > e:
            tmp_max = gf.eval(x)
            max_i = i
        if tmp_min - gf.eval(x) > e:
            tmp_min = gf.eval(x)
            min_i = i

    return max_i, min_i


def hooke_jeeves(gf, x0, dx=0.5, e=10e-6):
    xp = xb = x0

    while True:
        xn = explore(gf, xp, dx)
        if gf.eval(xn) < gf.eval(xb):
            for i in range(len(xp)):
                xp[i] = 2*xn[i] - xb[i]
            xb = xn
        else:
            dx = float(dx)/2
            xp = xb

        # print xb, xp, xn
        # print gf.eval(xb), gf.eval(xp), gf.eval(xn)

        if dx < e:
            return xb


def explore(gf, xp, dx):
    x = copy.copy(xp)
    for i in range(len(xp)):
        p = gf.eval(x)
        x[i] = x[i] + dx
        n = gf.eval(x)
        if n > p:
            x[i] = x[i] - 2*dx
            n = gf.eval(x)
            if n > p:
                x[i] = x[i] + dx
    return x


def f_golden(xs):
    if not type(xs) == list:
        return (xs-3)**2

    return sum(map(lambda x: (x-3)**2, xs))
    

def task1():
    for i in range(-20, 20, 5):
        gf = GoalFunction(f_golden, init=[i])
        print "Golden section starting from point {0}.".format(i)
        a,b = unimodal_interval(gf, gf.init, 1)
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(golden_section_search(gf, a, b), gf.count())

        gf.reset()
        print "Nelder-Mead simplex starting from point {0}".format(i)
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(simplex_nelder_mead(gf, gf.init), gf.count())

        gf.reset()
        print "Hooke-Jeeves starting from point {0}".format(i)
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(hooke_jeeves(gf, gf.init), gf.count())


def task2():
    gf1 = GoalFunction(f1, init=[-1.9,2])
    gf2 = GoalFunction(f2, init=[0.1, 0.3])
    gf3 = GoalFunction(f3, init=[0,0,0,0,0])
    gf4 = GoalFunction(f4, init=[5.1, 1.1])

    for i, gf in enumerate([gf1, gf2, gf3, gf4]):
        print "Nelder-Mead simplex starting from point {0} for function {1}.".format(gf.init, i+1)
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(simplex_nelder_mead(gf, gf.init), gf.count())
        gf.reset()

    print "#" * 100

    for i, gf in enumerate([gf1, gf2, gf3, gf4]):
        print "Hooke-Jeeves starting from point {0} for function {1}.".format(gf.init, i+1)
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(hooke_jeeves(gf, gf.init), gf.count())
        gf.reset()


def task3():
    gf = GoalFunction(f4, init=[5,5])
    
    print "Nelder-Mead simplex starting from point {0} for function {1}.".format(gf.init, 4)
    print "\t=> Found minimum in {0} in {1} iterations.\n".format(simplex_nelder_mead(gf, gf.init), gf.count())
    gf.reset()
    
    print "Hooke-Jeeves starting from point {0} for function {1}.".format(gf.init, 4)
    print "\t=> Found minimum in {0} in {1} iterations.\n".format(hooke_jeeves(gf, gf.init), gf.count())
    gf.reset()


def task4():
    gf = GoalFunction(f4, init=[0.5, 0.5])
    
    for i in range(20):
        print "Nelder-Mead simplex starting from point {0} for function {1} with moving step {2}.".format(gf.init, 4, i+1)
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(simplex_nelder_mead(gf, gf.init, mov_simplex=i+1), gf.count())
        gf.reset()
    
    print "#" * 100

    gf = GoalFunction(f4, init=[20, 20])
    for i in range(20):
        print "Nelder-Mead simplex starting from point {0} for function {1} with moving step {2}.".format(gf.init, 4, i+1)
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(simplex_nelder_mead(gf, gf.init, mov_simplex=i+1), gf.count())
        gf.reset()


def task5():
    found = 0
    for i in range(5000):
        x = random.randrange(-50, 51)
        y = random.randrange(-50, 51)

        gf = GoalFunction(f6, init=[x,y])
        sol = hooke_jeeves(gf, gf.init)

        if gf.eval(sol) < 10e-4:
            found += 1
        gf.reset()

    print found


N = sys.argv[1]
locals()["task" + str(N)]()