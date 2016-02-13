import copy
import math
import numpy as np
import random
import sys

class GoalFunction():

    def __init__(self, f, grad, hess=None, init=None):
        self.f = f
        self.grad = grad
        self.init = init
        self.hess = hess
        self.e_counter = 0
        self.g_counter = 0
        self.h_counter = 0
        self.e_y = dict()
        self.g_y = dict()
        self.h_y = dict()

    def eval(self, x):
        self.e_counter += 1
        if not self.e_y.get(str(x)):
            self.e_y[str(x)] = self.f(x)

        return self.f(x)

    def gradient(self, x):
        self.g_counter += 1
        if not self.g_y.get(str(x)):
            self.g_y[str(x)] = self.grad(x)

        return self.grad(x)

    def hessian(self, x):
        self.h_counter += 1
        if not self.h_y.get(str(x)):
            self.h_y[str(x)] = self.hess(x)

        return self.hess(x)

    def e_count(self):
        return self.e_counter

    def g_count(self):
        return self.g_counter

    def h_count(self):
        return self.h_counter

    def reset(self):
        self.e_counter = 0
        self.g_counter = 0
        self.h_counter = 0


class TransformedFunction():

    def __init__(self, f, u, ls, t, init, hs=None):
        self.f = f
        self.u = u
        self.ls = ls
        self.t = t
        self.init = init
        self.hs = hs

    def eval(self, x):
        return self.u(self.f, x, self.t, self.ls, self.hs)
        
    def set_init(self, init):
        self.init = init

    def set_t(self, t):
        self.t = t


def f_lambda(gf, x, v):
    return lambda l : gf.eval(np.array(x) + l * np.array(v))

def f1(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def f1_g(x):
    d1 = 400*x[0] * (x[0]**2 - x[1]) + 2*(x[0] - 1)
    d2 = 200 * (x[1] - x[0]**2)
    return [d1, d2]

def f1_h(x):
    d11 = 400*(3*x[0]**2 - x[1]) + 2
    d12 = -400*(x[0])
    d21 = -400*(x[0])
    d22 = 200
    return np.linalg.inv([[d11, d12],[d21,d22]]).tolist()

def f2(x):
    return (x[0]-4)**2 + 4*(x[1]-2)**2

def f2_g(x):
    d1 = 2*(x[0] - 4)
    d2 = 8*(x[1] - 2)
    return [d1, d2]

def f2_h(x):
    d11 = 2
    d12 = 0
    d21 = 0
    d22 = 8
    return np.linalg.inv([[d11, d12],[d21,d22]]).tolist()

def f3(x):
    return (x[0]-2)**2 + (x[1]+3)**2

def f3_g(x):
    d1 = 2*(x[0] - 2)
    d2 = 2*(x[1] + 3)
    return [d1, d2]

def f4(x):
    return (x[0]-3)**2 + x[1]**2

def f4_g(x):
    d1 = 2*(x[0] - 3)
    d2 = 2*x[1]
    return [d1, d2]

def l1(x):
    return x[1]-x[0]

def l2(x):
    return 2-x[0]

def l31(x):
    return x[0]+100

def l32(x):
    return x[1]+100

def l41(x):
    return 100-x[0]

def l42(x):
    return 100-x[1]

def l5(x):
    return 3-x[0]-x[1]

def l6(x):
    return 3+1.5*x[0]-x[1]


gf1 = GoalFunction(f1, f1_g, f1_h, init=[-1.9,2])
gf2 = GoalFunction(f2, f2_g, f2_h, init=[0.1, 0.3])
gf3 = GoalFunction(f3, f3_g, init=[0,0])
gf4 = GoalFunction(f4, f4_g, init=[0,0])


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


def golden_section_search(gf, xs):
    a, b = unimodal_interval(gf, xs, h=1)
    return golden_section_search1(gf, a, b)


def golden_section_search1(gf, a, b, e=10e-6):
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


def euclidean_norm(xs):
    return math.sqrt(sum(map(lambda x: x**2, xs)))


def gradient_descent(gf, e=10e-6, gss=False):
    x = copy.copy(gf.init)
    old = gf.eval(x)
    curr = old
    cnt = 0
    while euclidean_norm(gf.gradient(x)) > e:
        if curr != old:
            cnt = 0
        else:
            cnt += 1

        if cnt > 100:
            print "Does not converge!!!"
            return x

        grad = gf.gradient(x)

        if gss:
            gff = GoalFunction(f_lambda(gf, x, grad), None, init=[0])
            ratio = golden_section_search(gff, [0])
        else:
            ratio = -1

        x[0] += ratio * grad[0]
        x[1] += ratio * grad[1]

        old = curr
        curr = gf.eval(x)

    return x


def newton_raphson(gf, e=10e-6, gss=False):
    x = copy.copy(gf.init)
    old = gf.eval(x)
    curr = old
    cnt = 0

    grad = np.array(gf.gradient(x))
    hessian = np.array(gf.hessian(x))
    shift = np.dot(hessian, grad)
    
    while euclidean_norm(shift) > e:
        if curr != old:
            cnt = 0
        else:
            cnt += 1

        if cnt > 100:
            print "Does not converge!!!"
            return x

        grad = np.array(gf.gradient(x))
        hessian = np.array(gf.hessian(x))
        shift = np.dot(hessian, grad)
    
        if gss:
            gff = GoalFunction(f_lambda(gf, x, shift), None, init=[0])
            ratio = golden_section_search(gff, [0])
        else:
            ratio = -1

        x[0] += ratio * shift[0]
        x[1] += ratio * shift[1]

        old = curr
        curr = gf.eval(x)

    return x


def validate(x0, ls):
    for l in ls:
        if l(x0) < 0:
            return False
    return True


def get_centroid(xs, h):
    xc = []
    for i in range(len(xs[0])):
        xc.append(0)

    for i in range(len(xs)):
        if i == h:
            continue
        for j in range(len(xs[i])):
            xc[j] += xs[i][j]
    n = len(xs) - int(h != -1)
    xc = map(lambda x: x/float(n), xc)
    return xc


def get_max_index(gf, xs, h, e=10e-6):
    # tmp_max = gf.eval(xs[0])
    tmp_max = -10000000
    max_i = 0
    for i in range(len(xs)):
        x = xs[i]
        if gf.eval(x) - tmp_max > e and i != h:
            tmp_max = gf.eval(x)
            max_i = i

    return max_i


def reflection(xh, xc, alpha):
    tmp_xh = np.array(xh)
    tmp_c = np.array(xc)
    tmp_sol = (1 + alpha) * tmp_c - alpha * tmp_xh
    return tmp_sol.tolist()


def stop_value(gf, simplex, xc):
    sol = 0
    for i in range(len(simplex)):
        sol += (gf.eval(simplex[i]) - gf.eval(xc))**2

    return math.sqrt(sol / float(len(simplex)))


def box(gf, ls, ex=[-100,100], e=10e-6, alpha=1.3):
    x0 = gf.init
    if not validate(x0, ls) or x0[0] < ex[0] or x0[0] > ex[1] or x0[1] < ex[0] or x0[1] > ex[1]:
        print "Initial point not valid!"
        return
    xc = copy.copy(x0)
    n = len(x0)
    xs = []
    for i in range(2*n):
        xs.append([x0[0], x0[1]])

    for t in range(2*n):
        for i in range(n):
            r = random.random()
            xs[t][i] = ex[0] + r*(ex[1] - ex[0])
        
        while not validate(xs[t], ls):
            xs[t] = (0.5 * (np.array(xs[t]) + np.array(xc))).tolist()
        
        xc = get_centroid(xs, -1)
    
    old = gf.eval(xc)
    curr = old
    cnt = 0
    while True:
        if curr != old:
            cnt = 0
        else:
            cnt += 1

        if cnt > 100:
            print "Does not converge!!!"
            return xc

        h = get_max_index(gf, xs, -1)
        h2 = get_max_index(gf, xs, h)
        xc = get_centroid(xs, h)
        xr = reflection(xs[h], xc, alpha)

        # print h, gf.eval(xs[h])
        # print h2, gf.eval(xs[h2])

        for i in range(n):
            if xr[i] < ex[0]:
                xr[i] = ex[0]
            elif xr[i] > ex[1]:
                xr[i] = ex[1]

        while not validate(xr, ls):
            xr = (0.5 * (np.array(xr) + np.array(xc))).tolist()

        if gf.eval(xr) > gf.eval(xs[h2]):
            xr = (0.5 * (np.array(xr) + np.array(xc))).tolist()

        xs[h] = xr

        old = curr
        curr = gf.eval(xc)

        if stop_value(gf, xs, xc) <= e:
            return xc


def hooke_jeeves(gf, x0, dx=0.5, e=10e-6):
    xp = copy.copy(x0)
    xb = copy.copy(x0)

    while True:
        xn = explore(gf, xp, dx)

        if gf.eval(xn) < gf.eval(xb):
            for i in range(len(xp)):
                xp[i] = 2*xn[i] - xb[i]
            xb = xn
        else:
            dx = float(dx)/2
            xp = xb
        
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


def h(t, x):
    return t * (x[1]-1)**2


def u(f, x, t, ls, hs=None):
    sol = 0
    if hs != None:
        sol += hs[0](t, x)

    tmp = 0
    for l in ls:
        if l(x) <= 0:
            return float("inf")
        else:
            tmp += math.log(l(x))
        
    sol -= 1/t * tmp
    return sol + f(x)


def stop_iteration(x1, x2, e=10e-6):
    for i in range(len(x1)):
        if abs(x1[i]-x2[i]) > e:
            return False
    return True


def transform(gf, ls, hs=None, t=1.0, e=10e-6):
    gf_new = TransformedFunction(f=gf.f, u=u, ls=ls, t=t, init=gf.init, hs=hs)

    while True:
        old = gf_new.init
        curr = hooke_jeeves(gf_new, gf_new.init)

        t *= 10
        gf_new.set_t(t)
        gf_new.set_init(copy.copy(curr))

        if stop_iteration(old, curr):
            return curr, int(math.log(t, 10)+1)


def task1():
    print "Gradient descent starting from point {0} for function {1} without shift optimization.".format(gf3.init, 3)
    print "\t=> Found minimum in {0} in {1} iterations and {2} gradient calculations.\n".format(gradient_descent(gf3), gf3.e_count(), gf3.g_count())
    gf3.reset()

    print "Gradient descent starting from point {0} for function {1} with shift optimization.".format(gf3.init, 3)
    print "\t=> Found minimum in {0} in {1} iterations and {2} gradient calculations.\n".format(gradient_descent(gf3, gss=True), gf3.e_count(), gf3.g_count())
    gf3.reset()


def task2():
    print "Gradient descent starting from point {0} for function {1} with shift optimization.".format(gf1.init, 1)
    print "\t=> Found minimum in {0} in {1} iterations and {2} gradient calculations.\n".format(gradient_descent(gf1, gss=True), gf1.e_count(), gf1.g_count())
    gf1.reset()

    print "Newton-Raphson from point {0} for function {1} with shift optimization.".format(gf1.init, 1)
    print "\t=> Found minimum in {0} in {1} iterations, {2} gradient and {3} Hessian calculations.\n".format(newton_raphson(gf1, gss=True), gf1.e_count(), gf1.g_count(), gf1.h_count())
    gf1.reset()

    print "Gradient descent starting from point {0} for function {1} with shift optimization.".format(gf2.init, 2)
    print "\t=> Found minimum in {0} in {1} iterations and {2} gradient calculations.\n".format(gradient_descent(gf2, gss=True), gf2.e_count(), gf2.g_count())
    gf2.reset()

    print "Newton-Raphson from point {0} for function {1} with shift optimization.".format(gf2.init, 2)
    print "\t=> Found minimum in {0} in {1} iterations, {2} gradient and {3} Hessian calculations.\n".format(newton_raphson(gf2, gss=True), gf2.e_count(), gf2.g_count(), gf2.h_count())
    gf2.reset()


def task3():
    print "Box algorithm from point {0} for function {1}.".format(gf1.init, 1)
    print "\t=> Found minimum in {0} in {1} iterations.\n".format(box(gf1, ls=[l1,l2,l31,l32,l41,l42]), gf1.e_count())
    gf1.reset()

    print "Box algorithm from point {0} for function {1}.".format(gf2.init, 2)
    print "\t=> Found minimum in {0} in {1} iterations.\n".format(box(gf2, ls=[l1,l2,l31,l32,l41,l42]), gf2.e_count())
    gf2.reset()


def task4():
    for i, gf in enumerate([gf1, gf2]):
        print "Transforming function {1} from point {0} and minimizing using Hooke-Jeeves algorithm.".format(gf.init, i+1)
        sol, n = transform(gf, ls=[l1,l2,l31,l32,l41,l42])
        print "\t=> Found minimum in {0} in {1} iterations.\n".format(sol, n)
        gf.reset()


def task5():
    print "Transforming function {1} from point {0} and minimizing using Hooke-Jeeves algorithm.".format(gf4.init, 4)
    sol, n = transform(gf4, ls=[l5,l6], hs=[h])
    print "\t=> Found minimum in {0} in {1} iterations.\n".format(sol, n)
    gf4.reset()


N = sys.argv[1]
locals()["task" + str(N)]()
