from scipy.optimize import newton

def f(x):
    return x**2 - 4*x + 3

def g(y, x):
    return f(x) - y

#fonction permettant de calculer l'inverse
def f_inv(f,y):
    x0 = 0
    g_with_fixed_y = lambda x: f(x) - y
    x = newton(g_with_fixed_y, x0)
    return x

l = [34, 29, 32, 34, 34, 34, 32, 34, 36, 32, 30, 32, 34, 31, 33, 36, 33, 34, 32, 31, 31, 32, 34, 33, 34, 33, 33, 32, 33, 30, 34, 34, 33, 25, 32, 30, 13, 27, 27, 25, 26, 27, 28, 30, 31, 30, 28, 29, 29, 28, 29, 29, 26, 32, 19, 18, 19, 30, 28, 27, 26, 29, 30, 31, 20, 27, 26, 29, 29, 18, 32, 33, 21, 33, 35, 32, 34, 33, 32, 35, 31, 30, 34, 26, 31, 30, 27, 28, 29, 18, 30, 26, 27, 14, 26, 28, 28, 29, 27, 27, 25, 16, 20, 19, 16, 20, 18, 31, 27, 29, 23, 16, 27, 29, 26, 27, 25, 27, 19, 19, 15, 25, 22, 19, 22, 26, 27, 16, 25, 27, 27, 26, 19, 25, 23, 21, 22, 24, 26, 24, 13, 22, 23, 26, 27, 29, 31, 30, 29, 27, 27, 28, 26, 28, 29, 33, 33, 31, 36, 31, 37, 36, 36, 37, 35, 19, 23, 30, 27, 30, 33, 41, 31, 40, 40, 39, 38, 37, 38, 38, 37, 36, 35, 33, 34, 34, 32, 32, 26, 16, 33, 32, 31, 33, 22, 29, 35, 34, 33, 32, 32, 20, 29, 17, 31, 31, 30, 31, 32, 31, 31, 22, 33, 32, 34, 32, 31, 32, 33, 33, 32, 31, 29, 28, 29, 31, 26, 30, 32, 32, 32, 30, 26, 27, 17, 26, 28, 29, 30, 31, 32, 30, 29, 30, 32, 32, 33, 33, 33, 24, 33, 32, 32, 33, 32, 29, 30, 31, 28, 27, 34, 27, 32, 31, 33, 31, 27, 29, 29, 32, 27, 28, 30, 31, 32, 27, 25, 32, 32, 29, 30, 31, 35, 33, 32, 32, 33, 33, 35, 29, 35, 34, 18, 35, 22, 36, 36, 31, 36, 34, 33, 29, 32, 32, 30, 30, 35, 35, 35, 17, 30, 31, 31, 32, 33, 35, 37, 36, 36, 36, 35, 34, 33, 30, 33, 28, 29, 32, 28, 28, 29, 30, 24, 27, 30, 20, 16, 16, 26, 24, 24, 23, 23, 26, 26, 28, 29, 29, 27, 27, 18, 29, 29, 23, 29, 25, 25, 17, 15, 28, 25, 28, 30, 32, 33, 33, 36, 37, 37, 37, 34, 19, 38, 38, 37, 37, 36, 37, 34, 36, 32, 31, 30, 32, 36, 30, 18, 33, 33, 31, 18, 33, 31, 28, 30, 29, 27, 28, 28, 27, 28, 26, 19, 16, 22, 18, 15, 17, 16, 17, 16, 19, 19, 22, 19, 21, 22, 20, 24, 26, 18, 17, 19, 19, 17, 19, 20, 19, 19, 19, 23, 26, 26, 27, 28, 28, 28, 27, 28, 24, 30, 30, 30, 30, 29, 30, 31, 24, 30, 19, 29, 32, 34, 31, 33, 28, 36, 35, 37, 37, 37, 36, 33, 33, 35, 32, 32, 35, 35, 34, 34, 33, 33, 24, 33, 36, 37, 37, 34, 36, 36, 31, 35, 34, 31, 34, 32, 34, 33, 31, 29, 32, 32, 32, 31, 30, 17, 30, 31, 31, 31, 32, 33, 23, 34, 29, 33, 31, 31, 32, 32, 29, 34, 27, 32, 32, 31, 30, 16, 20, 20, 31, 26, 20, 23, 31, 26, 33, 32, 32, 30, 30, 18, 27, 29, 26, 30, 23, 31, 28, 27, 28, 29, 29, 19, 30, 31, 31, 30, 19, 33, 33, 32, 18, 13, 28, 28, 17, 24, 26, 15, 28, 16, 26, 28, 29, 28, 27, 30, 30, 16, 31, 32, 34, 17, 34, 22, 16, 33, 30, 30, 31, 32, 30, 27, 26, 29, 30, 26, 14, 17, 25, 27, 26, 31, 33, 34, 34, 16, 32, 32, 32, 32, 28, 21, 19, 28, 26, 27, 27, 18, 17, 21, 19, 19, 16, 19, 20, 22, 11, 15, 23, 28, 25, 24, 23, 20, 21, 23, 24, 27, 27, 31, 31, 34, 26, 25, 21, 22, 31, 18, 17, 25, 31, 22, 33, 32, 15, 31, 27, 30, 30, 19, 23, 22, 24]

# Trouver l'inverse de f pour la valeur cible y = 1
y = 1
x0 = 0
g_with_fixed_y = lambda x: g(y, x)
x = newton(g_with_fixed_y, x0)
print("f({}) = {}".format(x, y))

g_with_fixed_y = lambda x: g(y, x)
x = f_inv(f,y)
print("f_inv({}) = {}".format(x, y))