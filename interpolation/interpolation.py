import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(x) + 5 * x

def equally_spaced_nodes(a, b, n):
    return np.linspace(a, b, n)

def optimal_nodes(a, b, n):
    nodes = np.zeros(n)
    for i in range(n):
        xi = 0.5 * ((b - a) * np.cos(((2 * i + 1) / (2 * (n + 1))) * np.pi) + (b + a))
        nodes[i] = xi
    return nodes


def koeff_polynom_lagrange(x_values, k, x, n):
    l = 1.0
    for i in range(n):
        if i != k:
            l *= (x - x_values[i]) / (x_values[k] - x_values[i])
    return l

def res_polynom_lagrange(x_values, y_values, x, n):
    res = 0.0
    for k in range(n):
        res += koeff_polynom_lagrange(x_values, k, x, n) * y_values[k]
    return res

def differences_newton(x_values, y_values, n):
    table = np.zeros((n, n))
    for i in range(n):
        table[i][0] = y_values[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_values[i + j] - x_values[i])

    return table

def res_polynom_newton(table, x_values, n, x):
    result = table[0][0]

    temp = 1.0
    for i in range(1, n):
        temp *= (x - x_values[i - 1])
        result += table[0][i] * temp

    return result


def linear_spline_coeffs(args, vals, n):
    coeffs_size = 2 * (n - 1)
    coeffs = np.zeros(coeffs_size)

    for i in range(n - 1):
        A = np.array([[args[i], 1], [args[i + 1], 1]])
        B = np.array([vals[i], vals[i + 1]])

        solve = np.linalg.solve(A, B)
        coeffs[2 * i] = solve[0]
        coeffs[2 * i + 1] = solve[1]

    return coeffs

def linear_spline(x, args, coeffs, n):
    for i in range(n - 1):
        if (args[i] <= x <= args[i + 1]) or (args[i + 1] <= x <= args[i]):
            return coeffs[2 * i] * x + coeffs[2 * i + 1]
    return 0.0

def quadratic_spline_coeffs(x_vals, y_vals, n, cond):
    coeffsSize = 3 * (n - 1)
    X = np.zeros((coeffsSize, coeffsSize))
    y = np.zeros((coeffsSize, 1))

    for i in range(n - 1):
        y[3 * i][0] = y_vals[i]
        y[3 * i + 1][0] = y_vals[i + 1]

    y[coeffsSize - 1][0] = cond

    for i in range(n - 1):
        X[3 * i][3 * i] = x_vals[i] ** 2
        X[3 * i][3 * i + 1] = x_vals[i]
        X[3 * i][3 * i + 2] = 1

        X[3 * i + 1][3 * i] = x_vals[i + 1] ** 2
        X[3 * i + 1][3 * i + 1] = x_vals[i + 1]
        X[3 * i + 1][3 * i + 2] = 1

        if i < n - 2:
            X[3 * i + 2][3 * i] = 2 * x_vals[i + 1]
            X[3 * i + 2][3 * i + 1] = 1
            X[3 * i + 2][3 * i + 3] = -2 * x_vals[i + 1]
            X[3 * i + 2][3 * i + 4] = -1
        else:
            X[3 * i + 2][3 * i] = 2 * x_vals[i + 1]
            X[3 * i + 2][3 * i + 1] = 1

    return np.linalg.lstsq(X, y, rcond=None)[0]

def quadratic_spline_value(x, coeffs, x_vals, n):
    for i in range(n - 1):
        if x_vals[i] <= x <= x_vals[i + 1]:
            a = coeffs[3 * i][0]
            b = coeffs[3 * i + 1][0]
            c = coeffs[3 * i + 2][0]
            return a * x ** 2 + b * x + c
    return 0.0

def cubic_spline_coeffs(args, vals, n):
    h = np.diff(args)
    b = np.diff(vals) / h

    A = np.zeros((n, n))
    rhs = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * (b[i] - b[i - 1])

    c = np.linalg.solve(A, rhs)

    d = np.zeros(n - 1)
    b = np.zeros(n - 1)
    a = vals[:-1]

    for i in range(n - 1):
        b[i] = (vals[i + 1] - vals[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    coeffs = np.zeros((4 * (n - 1), 1))
    for i in range(n - 1):
        coeffs[4 * i] = a[i]
        coeffs[4 * i + 1] = b[i]
        coeffs[4 * i + 2] = c[i]
        coeffs[4 * i + 3] = d[i]

    return coeffs

def cubic_spline_value(x, args, coeffs, n):
    for i in range(n - 1):
        if args[i] <= x <= args[i + 1]:
            dx = x - args[i]
            return (coeffs[4 * i] +
                    coeffs[4 * i + 1] * dx +
                    coeffs[4 * i + 2] * dx**2 +
                    coeffs[4 * i + 3] * dx**3)
    return 0.0


def calculate_max_deviation(f, a, b, n, m):
    x_test = equally_spaced_nodes(a, b, m)
    y_test = f(x_test)

    x_values_eq = equally_spaced_nodes(a, b, n)
    y_values_eq = f(x_values_eq)
    x_values_opt = optimal_nodes(a, b, n)
    y_values_opt = f(x_values_opt)

    # Lagrange Polynomial
    max_deviation_lagrange_eq = max(abs(f(ti) - res_polynom_lagrange(x_values_eq, y_values_eq, ti, n)) for ti in x_test)
    max_deviation_lagrange_opt = max(abs(f(ti) - res_polynom_lagrange(x_values_opt, y_values_opt, ti, n)) for ti in x_test)

    # Newton Polynomial
    table_eq = differences_newton(x_values_eq, y_values_eq, n)
    table_opt = differences_newton(x_values_opt, y_values_opt, n)
    max_deviation_newton_eq = max(abs(f(ti) - res_polynom_newton(table_eq, x_values_eq, n, ti)) for ti in x_test)
    max_deviation_newton_opt = max(abs(f(ti) - res_polynom_newton(table_opt, x_values_opt, n, ti)) for ti in x_test)

    return max_deviation_lagrange_eq, max_deviation_lagrange_opt, max_deviation_newton_eq, max_deviation_newton_opt

def calculate_max_deviation_splines(f, a, b, n, m):
    x_test = np.linspace(a, b, m)
    y_test = f(x_test)

    x_nodes_eq = np.linspace(a, b, n)
    y_nodes_eq = f(x_nodes_eq)
    x_nodes_opt = 0.5 * ((b - a) * np.cos(((2 * np.arange(n) + 1) / (2 * (n + 1))) * np.pi) + (b + a))
    y_nodes_opt = f(x_nodes_opt)

    # Linear Spline
    coeffs_eq_linear = linear_spline_coeffs(x_nodes_eq, y_nodes_eq, n)
    coeffs_opt_linear = linear_spline_coeffs(x_nodes_opt, y_nodes_opt, n)

    max_deviation_linear_eq = max(abs(f(ti) - linear_spline(ti, x_nodes_eq, coeffs_eq_linear, n)) for ti in x_test)
    max_deviation_linear_opt = max(abs(f(ti) - linear_spline(ti, x_nodes_opt, coeffs_opt_linear, n)) for ti in x_test)

    # Quadratic Spline
    coeffs_eq_quadratic = quadratic_spline_coeffs(x_nodes_eq, y_nodes_eq, n, 0)
    coeffs_opt_quadratic = quadratic_spline_coeffs(x_nodes_opt, y_nodes_opt, n, 0)

    max_deviation_quadratic_eq = max(abs(f(ti) - quadratic_spline_value(ti, coeffs_eq_quadratic, x_nodes_eq, n)) for ti in x_test)

    max_deviation_quadratic_opt = max(abs(f(ti) - quadratic_spline_value(ti, coeffs_opt_quadratic, x_nodes_opt, n)) for ti in x_test)

    # Cubic Spline
    coeffs_eq_cubic = cubic_spline_coeffs(x_nodes_eq, y_nodes_eq, n)
    coeffs_opt_cubic = cubic_spline_coeffs(x_nodes_opt, y_nodes_opt, n)

    max_deviation_cubic_eq = max(abs(f(ti) - cubic_spline_value(ti, x_nodes_eq, coeffs_eq_cubic, n)) for ti in x_test)
    max_deviation_cubic_opt = max(abs(f(ti) - cubic_spline_value(ti, x_nodes_opt, coeffs_opt_cubic, n)) for ti in x_test)

    return max_deviation_linear_eq, max_deviation_linear_opt, max_deviation_quadratic_eq, max_deviation_quadratic_opt, max_deviation_cubic_eq, max_deviation_cubic_opt

def fill_spline_tables(a, b, node_counts, m):
    table_linear = []
    table_quadratic = []
    table_cubic = []

    for n in node_counts:
        deviations = calculate_max_deviation_splines(f, a, b, n, m)
        table_linear.append([n, m, deviations[0], deviations[1]])
        table_quadratic.append([n, m, deviations[2], deviations[3]])
        table_cubic.append([n, m, deviations[4], deviations[5]])

    return table_linear, table_quadratic, table_cubic






def plot_interpolations(a, b, node_counts):
    x_plot = np.linspace(a, b, 1000)
    y_plot = f(x_plot)

    # Plot for equally spaced nodes
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_plot, label='f(x)', color='black')
    for n in node_counts:
        x_nodes = equally_spaced_nodes(a, b, n)
        y_nodes = f(x_nodes)
        y_lagrange = [res_polynom_lagrange(x_nodes, y_nodes, x, n) for x in x_plot]
        plt.plot(x_plot, y_lagrange, label=f'L_{n}(x)')
    plt.title('Lagrange Interpolation with Equally Spaced Nodes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot for optimal nodes
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_plot, label='f(x)', color='black')
    for n in node_counts:
        x_nodes_opt = optimal_nodes(a, b, n)
        y_nodes_opt = f(x_nodes_opt)
        y_lagrange_opt = [res_polynom_lagrange(x_nodes_opt, y_nodes_opt, x, n) for x in x_plot]
        plt.plot(x_plot, y_lagrange_opt, label=f'Lopt_{n}(x)')
    plt.title('Lagrange Interpolation with Optimal Nodes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot for Newton interpolation with equally spaced nodes
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_plot, label='f(x)', color='black')
    for n in node_counts:
        x_nodes = equally_spaced_nodes(a, b, n)
        y_nodes = f(x_nodes)
        table = differences_newton(x_nodes, y_nodes, n)
        y_newton = [res_polynom_newton(table, x_nodes, n, x) for x in x_plot]
        plt.plot(x_plot, y_newton, label=f'N_{n}(x)')
    plt.title('Newton Interpolation with Equally Spaced Nodes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot for Newton interpolation with optimal nodes
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_plot, label='f(x)', color='black')
    for n in node_counts:
        x_nodes_opt = optimal_nodes(a, b, n)
        y_nodes_opt = f(x_nodes_opt)
        table_opt = differences_newton(x_nodes_opt, y_nodes_opt, n)
        y_newton_opt = [res_polynom_newton(table_opt, x_nodes_opt, n, x) for x in x_plot]
        plt.plot(x_plot, y_newton_opt, label=f'Nopt_{n}(x)')
    plt.title('Newton Interpolation with Optimal Nodes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_error_distribution(f, a, b, node_counts):
#     m = 1000  # number of test points
#     x_plot = np.linspace(a, b, m)
#     y_plot = f(x_plot)

#     # Plot error distribution for cubic spline and Lagrange polynomial
#     for n in node_counts:
#         # Calculate error for cubic spline
#         max_deviation_cubic_eq, max_deviation_cubic_opt = calculate_max_deviation_splines(f, a, b, n, m)[4:6]

#         # Calculate error for Lagrange polynomial
#         max_deviation_lagrange_eq, max_deviation_lagrange_opt = calculate_max_deviation(f, a, b, n, m)[:2]

#         # Plot error distribution for cubic spline
#         plt.figure(figsize=(12, 6))
# # Corrected function call
#         plt.plot(x_plot, abs(y_plot - cubic_spline_value(x_plot, equally_spaced_nodes(a, b, n), cubic_spline_coeffs(equally_spaced_nodes(a, b, n), f(equally_spaced_nodes(a, b, n)), n), n)), label='Cubic Spline (Equally Spaced)', color='blue')
#         plt.plot(x_plot, abs(y_plot - cubic_spline_value(x_plot, optimal_nodes(a, b, n), cubic_spline_coeffs(optimal_nodes(a, b, n), f(optimal_nodes(a, b, n)), n), n)), label='Cubic Spline (Optimal)', color='red')
#         plt.title(f'Absolute Error Distribution for Cubic Spline (n={n})')
#         plt.xlabel('x')
#         plt.ylabel('Absolute Error')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#         # Plot error distribution for Lagrange polynomial
#         plt.figure(figsize=(12, 6))
#         plt.plot(x_plot, abs(y_plot - [res_polynom_lagrange(equally_spaced_nodes(a, b, n), f(equally_spaced_nodes(a, b, n)), x, n) for x in x_plot]), label='Lagrange Polynomial (Equally Spaced)', color='blue')
#         plt.plot(x_plot, abs(y_plot - [res_polynom_lagrange(optimal_nodes(a, b, n), f(optimal_nodes(a, b, n)), x, n) for x in x_plot]), label='Lagrange Polynomial (Optimal)', color='red')
#         plt.title(f'Absolute Error Distribution for Lagrange Polynomial (n={n})')
#         plt.xlabel('x')
#         plt.ylabel('Absolute Error')
#         plt.legend()
#         plt.grid(True)
#         plt.show()


# Parameters
a = 1
b = 9
node_counts = [5, 10, 15, 20, 25]
m = 1000  # number of test points

# Tables
table_lagrange = []
table_newton = []

for n in node_counts:
    deviations = calculate_max_deviation(f, a, b, n, m)
    table_lagrange.append([n, m, deviations[0], deviations[1]])
    table_newton.append([n, m, deviations[2], deviations[3]])

# Print Tables
print("Lagrange Interpolation Table")
print("Nodes | Test Points | Max Deviation (Equally Spaced) | Max Deviation (Optimal)")
for row in table_lagrange:
    print(f"{row[0]:5} | {row[1]:10} | {row[2]:25.10f} | {row[3]:25.10f}")

print("\nNewton Interpolation Table")
print("Nodes | Test Points | Max Deviation (Equally Spaced) | Max Deviation (Optimal)")
for row in table_newton:
    print(f"{row[0]:5} | {row[1]:10} | {row[2]:25.10f} | {row[3]:25.10f}")

#plot_interpolations(a, b, node_counts)

table_linear, table_quadratic, table_cubic = fill_spline_tables(a, b, node_counts, m)

# Print Tables
print("Linear Spline Table")
print("Nodes | Test Points | Max Deviation (Equally Spaced) | Max Deviation (Optimal)")
for row in table_linear:
    print(f"{row[0]:5} | {row[1]:10} | {row[2]:25.10f} | {row[3]:25.10f}")

print("\nQuadratic Spline Table")
print("Nodes | Test Points | Max Deviation (Equally Spaced) | Max Deviation (Optimal)")
for row in table_quadratic:
    print(f"{row[0]:5} | {row[1]:10} | {row[2]:25.10f} | {row[3]:25.10f}")

# Print Tables
# Print Tables
# Print Tables
print("Cubic Spline Table")
print("Nodes | Test Points | Max Deviation (Equally Spaced) | Max Deviation (Optimal)")
for row in table_cubic:
    print(f"{row[0]:5} | {row[1]:10} | {row[2].item():25.10f} | {row[3].item():25.10f}")

#plot_error_distribution(f, a, b, node_counts)
