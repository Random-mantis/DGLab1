import fractions as fr
import math

import numpy as np


def identity_matrix(n):
    matrix = np.array(
        [[fr.Fraction(0) for _ in range(n)] for _ in range(n)],
        dtype=object,
    )
    for i in range(n):
        matrix[i][i] = fr.Fraction(1)
    return matrix


def zero_vector(n):
    return np.array([fr.Fraction(0) for _ in range(n)], dtype=object)


def as_fraction_array(matrix):
    matrix = np.asarray(matrix, dtype=object)
    if matrix.size > 0 and all(
        isinstance(value, fr.Fraction) for value in matrix.flat
    ):
        return matrix.copy()
    result = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        if isinstance(value, fr.Fraction):
            result[index] = value
        else:
            result[index] = fr.Fraction(value)
    return result


def Gauss_determinant(matrix):
    determinant = fr.Fraction(1)
    for i in range(len(matrix)):
        determinant *= matrix[i][i]
    return determinant


def LU_decomposition(matrix):
    n = len(matrix)
    U = as_fraction_array(matrix)
    L = identity_matrix(n)
    P = identity_matrix(n)
    swap_count = 0
    for i in range(n):
        max_pivot = max(range(i, n), key=lambda r: abs(U[r][i]))
        if U[max_pivot][i] == 0:
            raise ZeroDivisionError()
        if max_pivot != i:
            U[[i, max_pivot]] = U[[max_pivot, i]].copy()
            P[[i, max_pivot]] = P[[max_pivot, i]].copy()
            if i > 0:
                L[[i, max_pivot], :i] = L[[max_pivot, i], :i].copy()
            swap_count += 1
        for j in range(i + 1, n):
            factor = fr.Fraction(U[j][i], U[i][i])
            L[j][i] = factor
            U[j, i:] = U[j, i:] - factor * U[i, i:]
    return L, U, P, swap_count


def LU_solve(L, U, P, roots):
    y = (P @ as_fraction_array(roots)).copy()
    n = len(L)
    for i in range(n):
        for j in range(i):
            y[i] -= L[i][j] * y[j]
    x = y.copy()
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


def LU_inverse(L, U, P):
    n = len(L)
    inverse = np.empty((n, n), dtype=object)
    for i in range(n):
        e = zero_vector(n)
        e[i] = fr.Fraction(1)
        inverse[:, i] = LU_solve(L, U, P, e)
    return inverse


def LU_determinant(U, swap_count):
    determinant = Gauss_determinant(U)
    if swap_count % 2 == 1:
        determinant = -determinant
    return determinant


def Gauss_elimination(matrix, roots):
    L, U, P, swap_count = LU_decomposition(matrix)
    determinant = LU_determinant(U, swap_count)
    return determinant, LU_solve(L, U, P, roots)


def Thomas_method(lower, main, upper, right):
    lower = as_fraction_array(lower)
    main = as_fraction_array(main)
    upper = as_fraction_array(upper)
    right = as_fraction_array(right)
    n = len(main)
    a_curr = zero_vector(n)
    b_curr = zero_vector(n)
    a_curr[0] = -upper[0] / main[0]
    b_curr[0] = right[0] / main[0]
    for i in range(1, n):
        denom = main[i] + lower[i - 1] * a_curr[i - 1]
        if i < n - 1:
            a_curr[i] = -upper[i] / denom
        else:
            a_curr[i] = fr.Fraction(0)
        b_curr[i] = (right[i] - lower[i - 1] * b_curr[i - 1]) / denom
    result = zero_vector(n)
    result[n - 1] = b_curr[n - 1]
    for i in range(n - 2, -1, -1):
        result[i] = a_curr[i] * result[i + 1] + b_curr[i]
    return result, a_curr, b_curr


def Yakobi_Iteration(matrix, prev, right):
    result = as_fraction_array(right)
    n = len(right)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result[i] -= matrix[i][j] * prev[j]
        if matrix[i][i] == 0:
            raise ZeroDivisionError("Yakobi method can't used.")
        result[i] /= matrix[i][i]
    return result


def Yakobi_Norm(left, right, eps):
    for i in range(len(left)):
        if abs(left[i] - right[i]) > eps:
            return False
    return True


def Yakobi(matrix, right, eps):
    current = zero_vector(len(right))
    next = Yakobi_Iteration(matrix, current, right)
    iterations = 1
    while not Yakobi_Norm(next, current, eps):
        current = next
        next = Yakobi_Iteration(matrix, current, right)
        iterations += 1
        if iterations > 100000:
            break
    return next, iterations


def Zeidel_Iteration(matrix, right, prev):
    result = as_fraction_array(right)
    n = len(right)
    for i in range(n):
        for j in range(i):
            result[i] -= matrix[i][j] * result[j]
        for k in range(i + 1, n):
            result[i] -= matrix[i][k] * prev[k]
        if matrix[i][i] == 0:
            raise ZeroDivisionError("Zeidel method can't be used")
        result[i] /= matrix[i][i]
    return result


def Zeidel(matrix, right, eps):
    current = zero_vector(len(right))
    next = Zeidel_Iteration(matrix, right, current)
    iterations = 1
    while not Yakobi_Norm(next, current, eps):
        current = next
        next = Zeidel_Iteration(matrix, right, current)
        iterations += 1
        if iterations > 100000:
            break
    return next, iterations


def Rotate_Method(matrix, eps):
    n = len(matrix)
    matrix = np.array(matrix, dtype=float).copy()
    v = np.eye(n, dtype=float)
    maxx = 1
    while maxx > eps:
        p = q = 0
        t = np.eye(n, dtype=float)
        maxx = 0
        angle = 0.0
        for i in range(n):
            for j in range(n):
                if i != j and abs(matrix[i][j]) > maxx:
                    maxx = abs(matrix[i][j])
                    p = i
                    q = j
        if maxx <= eps:
            break
        if matrix[p][p] == matrix[q][q]:
            angle = math.pi / 4
        else:
            angle = math.atan(
                2 * matrix[p][q] / (matrix[q][q] - matrix[p][p])
            ) / 2
        t[p][p] = t[q][q] = math.cos(angle)
        t[q][p] = -math.sin(angle)
        t[p][q] = math.sin(angle)
        matrix = t.T @ matrix @ t
        v = v @ t
    return [matrix[i][i] for i in range(n)], v


def gram_schmidt(A):
    A = np.array(A, dtype=float)
    n = len(A)
    Q = np.zeros((n, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    for i in range(n):
        v = A[:, i].copy()
        for j in range(i):
            R[j][i] = Q[:, j] @ A[:, i]
            v = v - R[j][i] * Q[:, j]
        R[i][i] = math.sqrt(v @ v)
        Q[:, i] = v / R[i][i]
    return Q, R


def QR_norm(matrix, eps):
    n = len(matrix)
    for i in range(2, n):
        for j in range(i - 1):
            if abs(matrix[i][j]) > eps:
                return False
    i = 0
    while i < n:
        if i == n - 1 or abs(matrix[i + 1][i]) <= eps:
            i += 1
        else:
            if i + 2 < n and abs(matrix[i + 2][i + 1]) > eps:
                return False
            i += 2
    return True


def QR_eigenvalues_from_blocks(matrix, eps):
    n = len(matrix)
    values = []
    i = 0
    while i < n:
        if i == n - 1 or abs(matrix[i + 1][i]) <= eps:
            values.append(complex(matrix[i][i]))
            i += 1
        else:
            a = matrix[i][i]
            b = matrix[i][i + 1]
            c = matrix[i + 1][i]
            d = matrix[i + 1][i + 1]
            trace = a + d
            det = a * d - b * c
            disc = trace * trace - 4 * det
            if disc >= 0:
                root = math.sqrt(disc)
                values.append(complex((trace + root) / 2))
                values.append(complex((trace - root) / 2))
            else:
                real = trace / 2
                imag = math.sqrt(-disc) / 2
                values.append(complex(real, imag))
                values.append(complex(real, -imag))
            i += 2
    return values


def QR_method(matrix, eps):
    a_curr = np.array(matrix, dtype=float).copy()
    for _ in range(100000):
        q, r = gram_schmidt(a_curr)
        a_curr = r @ q
        if QR_norm(a_curr, eps):
            break
    return QR_eigenvalues_from_blocks(a_curr, eps)
