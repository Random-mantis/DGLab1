import fractions as fr
import os
import sys

import numpy as np

import Gauss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(
    r"c:\Users\Kostya\Desktop\Important\Programs\VUS" +
    r"\Third Course\DigitalMethods",
    "data",
)
REPORT_PATH = os.path.join(DATA_DIR, "check_report.txt")
DEFAULT_EPS = fr.Fraction(1, 1000)
_report = None


class ReportWriter:
    def __init__(self, path):
        self.path = path
        self._file = None
        self._stdout = sys.stdout
        self.encoding = "utf-8"

    def __enter__(self):
        self._file = open(self.path, "w", encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._stdout
        if self._file is not None:
            self._file.close()
            self._file = None
        return False

    def write(self, text):
        if self._file is not None:
            self._file.write(text)

    def flush(self):
        if self._file is not None:
            self._file.flush()

    def isatty(self):
        return False

    def console(self, text=""):
        line = text if text.endswith("\n") else text + "\n"
        self._stdout.write(line)
        self._stdout.flush()
        self.write(line)


def console(text=""):
    if _report is not None:
        _report.console(text)
    else:
        print(text)


def to_fraction(value):
    text = value.decode() if isinstance(value,
                                        (bytes, bytearray)) else str(value)
    return fr.Fraction(text.strip())


def matrices_from_file(filename):
    matrices = []
    rows = []
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if rows:
                    matrices.append(np.array(rows, dtype=object))
                    rows = []
                continue
            rows.append([fr.Fraction(part) for part in line.split()])
    if rows:
        matrices.append(np.array(rows, dtype=object))
    return matrices


def Matrix_from_file(filename):
    return matrices_from_file(filename)[0]


def read_eps_list(filename):
    epsilons = []
    try:
        with open(filename) as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                epsilons.append(fr.Fraction(line.split()[0]))
    except FileNotFoundError:
        pass
    return epsilons


def read_eps(epsilons, index=0):
    if index < len(epsilons):
        return epsilons[index]
    return DEFAULT_EPS


def format_number(value):
    if isinstance(value, complex):
        real = value.real
        imag = value.imag
        if abs(imag) < 1e-12:
            return f"{real:10.4g}"
        return f"{real:8.4g}{imag:+8.4g}j"
    if isinstance(value, fr.Fraction):
        value = float(value)
    if abs(value) < 1e-12:
        return f"{0:10.4g}"
    return f"{float(value):10.4g}"


def print_method_header(title):
    line = "=" * 64
    print()
    print(line)
    print(f"  {title}")
    print(line)


def print_system_header(index, total, eps=None):
    print()
    print("-" * 64)
    title = f"  Система {index}/{total}"
    if eps is not None:
        title += f"   |   eps = {format_number(eps).strip()}"
    print(title)
    print("-" * 64)


def print_block(title):
    print()
    print(f"[{title}]")


def print_matrix(title, matrix):
    print(f"  {title}:")
    matrix = np.array(matrix)
    if matrix.ndim == 0:
        print(f"    {format_number(matrix).strip()}")
        return
    if matrix.ndim == 1:
        print("   ", "  ".join(format_number(value) for value in matrix))
        return
    for row in matrix:
        print("   ", "  ".join(format_number(value) for value in row))


def print_vector(title, vector, names=None):
    vector = np.array(vector).reshape(-1)
    print(f"  {title}:")
    for i, value in enumerate(vector):
        if names is None:
            name = f"x{i + 1}"
        else:
            name = names[i]
        print(f"    {name} = {format_number(value).strip()}")


def print_scalar(title, value):
    print(f"  {title}: {format_number(value).strip()}")


def progress_system(method, index, total, note=""):
    text = f"  {method}: система {index}/{total}"
    if note:
        text += f"  |  {note}"
    console(text)


def run_lu(filename):
    print_method_header("1.1  LU-разложение")
    console("1.1  LU-разложение")
    matrices = matrices_from_file(filename)
    for index, augmented in enumerate(matrices):
        print_system_header(index + 1, len(matrices))
        A = augmented[:, :-1]
        b = augmented[:, -1]
        L, U, P, swap_count = Gauss.LU_decomposition(A)
        x = Gauss.LU_solve(L, U, P, b)
        inverse = Gauss.LU_inverse(L, U, P)
        determinant = Gauss.LU_determinant(U, swap_count)

        print_block("Исходные данные")
        print_matrix("A", A)
        print_vector("b", b, names=[f"b{i + 1}" for i in range(len(b))])

        print_block("Разложение")
        print_matrix("L", L)
        print_matrix("U", U)
        print_matrix("P", P)
        print_matrix("L * U", L @ U)
        print_matrix("P * A", P @ A)

        print_block("Результаты")
        print_vector("Решение Ax = b", x)
        print_scalar("det(A)", determinant)
        print_matrix("A^(-1)", inverse)

        print_block("Проверки")
        print_matrix("A * A^(-1)", A @ inverse)
        print_matrix("A * x", A @ x)
        print_vector("b (для сравнения)", b,
                     names=[f"b{i + 1}" for i in range(len(b))])
        xs = ", ".join(format_number(v).strip() for v in x)
        progress_system("LU", index + 1, len(matrices),
                        f"x = [{xs}]" +
                        f"det = {format_number(determinant).strip()}")


def run_thomas(filename):
    print_method_header("1.2  Метод прогонки (Томаса)")
    console("1.2  Метод прогонки (Томаса)")
    matrices = matrices_from_file(filename)
    for index, data in enumerate(matrices):
        print_system_header(index + 1, len(matrices))
        lower = data[1:, 0]
        main = data[:, 1]
        upper = data[:-1, 2]
        right = data[:, 3]
        result, a_curr, b_curr = Gauss.Thomas_method(lower, main, upper, right)

        print_block("Диагонали и правая часть")
        print_matrix("Нижняя (a)", lower)
        print_matrix("Главная (b)", main)
        print_matrix("Верхняя (c)", upper)
        print_matrix("Правая часть (d)", right)

        print_block("Прогоночные коэффициенты")
        print_matrix("alpha", a_curr)
        print_matrix("beta", b_curr)

        print_block("Результат")
        print_vector("Решение", result)
        xs = ", ".join(format_number(v).strip() for v in result)
        progress_system("Прогонка", index + 1, len(matrices), f"x = [{xs}]")


def run_iter(filename, eps_filename):
    print_method_header("1.3  Метод простых итераций и метод Зейделя")
    console("1.3  Метод простых итераций и метод Зейделя")
    matrices = matrices_from_file(filename)
    epsilons = read_eps_list(eps_filename)
    for index, body in enumerate(matrices):
        eps = read_eps(epsilons, index)
        print_system_header(index + 1, len(matrices), eps)
        matrix = body[:-1]
        right = body[-1]
        _, exact = Gauss.Gauss_elimination(matrix, right)
        yakobi_x, yakobi_iter = Gauss.Yakobi(matrix, right, eps)
        zeidel_x, zeidel_iter = Gauss.Zeidel(matrix, right, eps)

        print_block("Исходные данные")
        print_matrix("A", matrix)
        print_vector("b", right,
                     names=[f"b{i + 1}" for i in range(len(right))])

        print_block("Эталон (исключение Гаусса)")
        print_vector("x*", exact)

        print_block("Метод Якоби")
        print_scalar("Итераций", yakobi_iter)
        print_vector("x", yakobi_x)

        print_block("Метод Зейделя")
        print_scalar("Итераций", zeidel_iter)
        print_vector("x", zeidel_x)

        print_block("Сравнение скорости")
        if yakobi_iter < zeidel_iter:
            faster = "Якоби"
            print("  Быстрее: метод Якоби")
        elif zeidel_iter < yakobi_iter:
            faster = "Зейдель"
            print("  Быстрее: метод Зейделя")
        else:
            faster = "одинаково"
            print("  Одинаковое число итераций")
        progress_system(
            "Итерации", index + 1, len(matrices),
            f"Якоби {yakobi_iter} итер.," +
            f"Зейдель {zeidel_iter} итер., быстрее: {faster}",
        )


def run_rotate(filename, eps_filename):
    print_method_header("1.4  Метод вращений Якоби")
    console("1.4  Метод вращений Якоби")
    matrices = matrices_from_file(filename)
    epsilons = read_eps_list(eps_filename)
    for index, matrix in enumerate(matrices):
        eps = read_eps(epsilons, index)
        print_system_header(index + 1, len(matrices), eps)
        values, vectors = Gauss.Rotate_Method(matrix, float(eps))
        Lambda = np.diag(values)
        A = np.array(matrix, dtype=float)

        print_block("Исходные данные")
        print_matrix("A", matrix)

        print_block("Собственные значения")
        print_vector("lambda", values,
                     names=[f"λ{i + 1}" for i in range(len(values))])

        print_block("Собственные векторы")
        print_matrix("V (столбцы — векторы)", vectors)
        for i in range(len(values)):
            print_vector(f"h{i + 1}", vectors[:, i],
                         names=[f"h{i + 1},{j + 1}"
                                for j in range(len(values))])

        print_block("Проверка A*V ≈ V*Λ")
        print_matrix("A * V", A @ vectors)
        print_matrix("V * Λ", vectors @ Lambda)
        lams = ", ".join(format_number(v).strip() for v in values)
        progress_system("Вращения", index + 1, len(matrices),
                        f"λ = [{lams}]")


def run_qr(filename, eps_filename):
    print_method_header("1.5  QR-разложение и QR-алгоритм")
    console("1.5  QR-разложение и QR-алгоритм")
    matrices = matrices_from_file(filename)
    epsilons = read_eps_list(eps_filename)
    for index, matrix in enumerate(matrices):
        eps = read_eps(epsilons, index)
        print_system_header(index + 1, len(matrices), eps)
        Q, R = Gauss.gram_schmidt(matrix)
        values = Gauss.QR_method(matrix, float(eps))

        print_block("Исходные данные")
        print_matrix("A", matrix)

        print_block("QR-разложение")
        print_matrix("Q", Q)
        print_matrix("R", R)
        print_matrix("Q * R", Q @ R)

        print_block("Собственные значения (QR-алгоритм)")
        print_vector("lambda", values,
                     names=[f"λ{i + 1}" for i in range(len(values))])
        lams = ", ".join(format_number(v).strip() for v in values)
        progress_system("QR", index + 1, len(matrices), f"λ = [{lams}]")


def main():
    global _report
    with ReportWriter(REPORT_PATH) as _report:
        run_lu(
            os.path.join(DATA_DIR, "lu.txt")
        )
        run_thomas(os.path.join(DATA_DIR, "thomas.txt"))
        run_iter(
            os.path.join(DATA_DIR, "iter.txt"),
            os.path.join(DATA_DIR, "iter_eps.txt"),
        )
        run_rotate(
            os.path.join(DATA_DIR, "rotate.txt"),
            os.path.join(DATA_DIR, "rotate_eps.txt"),
        )
        run_qr(
            os.path.join(DATA_DIR, "qr.txt"),
            os.path.join(DATA_DIR, "qr_eps.txt"),
        )


if __name__ == "__main__":
    main()
