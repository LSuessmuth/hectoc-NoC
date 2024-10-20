import random
from itertools import product
from time import time

import re

import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit_long_eval(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # Use Linux for the next code line to work!
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


example_hectocs = ["677869", "648262"]
add_sub_hectocs = ["987654", "741641"]
mult_div_hectocs = ["422125"]
parenthesis_hectocs = ["111111", "123456"]
unsolvable_hectocs = ["112117", "114123", "115567", "778551"]
OPERATIONS = ["", "-", "+", "*", "/", "^"]

HECTOC_LENGTH = 6
NUM_OPERATIONS = HECTOC_LENGTH - 1
DESIRED_OUTPUT = 100


def replace_all(input_str, to_replace_strs, replacement):

    for replace_str in to_replace_strs:
        input_str = input_str.replace(replace_str, replacement)
    return input_str


def limit_large_powers(expr):
    # Check if the number of ** is more than 4
    if expr.count("^") > 2:
        return True

    if (
        re.search(r"(\d+|\(.+\))\^(\d+|\(.+\))\^(\d+|\(.+\))", expr) is not None
        or re.search(r"(\d+|\(.+\))\^\(.*\^.*\)", expr) is not None
    ):
        return True

    # Check if any base or exponent in ** is too large
    matches = re.split(r"\s*\^\s*", expr)
    if (
        len(matches) > 2
        and sum(
            [
                "X" not in replace_all(match, OPERATIONS[1:], "X")
                for match in matches[1:-1]
            ]
        )
        > 0
    ):
        return True

    for base_str, exp_str in zip(matches[:-1], matches[1:]):
        base_str = replace_all(
            replace_all(base_str, [")", "("], ""), OPERATIONS[1:] + ["(", ")"], " "
        ).split(" ")[-1]
        exp_str = replace_all(
            replace_all(exp_str, [")", "("], ""), OPERATIONS[1:], " "
        ).split(" ")[0]

        base, exponent = int(eval(base_str)), int(eval(exp_str))

        if base > 20 or exponent > 10:
            return True

    return False


def find_index_closest_to_100(num_str: str) -> int:
    """
    Finds the starting index of two-digit number closest to 100
    """
    return min(
        range(len(num_str) - 1),
        key=lambda idx: abs(100 - int(num_str[idx]) * 10 + int(num_str[idx])),
    )


def eval_result(num_str: str) -> int | None:
    num_str = num_str.replace("^", "**")
    try:
        # computed_result = numexpr.evaluate(num_str)
        computed_result = eval(num_str)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None

    if (
        type(computed_result) != complex
        and abs(computed_result) != float("inf")
        and computed_result == int(computed_result)
    ):
        return computed_result
    return None


def get_random_hectoc() -> str:
    """
    Generates a random 6-digit hectoc
    """
    random_input = "".join(
        [f"{random.randint(a=1, b=9)}" for _ in range(HECTOC_LENGTH)]
    )
    return random_input


def combine_input_with_ops(num_str, operations: list[str]):
    result = num_str[0]
    for digit, oper in zip(num_str[1:], operations):
        result += oper + digit
    return result


def get_possible_par_start_positions(expr: str):
    nums = "123456789"
    return [
        x
        for x in range(len(expr))
        if x == 0 or (expr[x] in nums and expr[x - 1] not in nums)
    ]


def get_possible_par_end_positions(expr: str):
    nums = "123456789"
    return [
        x
        for x in range(len(expr))
        if x == len(expr) - 1 or (expr[x] in nums and expr[x + 1] not in nums)
    ]


def put_to_pos(expr, char, idx):
    return expr[:idx] + char + expr[idx:]


def apply_parenthesis(expr, par_list: list[tuple[int]]) -> str:
    positions_to_substrings = {idx: val for idx, val in enumerate(expr)}
    for left_par_idx, right_par_idx in par_list:
        positions_to_substrings[left_par_idx] = (
            "(" + positions_to_substrings[left_par_idx]
        )
        positions_to_substrings[right_par_idx] += ")"

    return "".join(
        [positions_to_substrings[key] for key in sorted(positions_to_substrings)]
    )


def get_parenthesis_positions(expr, max_pars=3):
    left_pars = get_possible_par_start_positions(expr)
    right_pars = get_possible_par_end_positions(expr)

    all_pars = [
        (x, y) for x, y in product(left_pars, right_pars, repeat=1) if y > x + 1
    ]

    combined_pars = product(all_pars, repeat=max_pars)
    all_combinations = list(set([tuple(sorted(set(x))) for x in combined_pars]))

    return all_combinations


def print_text(i, show_text=False):
    if i % 50000 != 0:
        return
    if show_text and i == 50000:
        print("MOMENT NOCH!\n\n")
    elif show_text and i == 100000:
        print("GLEICH FERTIG!\n")
    else:
        print(f"Iteration {i:_}".replace("_", " "))


def filter_singular_pars(expr):
    # print(expr)
    expr = re.sub(r"\((\d+)\)", r"\1", expr)
    while True:

        orig_str = expr
        expr = re.sub(r"(\+|\(|^)\(([^\(\)]*)\)(\+|\)|$)", r"\1\2\3", expr)
        expr = re.sub(r"(\+|-|\(|^)\(([^\(\+\)-]*)\)(\+|-|\)|$)", r"\1\2\3", expr)
        if expr == orig_str:
            return expr
    # return re.sub(r"^\((.*)\)$", r"\1", expr)


def filter_solutions(sol_list):
    sol_list = [filter_singular_pars(sol) for sol in sol_list]
    return list(set(sol_list))


def brute_force_without_pars(num_str: str, show_text=False, max_pars=3) -> None:
    all_sols = []
    i = 0
    for operation_perm in product(OPERATIONS, repeat=NUM_OPERATIONS):
        base_str_wout_pars = combine_input_with_ops(num_str, operation_perm)

        strs_with_parenthesis = [
            apply_parenthesis(base_str_wout_pars, par)
            for par in get_parenthesis_positions(base_str_wout_pars, max_pars=max_pars)
        ]
        filtered_strs_with_pars = filter_solutions(strs_with_parenthesis)
        for str_with_ops in filtered_strs_with_pars:
            i += 1
            print_text(i, show_text)

            if limit_large_powers(str_with_ops):
                continue

            try:
                with time_limit_long_eval(1):
                    computed_result = eval_result(str_with_ops)
            except TimeoutException:
                computed_result = None
                print(str_with_ops, "timeout")
            if computed_result == DESIRED_OUTPUT:
                print("Found solution:", str_with_ops)
                all_sols.append(str_with_ops)
    print("ALL POSSIBLE SOLUTIONS FOUND! 100% LEGIT.\n\n")
    return all_sols


if __name__ == "__main__":
    # input_data = unsolvable_hectocs[0]
    input_data = "648282"

    print(input_data)
    start_time = time()
    all_sols = brute_force_without_pars(input_data, True, max_pars=2)
    end_time = time() - start_time
    unique_sols = filter_solutions(all_sols)

    print("---SOLUTIONS---")
    print(*unique_sols, sep="\n")
    print(len(unique_sols), "unique solutions found")
    print(f"Took {end_time:.2f} seconds.\n\nTHAT WAS QUICK!")
