from collections import Counter

from qiskit.primitives import PrimitiveResult


def most_common_value(results: PrimitiveResult, index: int, register_name:str="c") -> str:
    return Counter(results[index].data[register_name].get_counts()).most_common(1)[0][0]