from collections import Counter

from qiskit.primitives import PrimitiveResult


def most_common_value(results: PrimitiveResult, index: int) -> str:
    return Counter(results[index].data.c.get_counts()).most_common(1)[0][0]