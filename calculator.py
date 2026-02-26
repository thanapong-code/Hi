"""Simple calculator supporting basic arithmetic operations."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def calculate(expression):
    """Evaluate a simple arithmetic expression string.

    Supports +, -, *, / operators with integer or float operands.
    Example: calculate("3 + 4") -> 7.0
    """
    parts = expression.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Invalid expression: '{expression}'")

    a, op, b = parts
    try:
        a, b = float(a), float(b)
    except ValueError:
        raise ValueError(f"Invalid operands: '{a}', '{b}'")

    ops = {
        "+": add,
        "-": subtract,
        "*": multiply,
        "/": divide,
    }
    if op not in ops:
        raise ValueError(f"Unsupported operator: '{op}'")

    return ops[op](a, b)


def main():
    print("Simple Calculator")
    print("Supported operators: +  -  *  /")
    print("Enter 'quit' to exit.\n")

    while True:
        try:
            expression = input("Expression: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if expression.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = calculate(expression)
            # Display as int when result is a whole number
            print(f"= {int(result) if result == int(result) else result}\n")
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
