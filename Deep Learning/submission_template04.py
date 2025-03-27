def sum_numbers(a, b):
    return a + b

# Пример использования
result = sum_numbers(5, 3)
print(result)  # Вывод: 8
def is_even(number):
    return number % 2 == 0

# Пример использования
print(is_even(4))  # Вывод: True
print(is_even(5))  # Вывод: False
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Пример использования
print(factorial(5))  # Вывод: 120
def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

# Пример использования
numbers = [3, 7, 2, 8, 4]
print(find_max(numbers))  # Вывод: 8
def is_prime(num):
    import json

def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("Файл не найден!")
    except json.JSONDecodeError:
        print("Ошибка при декодировании JSON!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования
read_json('data.json')
    if num <= 1:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

def print_primes(n):
    for num in range(2, n + 1):
        if is_prime(num):
            print(num, end=" ")
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

def print_primes(n):
    for num in range(2, n + 1):
        if is_prime(num):
            print(num, end=" ")

# Пример использования
print_primes(20)  # Вывод: 2 3 5 7 11 13 17 19
# Пример использования
read_and_write_file('input.txt', 'output.txt')
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Привет! Меня зовут {self.name}, и мне {self.age} лет."

# Пример использования
person = Person("Анна", 25)
print(person.introduce())  # Вывод: Привет! Меня зовут Анна, и мне 25 лет.
def divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Ошибка: деление на ноль!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        
