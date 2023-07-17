def format_large_number(number):
    units = [' ', 'k', 'M', 'G', 'T', 'P', 'E']  # Units in increasing order
    unit_index = 0

    while abs(number) >= 1000 and unit_index < len(units) - 1:
        number /= 1000
        unit_index += 1

    return f"{number:.2f}{units[unit_index]}"

def format_small_number(number):
    units = [' ', 'm', 'u', 'n', 'p', 'f', 'a']  # Units in increasing order
    unit_index = 0

    while abs(number) <= 1 and unit_index < len(units) - 1:
        number *= 1000
        unit_index += 1

    return f"{number:.2f}{units[unit_index]}"