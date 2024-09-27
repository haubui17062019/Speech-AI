def process(list_num, target_sum):
    list_num = sorted(list_num)
    list_result = []

    length = len(list_num)
    for i in range(length - 2):
        if i > 0 and list_num[i] == list_num[i-1]:
            continue

        left, right = i + 1, length - 1
        while left < right:
            total = list_num[i] + list_num[left] + list_num[right]
            if total == target_sum:
                list_result.append([list_num[i], list_num[left], list_num[right]])
                while left < right and list_num[left] == list_num[left+1]:
                    left += 1
                while left < right and list_num[right] == list_num[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < target_sum:
                left += 1
            else:
                right -= 1

    return list_result


if __name__ == "__main__":
    list_num = [-1, 0, 1, 2, -1, 4, 4, 3, 8, -11]
    target_sum = 0

    list_resutl = process(list_num, target_sum)
    print(list_resutl)