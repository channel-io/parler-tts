import psutil
import os

# 현재 프로세스 메모리 사용량을 반환하는 함수
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # 메모리 사용량(MB)

def get_iter():
    for i in range(1_000_000):
        yield i

# 메모리 측정 전
data_before = get_memory_usage()
print(f"메모리 사용량 (리스트 선언 전): {data_before:.2f} MB")

# 100만개의 아이템이 있는 리스트 선언
large_list = list(range(1_000_000))

# 메모리 측정 후
data_after = get_memory_usage()
print(f"메모리 사용량 (리스트 선언 후): {data_after:.2f} MB")
print(f"리스트가 사용한 메모리: {data_after - data_before:.2f} MB")

data_before = get_memory_usage()
print(f"메모리 사용량 (리스트 선언 전): {data_before:.2f} MB")

large2 = iter(large_list)
del large_list

data_after = get_memory_usage()
print(f"메모리 사용량 (리스트 선언 후): {data_after:.2f} MB")
print(f"리스트가 사용한 메모리: {data_after - data_before:.2f} MB")
print(next(large2))
print(next(large2))
