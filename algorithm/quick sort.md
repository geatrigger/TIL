# quick sort란

* 기준 데이터(pivot)를 설정하고, 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
* 기준 데이터를 정확히 정렬된 위치에 놓아가면서 재귀적으로 정렬
* 이상적인 경우 분할이 절반씩 일어난다면 젠체 연산 횟수는 O(N log N)
  * 피벗을 기준으로 위치를 바꾸는데 약 N번
  * 분할이 절반씩 일어나면 1이 될 때까지 분할 횟수 log N
* 최악의 경우 O(N^2)의 시간 복잡도
  * 첫번째 원소를 피벗으로 삼는 퀵정렬의 경우, 이미 정렬된 배열에 대해서는 분할이 N번 일어나 O(N^2)
* 공간복잡도 O(1)

```python
# 1 3 4 2 5, 2 3 1 4 5
```

```python
def quick_sort(arr):
    def sort(low, high):
        if high <= low:
            return
        mid = partition(low, high)
        sort(low, mid - 1)
        sort(mid, high)

    def partition(low, high):
        pivot = arr[(low + high) // 2]
        while low <= high:
            while arr[low] < pivot:
                low += 1
            while arr[high] > pivot:
                high -= 1
            if low <= high:
                arr[low], arr[high] = arr[high], arr[low]
                low, high = low + 1, high - 1
        return low

    return sort(0, len(arr) - 1)
```

