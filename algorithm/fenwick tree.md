# fenwick tree(binary index tree)

* 2진법 인덱스 구조를 활용해 구간 합 문제 효과적 해결 가능
* 어떤 인덱스에서 0이 아닌 마지막 비트값 = 저장하고 있는 값들의 개수인 트리를 생성
  * 정수 K에서 0이 아닌 마지막 비트를 찾는 법 : K & -K
  * 예시)4->0100, 4개 / 6->0110, 2개
* 특정 값을 변경할 때
  * 0이 아닌 마지막 비트만큼 더하면서 구간들의 값을 변경
  * 예시) 3번째 값을 변경하면 3, 4, 8, 16번째 값을 변경해야 한다
* 구간 합 구하기
  * 1부터 N, 1부터 M까지 누적합을 구하고 서로 빼서 구한다
  * 0이 아닌 마지막 비트만큼 빼면서 구간들의 값의 합을 계산
  * 예시)1~11번째까지의 합은 11, 10, 8번째 값의 합이다

# 구간 합 구하기 문제

* 어떤 N개의 수가 있고, 중간의 수의 변경이 계속 일어날 때, 어떤 구간의 합을 구하는 문제
* 백준 2042번

```python
# 입력
# N, M, K : 수의 개수, 수의 변경이 일어나는 횟수, 구간의 합을 구하는 횟수
# N개의 수
# a, b, c : a가 1일 경우 b번째 수를 c로 바꿈, a가 2일 경우 b~c번째 구간의 합 출력
5 2 2
1
2
3
4
5
1 3 6
2 2 5
1 5 2
2 3 5
# 출력
17
12
```



```python
import sys

def get_last_bit(val):
    return val & -val

def update(i, val):
    global values
    global fenwick
    global N
    diff = val - values[i]
    values[i] = val
    new_i = i
    while new_i <= N:
        fenwick[new_i] += diff
        new_i += get_last_bit(new_i)
    return

def cumulative_sum(i):
    global fenwick
    new_i = i
    result = 0
    while new_i > 0:
        result += fenwick[new_i]
        new_i -= get_last_bit(new_i)
    return result

def interval_sum(i, j):
    return cumulative_sum(j) - cumulative_sum(i - 1)

input = sys.stdin.readline
N, M, K = map(int, input().split())

values = [0 for x in range(N+1)]
fenwick = [0 for x in range(N+1)]

for i in range(1, N + 1):
    a = int(input())
    update(i, a)

for i in range(M + K):
    a, b, c = map(int, input().split())
    if a == 1:
        update(b, c)
    elif a == 2:
        total = interval_sum(b, c)
        print(total)
```

```cpp
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <tuple>
#include <cmath>
#include <set>
#include <deque>
#include <queue>
#include <unordered_map>
#include <regex>

using namespace std;

vector<long long> values(1000001);
vector<long long> fenwick(1000001);
int N, M, K;

long long get_last_bit(long long val) {
	return val & -val;
}

void update(int i, long long val) {
	long long diff;
	diff = val - values[i];
	values[i] = val;
	int new_i = i;
	while (new_i <= N) {
		fenwick[new_i] += diff;
		new_i += get_last_bit(new_i);
	}
	return;
}

long long cumulative_sum(int i) {
	int new_i = i;
	long long result = 0;
	while (new_i > 0) {
		result += fenwick[new_i];
		new_i -= get_last_bit(new_i);
	}
	return result;
}

long long interval_sum(int i, int j) {
	return cumulative_sum(j) - cumulative_sum(i - 1);
}

int main()
{
	cin.tie(nullptr);
	cout.tie(nullptr);
	ios_base::sync_with_stdio(false);
    
	cin >> N >> M >> K;
	for (int i = 1; i <= N; i++) {
		long long val;
		cin >> val;
		update(i, val);
	}
	for (int i = 0; i < M + K; i++) {
		int a;
		long long b, c;
		cin >> a >> b >> c;
		if (a == 1) {
			update((int)b, c);
		}
		else if (a == 2) {
			long long result = interval_sum((int)b, (int)c);
			cout << result << '\n';
		}
	}
}
```

