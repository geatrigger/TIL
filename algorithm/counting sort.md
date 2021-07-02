# counting sort란

* 계수정렬

* 데이터 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때 사용 가능

* 시간, 공간 복잡도 O(N + K) 보장(N은 데이터 개수, K는 데이터 최댓값)

* 데이터 크기 범위가 너무 클 때 공간복잡도가 너무 커진다(ex 0, 999999의 경우)

* 동일한 값을 가지는 데이터가 여러 번 등장할 때 효과적으로 사용할 수 있다

  ```python
  # 입력
  [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
  # 출력
  [0, 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9]
  ```

  

  ```python
  array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
  
  max_value = max(array)
  count_array = [0 for x in range(max_value + 1)]
  
  for val in array:
      count_array[val] += 1
  
  for i in range(0, max_value + 1):
      for j in range(count_array[i]):
          print(i, end=' ')
  ```

  ```c++
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
  
  int main()
  {
  	cin.tie(nullptr);
  	cout.tie(nullptr);
  	ios_base::sync_with_stdio(false);
  	//main
  	vector<int> arr = { 7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2 };
  	vector<int> count_arr(1000000);
  	int max_val = 0;
  	for (int val : arr) {
  		max_val = max_val > val ? max_val : val;
  		count_arr[val] += 1;
  	}
  	for (int i = 0; i <= max_val; i++) {
  		for (int j = 0; j < count_arr[i]; j++) {
  			cout << i << ' ';
  		}
  	}
  	////
  }
  ```

  ```java
  import java.util.*;
  
  public class Main {
      public static final int MAX_VALUE = 9;
  
      public static void main(String[] args) {
          int n = 15;
          // 모든 원소의 값이 0보다 크거나 같다고 가정
          int[] arr = {7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2};
          // 모든 범위를 포함하는 배열 선언(모든 값은 0으로 초기화)
          int[] cnt = new int[MAX_VALUE + 1];
  
          for (int i = 0; i < n; i++) {
              cnt[arr[i]] += 1; // 각 데이터에 해당하는 인덱스의 값 증가
          }
          for (int i = 0; i <= MAX_VALUE; i++) { // 배열에 기록된 정렬 정보 확인
              for (int j = 0; j < cnt[i]; j++) {
                  System.out.print(i + " "); // 띄어쓰기를 기준으로 등장한 횟수만큼 인덱스 출력
              }
          }
      }
  
  }
  ```

  

