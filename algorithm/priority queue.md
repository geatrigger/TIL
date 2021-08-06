# priority queue

* 우선순위에 따라 처리할 때 사용
* 구현 방법
  * 리스트
    * 삽입 : O(1), 삭제 : O(N)
  * 힙
    * 삽입 : O(log N), 삭제 : O(log N)

# heap

* 완전 이진 트리 자료구조
  * 루트부터 시작해서 왼쪽에서 오른쪽으로 차례대로 삽입됨
* 항상 루트노드를 제거한다(최소 힙일 경우 가장 작은 값, 최대 힙일 경우 가장 큰 값)
* 원소가 추가될 때 가장 마지막에 오고 마지막 노드에서 상향식 heapify진행
* 원소가 제거될 때 루트노드가 제거되고 가장 마지막 노드가 루트로 온다음 루트노드부터 하향식 heapify진행

# heapify

* 힙을 구성하기 위한 함수

# 힙정렬

* 데이터를 힙에 넣었다가 모두 꺼내면 정렬이 됨

* 시간 복잡도 O(N log N)

* python

  ```python
  import sys
  import heapq
  input = sys.stdin.readline
  
  def heapsort(iterable):
      h = []
      result = []
      # 모든 원소를 차례대로 힙에 삽입
      for value in iterable:
          heapq.heappush(h, value)
      # 힙에 삽입된 모든 원소를 차례대로 꺼내어 담기
      for i in range(len(h)):
          result.append(heapq.heappop(h))
      return result
  
  n = int(input())
  arr = []
  
  for i in range(n):
      arr.append(int(input()))
  
  res = heapsort(arr)
  
  for i in range(n):
      print(res[i])
  ```

* cpp

  ```cpp
  #include <bits/stdc++.h>
  
  using namespace std;
  
  void heapSort(vector<int>& arr) {
      priority_queue<int> h;
      // 모든 원소를 차례대로 힙에 삽입
      for (int i = 0; i < arr.size(); i++) {
          h.push(-arr[i]);
      }
      // 힙에 삽입된 모든 원소를 차례대로 꺼내어 출력
      while (!h.empty()) {
          printf("%d\n", -h.top());
          h.pop();
      }
  }
  
  int n;
  vector<int> arr;
  
  int main() {
      cin >> n;
      for (int i = 0; i < n; i++) {
          int x;
          scanf("%d", &x);
          arr.push_back(x);
      }
      heapSort(arr);
  }
  ```

  