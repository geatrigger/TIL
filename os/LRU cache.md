# LRU 알고리즘이란

* Least Recently Used Algorithm
* 사용배경
  * 캐시가 사용하는 리소스의 양이 제한되어 있다
  * 캐시는 데이터를 빠르게 저장하고 접근할 수 있어야 한다
  * cache miss를 최대한 덜 나게 하기위해 가장 최근에 사용된 적이 없는 캐시의 메모리부터 대체하여 새로운 데이터로 갱신시키게 되었다

# LRU 알고리즘 구현

* 설명

  * list 컨테이너
    * linked list
    * size : 원소 개수 반환
    * back : 마지막 원소의 reference(값)를 반환 <-> end : 마지막 원소의 iterator(주소와 비슷)
    * pop_back : 마지막 원소 제거 후 size 하나 감소
    * erase : 위치 or 범위에 따라 값을 제거
    * push_front : list의 맨 앞에 원소 삽입, size 하나 증가
    * begin : 첫번째 iterator 반환
  * unordered_map
    * end : 마지막 원소의 iterator
    * find : key 값을 갖는 원소의 iterator를 반환(못 찾으면 unordered_map::end iterator반환)
    * erase : 위치, 키, 범위에 따라 값을 제거
  * 순서
    * list<int> dq, unordered_map<int, list<int>::iterator> ma
    * 접근한 값 x에 대해 ma에 있는지 확인한다(O(1))
      * 있으면 해당 값을 dq에서 지운다(O(1))
      * 없고 dq가 이미 캐시사이즈만큼 차있다
        * 가장 덜 최근에 참조되었던 dq의 맨 마지막 원소를 지운다(마지막 원소 접근 시간 O(1))
    * x값을 dq 가장 앞에 넣고(O(1)) key를 x, value를 해당 위치(iterator)값으로 하여 ma에 넣는다.(O(1))

* 코드

  ```cpp
  #include <iostream>
  #include <list>
  #include <unordered_map>
  #include <vector>
  using namespace std;
  
  // 참조 순서 : 1 2 3 1 4 5실행 후 cache : 5 4 1 3
  class LRUCache
  {
    list<int> dq;
    unordered_map<int, list<int>::iterator> ma;
    int s;
  public:
    LRUCache(int n);
    void refer(int n);
    void display();
  };
  
  LRUCache::LRUCache(int n) {
    s = n;
  }
  
  void LRUCache::refer(int x) {
    if (ma.find(x) != ma.end()) {
      dq.erase(ma[x]);
    }
    else {
      if (dq.size() >= s) {
        int old_x = dq.back();
        dq.pop_back();
        ma.erase(old_x);
      }
    }
    dq.push_front(x);
    ma[x] = dq.begin();
  }
  
  void LRUCache::display() {
    for (int e : dq) {
      cout << e << ' ';
    }
    cout << '\n';
  }
  
  // Driver program to test above functions 
  int main()
  {
    LRUCache ca(4);
    ca.refer(1);
    ca.display();
    ca.refer(2);
    ca.display();
    ca.refer(3);
    ca.display();
    ca.refer(1);
    ca.display();
    ca.refer(4);
    ca.display();
    ca.refer(5);
    ca.display();
  
    return 0;
  }
```
  
  

# 참고자료

* https://jins-dev.tistory.com/entry/LRU-Cache-Algorithm-%EC%A0%95%EB%A6%AC
* https://j2wooooo.tistory.com/121