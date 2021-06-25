# Pointer

* 어떤 변수의 주소값을 갖는 변수
* 증가, 감소 연산 사용 가능
* delete를 이용해 pointer를 지울 수 있다

# Iterator

* array, container 등에서 어떤 element를 가리키고 있다
* 범위 내의 원소들에 대해서만 iterate(순회)할 수 있다
* 실제 위치가 따로따로 떨어져 있어도 iterate가능하다
* 경우에 따라 증가, 감소하는 연산을 할 수 없다.
* delete로 iterator를 지울 수 없다(container가 메모리 관리에 책임이 있다)

# 참고자료

* https://www.geeksforgeeks.org/difference-between-iterators-and-pointers-in-c-c-with-examples/