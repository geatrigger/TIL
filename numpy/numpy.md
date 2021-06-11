# 210602

* numpy특징

  * Numerical Python
  * 파이썬에서 산술 계산을 위한 가장 필수 패키지(과학 계산을 위한 대부분의 패키지는 numpy 배열 객체를 사용)
  * ndarray(n dimensional array)는 빠른 배열 계산과 유연한 브로드캐스팅 기능 제공
  * 반복문 작성 필요 없이 전체 데이터 배열을 빠르게 계산
  * 배열 데이터를 디스크에 쓰거나 읽고, 메모리에 적재된 파일을 다루는 도구
  * 선형대수, 난수 생성기 등에 사용
  * C, C++, 포트란으로 작성한 코드를 연결할 수 있는 C API가 제공

* 데이터 생성

  * 모두 데이터 타임은 numpy.ndarray
  * np.zeros : 실수 0으로 채워진 array
  * np.ones : 실수 1로 채워진 array
  * np.array : 입력한 list를 array로 변환
  * np.eye : diagonal의 시작 위치를 정하여 1을 넣을 수 있음
  * np.full : 지정한 크기의 배열을 생성하고 채울 값을 지정
  * np.empty : random값으로 array생성(이전에 같은 크기로 np.zeros, np.ones, np.eye, np.ones, np.identity를 호출했으면 마지막에 생성된 값이 나옴)
  * np.empty_like : 똑같은 크기의 배열 생성
  * np.linspace : start부터 stop까지 동등한 간격으로 num개수만큼 생성
  * np.logspace :  start부터 stop까지 동등한 간격으로 num개수만큼 생성(log scale)

  ```python
  import numpy as np
  
  a = np.zeros(3) # array([0., 0., 0.])
  a = np.ones(3) # array([1., 1., 1.])
  c = np.array([[[1, 2], [3, 4], [5, 6]]]) # array([[[1, 2], [3, 4], [5, 6]]])
  a = np.zeros((2, 3)) # array([[0., 0., 0.], [0., 0., 0.]])
  c = np.eye(3, 4, k=1) # array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
  a = np.identity(3) # array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
  a = np.full((2, 2), 10) # array([[10, 10], [10, 10]])
  c = np.full((2, 3), [[1, 2, 3]]) # array([[1, 2, 3], [1, 2, 3]])
  a = np.empty(3) # array([8.5295579e-312, 0.0000000e+000, 2.2251825e-306])
  a = np.array([[1, 2, 3], [4, 5, 6]])
  b = np.empty_like(a) # array([[         0, 1072693248,          0], [1072693248,          0, 1072693248]])
  np.linspace(1, 10, 3) # array([ 1. ,  5.5, 10. ])
  np.logspace(1, 50, 10) # array([1.00000000e+01, 2.78255940e+06, 7.74263683e+11, 2.15443469e+17, 5.99484250e+22, 1.66810054e+28, 4.64158883e+33, 1.29154967e+39, 3.59381366e+44, 1.00000000e+50])
  ```

* 함수찾기

  ```python
  np.lookfor('sum') # sum이 포함된 함수 찾기
  ```

* ndarray의 속성

  ```python
  a = np.array([[1,2,3], [4, 5, 6]])
  a.shape # (2, 3)
  a.ndim # 2
  a.dtype # dtype('int32')
  a.size # 6, 요소의 개수
  a.itemsize # 4, 요소당 byte 수
  ```

* numpy와 python의 속도차이

  ```python
  # numpy를 이용한 계산
  %time np.sum(np.arange(100000000)) # 156 ms
  # python을 이용한 계산
  %time sum(range(100000000)) # 2.84 s
  ```

* python list의 특징

  * heterogeneous, sequence, mutable

  * slicing, copy모두 기본적으로 shallow copy이다

  * deepcopy 사용하려면 copy.deepcopy호출

    ```python
    import copy
    a = [[1, 2, 3]]
    b = copy.deepcopy(a)
    a[0] is b[0] # False
    ```

* numpy ndarray의 특징

  * homogeneous, sequence, mutable
  * copy는 deep copy이다

* as 계열

  * 다른 데이터 타입을 가져와 바꿔서 사용할 때는 as라는 이름이 붙는다

    ```python
    # python의 list 데이터 타입을 float ndarray로 변환
    a = [1, 2, 3]
    print(type(a))
    print(type(a[0]))
    b = np.asfarray(a)
    print(type(b))
    print(type(b[0]))
    # python의 list 데이터 타입을 float ndarray로 변환
    # python의 list 데이터 타입이 int이면 int ndarray로 변환
    a = [1., 2., 3.]
    print(type(a))
    print(type(a[0]))
    b = np.asarray(a) # asiarray없음
    print(type(b))
    print(type(b[0]))
    ```

* 값가져오기

  * comma로 값 가져오기

    ```python
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print(b[0, 2]) # 3
    ```

  * boolean indexing

    * array와 scalar의 elementwise연산이 가능하기 때문에 가능

    ```python
    b = np.array([[1, 2, 3, 4]])
    b > 2 # array([[False, False,  True,  True]])
    b[b>2] # array([3, 4])
    ```

  * fancy indexing

    * indexing은 원래 차원에서 하나 감소해서 가져오는데 fancy indexing은 원래 차원 그대로 가져온다

    ```python
    b = np.arange(100).reshape(20, 5)
    print(b[1:3, 2:4]) # 1행부터 3행 미만이면서 2열부터 4열 미만의 데이터를 가져온다
    print(b[[1, 3, 5], [0, 2, 4]]) # 1행0열, 3행2열, 5행4열의 데이터를 가져온다
    ```

    

# 210607

* Ellipsis

  * 이후로 모든 것을 의미

  ```python
  b = np.arange(24).reshape(2,3,4)
  b[0,...] # b[0,:,:]
  ```

* 줄임표현

  * 데이터 수가 엄청 클 때 보이는 출력

  ```python
  np.arange(100000)
  np.set_printoptions(edgeitems=6) #  줄임표현 앞뒤에 출력될 갯수 지정가능
  np.arange(100000) # array([    0,     1,     2,     3,     4,     5, ..., 99994, 99995, 99996, 99997, 99998, 99999])
  ```

* 연산

  * elementwise로 vectorize 연산을 한다.

  ```python
  a = np.arange(12).reshape(3,4)
  b = np.arange(1,13).reshape(3,4)
  a + b
  # python함수도 vectorize 가능
  @np.vectorize
  def plus(a,b):
      return a+b
  plus([1, 2, 3], [4, 5, 6]) # array([5, 7, 9])
  a * b
  a * b.T # 행렬곱, np.dot(a, b.T)
  a = np.arange(6).reshape(2,3)
  a + np.array([3,4,5]) # broadcasting
  ```

* strides

  * 차원별 메모리 간격을 알려준다

  ```python
  a = np.arange(12).shape(3, 4)
  a.strides # (16, 4)
  a.itemsize # 4
  a.shape # (3, 4)
  ```

* axis

  ```python
  a = np.arange(24).reshape(2,3,4)
  # 기준이 되는 axis값이 없다고 생각하고 같은 위치의 값들 다 더해서 결과 냄
  a.sum(axis=0) # 공간기준(공간마다)
  a.sum(axis=1) # 행기준(행마다)
  a.sum(axis=2) # 열기준(열마다)
  ```

* ufunc

  * universal function의 줄임말
  * array뿐만 아니라 list, int에 대해서 모두 쓸 수 있다.

  ```python
  np.add([1, 2, 3], [4, 5, 6]) # array([5, 7, 9])
  ```

* reshape

  * 차원을 바꿔주는 기능

  * 인자 하나를 음수로 사용하면 알아서 맞춰준다(두 개 이상은 불가)

  * 원래 데이터는 변경하지 않고 새로운 값을 리턴한다

    ```python
    a = np.arange(24).reshape(3, 8)
    a.reshape(2, 3, -100)
    ```

* resize

  * 차원을 바꿔주는 기능

  * 원래의 원소 개수와 원소 개수가 안 맞아도 된다.(변형 전보다 크게 바꾸면 남는 자리는 0으로 채워짐)

  * return이 없고 자기 자신을 바꾼다

    ```python
    a = np.arange(24)
    a.resize(2, 3, 1) # return 없음
    a # array([[[0], [1], [2]], [[3], [4], [5]]])
    ```

* array 쪼개기

  * split
  
    ```python
    a = np.arange(24)
    a = np.split(a, 2) # [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])]
    b = a.reshape(4, 6)
    np.vsplit(b, 2) # np.split(b, 2, axis=0)
    [array([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11]]),
     array([[12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]])]
    np.hsplit(b, 2) # np.split(b, 2, axis=1)
    [array([[ 0,  1,  2],
            [ 6,  7,  8],
            [12, 13, 14],
            [18, 19, 20]]),
     array([[ 3,  4,  5],
            [ 9, 10, 11],
            [15, 16, 17],
            [21, 22, 23]])]
    ```
  
    
  
  * stack
  
    ```python
    a = np.arange(24)
    x, y, z = np.split(a, 3)
    np.hstack((x, y))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
    np.vstack((x, y))
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15]])
    # 합칠 array들이 1차원, 2차원일 경우는 차원을 늘린 후 합친다.
    np.dstack((x, y))
    array([[[ 0,  8],
            [ 1,  9],
            [ 2, 10],
            [ 3, 11],
            [ 4, 12],
            [ 5, 13],
            [ 6, 14],
            [ 7, 15]]])
    np.stack((x, y), axis=1)
    ```
  
  * concatenate
  
    ```python
    a = np.arange(12).reshape(3, 4)
    x, y = np.split(a, 2, axis=1)
    np.concatenate((x, y))
    array([[ 0,  1],
           [ 4,  5],
           [ 8,  9],
           [ 2,  3],
           [ 6,  7],
           [10, 11]])
    ```
  
  * 차원 확장
  
    ```python
    a[np.newaxis]
    a[np.newaxis,:,:].shape # (1, 3, 4) np.expand_dims(a, 0)
    a[:,np.newaxis,:].shape # (3, 1, 4)
    a[:,:,np.newaxis].shape # (3, 4, 1)
    
    ```
  
  * array 한줄로펴기
  
    * 결과를 리턴하고 원래값 변경 x
  
    ```python
    a = np.arange(12).reshape(3, 4)
    b = a.flatten()
    ```
  
  * structured array(구조화된 배열)
  
    * 우리만의 데이터 타입
  
    ```python
    x = np.array([('Rex', 9, 81.0), ('Fibo', 3, 27.0)], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    x[0] # ('Rex', 9, 81.)
    x['name'] # x.name과 같은 property 접근방식 사용불가, dictionary 접근방식 가능
    x.dtype.descr # [('name', '<U10'), ('age', '<i4'), ('weight', '<f4')]
    ```
  
    