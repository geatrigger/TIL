# 210517

# 210518

# 210520

# 210521

* expression(식)

  * 하나의 값으로 축약 가능

    ```python
    1 + 3
    ```

* statement(문)

  * 실행가능한 코드 조각, 어떤 값으로 축약되지 않는다

    ```python
    a = 1
    ```

* whos

  * 주피터에서 현재 네임스페이스에 저장된 변수들을 알려준다

    ```python
    %whos
    ```

* time, timeit

  * time : 셀 안에 있는 코드를 실행하는데 걸린 시간 측정

  * timeit : 여러 번 반복해서 실행평균시간 측정

    ```python
    %%time
    for i in range(1000000):
        if i == 1000000:
            print(i)
    ```

    ```python
    %%timeit
    for i in range(1000000):
        if i == 1000000:
            print(i)
    ```

* help : help(in)하면 어떤 것인지 알려줌

* 복합할당문

  * 같은 값을 동시에 할당하는 것(똑같은 주소의 값을 가리키게 됨)

    ```python
    a = b = 257
    a = 2
    print(b) # 따라서 b의 값은 변하지 않는다
    a = b = [1, 2, 3] # mutable
    a.append(4)
    print(b) # 재할당이 아닌 함수를 호출한 것이므로 b값도 변함
    ```

* unpacking

  * 좌우변 개수가 같으면 동시 할당 가능

    ```python
    a, b = 1, 2
    ```

* starred

  * unpacking 할당할 때 나머지 값을 할당

  * 단독으로 사용 불가

  * 2개 이상 사용 불가

    ```python
    a, *b = 1, 2, 3, 4 # b = [2, 3, 4]
    *b, = 1, 2, 3, 4 # b = [1, 2, 3, 4]
    ```

* interning

  * 파이썬은 느리기 때문에 중간에 cache 저장하는 기법이 있음

  * 숫자는 -5~256

  * 문자열은 a-zA-Z0-9_로만 이루어졌을 때

    ```python
    # -5~256
    a = 100
    b = 100
    a == b # True
    a is b # True
    # 그 외
    a = 257
    b = 257
    a == b # True
    a is b # False
    # a-zA-Z0-9_
    a = 'python3'
    b = 'python3'
    a == b # True
    a is b # True
    # 그 외
    a = '파이썬'
    b = '파이썬'
    a == b # True
    a is b # False
    ```

  * 문자열의 경우 intern함수를 통해 interning이 가능

    ```python
    from sys import intern
    
    a = intern('한글')
    b = intern('한글')
    a == b # True
    a is b # True
    ```

  * python에서 else가 사용되는 3가지 용법

    * 조건문

      * if ~ else에서 조건식이 False일 때 else 실행

        ```python
        a = 1
        if a > 3:
            print('yes')
        else:
            print('no')
        ```

      * existence

        * python에서 조건은 존재론(existence)적이다(truthy, falsy)

        * 0.0, None, {}, 0 + 0j, [] 등이 falsy이다.(주의 음수값은 falsy가 아니다)

        * 조건문의 경우 and이면 앞에서부터 확인하여 Truthy이면 다음 식을 실행하고, 마지막값이거나 Falsy이면 해당 식의 값을 출력한다.

        * 조건문의 경우 or이면 앞에서부터 확인하여 Falsy이면 다음 식을 실행하고, 마지막값이거나 Truthy이면 해당 식의 값을 출력한다.

          ```python
          -1 and 4 # 4
          '' and 4 # ''
          -1 or 4 # -1
          '' or 4 # 4
          ```

      * 조건문에서 할당연산 불가

        ```python
        (a = 1) == 1
        ```

    * 반복문

      * else를 반복문이 완벽하게 끝났을 때 실행

        ```python
        a = 10
        while a > 10:
            a -= 1
            print(a)
        else:
            print('end')
        # break로 중단된 경우 완벽하게 끝난 것이 아님
        a = 10
        while a > 1:
            a -= 1
            print(a)
            if a == 5:
                break
        else:
            print('end')
        ```

      * for ~ in 다음에 iterable이 온다

      * iterable

        * membership을 하나씩 차례로 반환 가능한 객체
        * 하나 이상의 데이터를 가지고 있는 container
        * iter가 정의됨
        * 하나 이상의 데이터를 가지고 있는 container

      * iterator

        * iter, next가 정의됨
        * 여기서 iter는 self를 반환하고, next는 다음 item을 가지고 온다

      * for a in iterable 과정

        * iter(iterable) -> iterator
        * next(iterator) -> next item
        * next item -> binding(a에 저장)
        * StopIteration Exception

      * iter, next 사용

        ```python
        a = iter([1,2,3])
        b = iter({1,2,3})
        c = iter('python')
        d = iter((1,2,3))
        print(type(a), type(b), type(c), type(d))
        
        print(next(a))
        print(next(a))
        print(next(a))
        print(next(a))
        ```

      * for문 바깥에 있는 a를 호출해도 정상적으로 호출이 됨

        ```python
        num = [1, 2, 3, [1, 2]]
        for a in num:
            print(a)
        a
        ```

      * python 3.7버전 이후로 dictionary로 for문을 쓸 때 정의된 순서대로 값을 반환한다(단, set은 그렇지 않다)

        ```python
        for i in {'b': 2, 'a': 1}:
            print(i)
        ```

      * keys(), values(), items()

      * tuple을 입력받을 때 일부 원소 스킵하는 법

        ```python
        for i, _, j in [(1, 2, 3), (4, 5, 6)]:
            print(i, j)
        ```

      * zip

        * iterable의 요소 개수가 같지 않으면 가장 작은 개수에 맞춰 반환

          ```python
          for i in zip([1, 2, 3], ['a', 'b', 'c', 'd', 'e'], [10, 20, 30, 40, 50]):
              print(i)
          #(1, 'a', 10)
          #(2, 'b', 20)
          #(3, 'c', 30)
          ```

          

      * enumerate

    * 예외처리문

      * 에러가 발생하도 코드를 중단하지 않고 실행

      * try-except-else-finally

      * try에서 에러 발생하면 except

      * try에서 정상적으로 처리되면 else

      * finally는 무조건 마지막에 실행

        ```python
        try:
        #     a = 1 / 'a'
            a = 1 / 0
        except ZeroDivisionError:
            print('분모는 0보다 큰 수만 가능합니다.')
        except TypeError as e:
            print('숫자만 가능합니다.')
            print(e)
        except:
            print('except')
        else:
            print('end')
        finally:
            print('program end')
        ```

        

# 210524

* function

  * def를 이용해 선언

  * 코드 작성이 줄어들고, 유지보수가 편리해짐

  * __name__은 함수의 원래 이름을 알려준다

    ```python
    def proc(a, b):
        return a+b
    proc.__name__ # 'proc'
    display = proc
    display.__name__ # 'proc'
    ```

  * 외부에서 함수 내부로 접근할 수 없다

  * 함수 내부에서 외부에 접근할 수는 있지만, 변경은 할 수 없다.

    ```python
    c = 10
    def display():
        print(c)
    display() # c 접근 가능
    t = 3
    def prn():
        t = t + 1
        print(t)
    prn() # error 발생, 외부의 t값 변경 불가
    ```

  * global, nonlocal

    * global

      * 내부와 외부의 객체를 동기화(sync)시켜준다

      * 흐름 파악하기 힘들어 남용하지 말아야 함

        ```python
        t = 3
        def prn():
            global t
            t = t + 1
            print(t)
        prn()
        ```

    * nonlocal

      * 함수 안에 있는 영역에 동기화(sync)한다

      * 외부의 객체 중 가장 가까운 객체와 동기화

        ```python
        k = 1
        def outer():
            m = 1
            def inner():
                global k
                k = k + 1
                nonlocal m
                m = m + 1
                return k, m
            return inner()
        outer()
        ```

        

* callable

  * function, class(init가 정의되었을 경우), object(call가 정의되었을 경우)가 있다.

    ```python
    # 내장함수 callable은 객체가 ()을 붙여 사용할 수 있는지를 True/False로 알려준다
    callable(sum)
    a = 1
    callable(a)
    ```

* subscriptable

  * 숫자나 문자로 원소에 접근할 수 있는 것

    ```python
    # set의 경우 subscriptable하지 않아 에러남
    k = {1, 2, 3}
    k[1]
    ```

* 매개변수(parameter), 인자(argument)

  * 매개변수 : 함수를 정의할 때 쓰는 변수

  * 인자 : 함수를 사용할 때 쓰는 변수

  * 사용방식 7가지

    * positional

      * 선언한 파라미터 순서 = 인자의 입력 순서

      ```python
      def proc(a, b):
          return a, b
      proc(2, 4)
      ```

    * keyword

      * 순서와 관계없이 파라미터 이름에 따라 값이 대입

        ```python
        proc(b=4, a=2)
        ```

        

    * positional & keyword

      * keyword방식은 positional 뒤에 나와야만 한다

        ```python
        def proc(a, b, c):
            return a, b, c
        proc(2, c=5, b=3)
        ```

        

    * positional only

      * /의 앞에 오는 인자들은 position방식으로만 입력 가능하다

        ```python
        def proc(a, b, c, /):
            return a, b, c
        proc(4, 5, 6)
        ```

        

    * keyword only

      * \*이후로는 keyword 방식만 가능

        ```python
        def proc(*, a, b, c):
            return a, b, c
        proc(a=3, b=4, c=5)
        ```

        

    * 가변 positional

      * positional 방식으로 입력되는 인자의 개수에 상관없이 모두 받아온다.

      * parameter에서 하나만 사용 가능

        ```python
        def proc(*a):
            return a
        proc()
        proc(1, 2)
        ```

        

    * 가변 keyword

      * keyword방식으로 입력되는 인자의 개수에 상관없이 모두 받아온다.

      * parameter에서 하나만 사용 가능

        ```python
        def proc(**a):
            print(type(a))
            return a
        proc()
        proc(x=2, y=3, z=[1, 2, 3])
        ```

* return

  * 함수의 결과값을 넘겨줌
  * 함수를 강제적으로 종료
  * 결과를 2개 이상 받고 싶으면 unpacking해서 받는다
  * return을 2번 이상 사용할 수 없다(써놓는다고 에러나진 않음)

* overloading

  * 함수의 이름은 같은데, 인자의 타입 또는 개수에 따라 함수가 실행되는 것
  * python은 지원 안하는 기능

  

# 210525

* functional programming

  * math의 function 개념과 차이를 최대한 줄여서 programming하는 기법

  * 문보단 식 선호

  * loop문, mutable(global, nonlocal) 사용 안함

  * 이론과 실제의 간극을 줄임

    ```python
    # input(정의역)이 없으므로 수학적 관점에서 나쁜 함수
    def f():
        return 1
    
    # output(치역)이 없으므로 수학적 관점에서 나쁜 함수
    def g():
        print(3)
    ```

  * 순수함수

    * 동일한 입력 값에 항상 동일한 출력 값을 나타냄

      ```python
      # 비순수함수
      # 매개변수 값을 변경하므로 순수함수가 아니다
      def func(a):
          a.append(3)
          a.append(4)
          return a
      # 비순수함수
      # c의 값이 달라지면 결과값도 달라지므로 순수함수가 아니다
      c = 10
      def func(a, b):
          return a+b+c
      # 비순수함수
      # dict의 상태를 변경시키므로 순수함수가 아니다
      obj = {'x': 10}
      def func(obj, b):
          obj['x'] += b
      # 순수함수
      # 동일한 입력값에 항상 동일한 출력 값을 나타냄
      def func(a, b):
          return a+b
      ```

  * 합성함수

    * 새로운 함수를 만들거나 계산하기 위해 둘 이상의 함수를 조합하는 것

      ```python
      def add(a, b):
          return a + b
      def square(x):
          return x*x
      def func(s):
          return s + 10
      func(square(add(3, 5)))
      ```

  * lambda

    * 식 형태로 되어 있다고 해서 람다 표현식(lambda expression)이라고 부른다

    * 재사용하지 않을 함수를 만들 때 사용(익명함수)

    * 함수를 간편하게 작성할 수 있어 다른 함수의 인수로 넣을 때 사용

    * 익명함수

      ```python
      anc = lambda a, b : a + b
      print(anc.__name__) # '<lambda>'
      def add(a, b):
          return a + b
      a = add
      print(a.name) # add
      ```

    * 람다표현식 자체 호출

      ```python
      (lambda x : x + 10)(3)
      ```

      

    * 람다 표현식 바깥에 있는 변수는 사용할 수 있다.

      ```python
      y = 10
      (lambda x : x + y)(3)
      ```

    * 매개변수들은 생략할 수 있지만 콜론 뒤의 표현식은 생략불가

      ```python
      (lambda :1)()
      ```

    * map, filter, reduce 함수의 인수로 사용가능

      ```python
      a = [1, 2, 3, 4]
      b = [5, 6, 7, 9]
      list(map(lambda x, y : x + y, a, b)) # [6, 8, 10, 13]
      a = [1, 2, 3, 4, 5, 6, 7, 8]
      list(filter(lambda x: x%2 == 0, a))
      from functools import reduce
      reduce(lambda x, total : x + total, [1, 2, 3, 4])
      ```

  * higer order function(고차 함수)

    * function을 인자로 사용할 수 있다.(함수이름, lambda)

    * function을 return으로 사용할 수 있다.

    * map, filter, reduce등이 있다.

      ```python
      def ss():
          return 3
      def dicplay(a):
          return a
      display(ss()) # 3
      k = display(ss)
      k() # 3
      k.__name__ # 'ss'
      ```

  * 일급객체함수(first class function)

    * 함수를 변수나 자료구조에 저장할 수 있다.

      ```python
      def plus(a, b):
          return a + b
      data = plus
      ```

      

    * higher order funciton

  * closure

    * 함수 실행 후 외부에서 해당 함수 내부 값에 접근하기 위해 사용

    * 함수를 정의할 때 outer함수는 inner함수를 return하고, inner함수에서는 outer함수의 자원을 사용한다.

      ```python
      def t(a):
          def x():
              return a + 1 # 내부함수에서 외부함수의 자원을 사용해야 한다
          return x # 외부함수에서 내부함수를 리턴해야 함
      # inner함수를 리턴받아 p변수에 할당한다.
      p = t(3) # 일급 객체 함수가 가능해야 가능
      # 함수를 실행한다. => x()내부함수가 실행된다.
      p()
      ```

  * functional programming에서 loop를 사용하지 않는 기법

    * comprehension

      * iterable한 오브젝트를 생성하기 위한 방법 중 하나

      * 실행속도가 빠르고 코드가 간결

        ```python
        # list comprehension
        a = [i for i in range(5)]
        # set comprehension
        a = { i for i in 'apple'}
        # dict comprehension
        a = {x:y for x, y in zip(range(3), ['red', 'green', 'blue'])}
        ```

        

    * recursion

      * 함수가 자기자신을 다시 실행하는 기법

        ```python
        def x(t):
            if t < 3:
                return 1
            return x(t-1)
        ```

      * Tail recursion

        * 재귀호출 시 stack에 데이터가 지나치게 많이 쌓이는 문제가 발생할 수 있다

        * 그래서 return값에 자기자신 함수 하나만 리턴하여 스택에 남는 데이터를 줄인다

          ```python
          def febTailRecursion(n, previousFibo, previousPreviousFibo):
              currentFibo = None
              
              if n < 2:
                  return n * previousFibo
              # 이번 호출의 피보나치 수를 구하고
              currentFibo = previousFibo + previousPreviousFibo
              # 다음번 재귀 호출을 위해 앞의 피보나치 수를 앞의 피보나치 수로 한 칸 미루고
              previousPreviousFibo = previousFibo
              # 다음 번 재귀 호출을 휘애 현재의 피보나치 수를 한 칸 미룬다.
              previousFibo = currentFibo
              
              return febTailRecursion(n-1, previousFibo, previousPreviousFibo) # (5, 1, 1) (4, 2, 1) (3, 3, 2), + 1 같은 것도 안됨
          ```

          

    * iterator, generator

      * iterator

        * 데이터를 사용할 때마다 메모리에 올린다(효율적인 메모리 사용)

        * index, len 사용불가

          ```python
          a = {1, 2, 3}
          b = iter(a)
          next(b)
          next(b)
          ```

      * generator

        * tuple

          ```python
          # generator 만드는 첫번째 방법 : tuple을 comprehension처럼 사용
          t = (i for i in range(5))
          ```

        * yield

          * 함수를 빠져나오고 다시 실행했을 때 yield 다음부터 실행하게 함

          ```python
          # generator 만드는 두번째 방법 : yield를 이용하는 방법
          # 작업하는 데이터를 모두 데이터 올리기엔 불가능한 경우가 있다
          # 그래서 하나씩 처리할 수 있는 방법 제공
          def upper_gen(x):
              print('upper_gen start')
              for i in x:
                  result = i.upper()
                  print('{} generated'.format(result))
                  yield result
          
          fruits = ['apple', 'pear', 'grape', 'pineapple', 'orange']
          for i in upper_gen(fruits):
              print('{} printed'.format(i))
              print(i)
          ```

          

    * map, filter, reduce

# 210526

* with문으로 파일입출력

  ```python
  # write
  with open('info.txt', 'w') as f:
      for i in range(100):
          name = random.choice(hanguls) + random.choice(hanguls)
          weight = random.randrange(40, 100)
          height = random.randrange(140, 200)
          f.write('{},{},{}\n'.format(name, weight, height))
  print('save')
  # read
  with open('info.txt', 'r') as f:
      for line in f:
          name, weight, height = line.strip().split(',')
          bmi = int(weight) / ((int(height) / 100) ** 2)
          result = ''
          if 25 <= bmi:
              result = '과체중'
          elif 18.5 <= bmi:
              result = '정상체중'
          else:
              result = '저체중'
  #         print(result)
          print('\n'.join(['이름:{}', '몸무게:{}', '키:{}', 'BMI:{}', '결과:{}']).format(name, weight, height, bmi, result))
  ```

* 객체지향 프로그래밍

  * 함수와 클래스의 차이

    * 함수
      * 정의 + 호출
      * def사용
      * 외부에서 내부 접근 불가
    * 클래스
      * 정의 + 인스턴스
      * class사용
      * 외부에서 내부 접근 가능(접근제한자가 없다, 자기결정권을 존중하는 파이썬 철학)

  * isinstance

    * 변수가 어떤 클래스의 인스턴스인지 확인

      ```python
      a = 1
      type(a) # int
      isinstance(a, int) # True
      isinstance(a, str) # False
      ```

      

  * 클래스, 인스턴스에 있는 값 접근

    * 인스턴스 변수 동적 할당 가능(class 정의할 때 안써도 됨)
    * 인스턴스의 경우 인스턴스 변수를 살펴보고 없으면 클래스 변수를 살펴본다
    * 클래스의 경우 클래스 변수를 살펴본다
    * 인스턴스의 경우 인스턴스 메소드를 살펴보고 없으면 클래스 메소드를 살펴본다
    * 클래스의 경우 클래스 메소드를 살펴본다
    * 클래스 변수, 인스턴스 변수, 인스턴스 메소드 등 모두 마지막에 업데이트 된 값만 남는다.(같은 값의 변수, 메소드 선언하면 안됨)

    ```python
    class Location:
        x = 1 # 클래스 변수
        def city(self, x):
            self.x = x # 인스턴스 변수
    class X:
        x = 1 # 클래스 변수
    xi = X() # class는 callable하고, X()는 인스턴스를 만든다
    print(X.x) # 클래스 변수 접근
    print(xi.x) # 인스턴스 변수가 없으므로 클래스 변수 접근
    xi.x = 2
    print(X.x) # 클래스 변수 접근
    print(xi.x) # 인스턴스 변수 접근
    X.x = 3
    print(X.x) # 클래스 변수 접근
    print(xi.x) # 인스턴스 변수 접근
    ```

  * decorator

    * @을 붙이면 decotrator이다.

    * @ 밑에 있는 함수를 @에 적힌 함수의 인자값으로 전달하겠다는 의미

      ```python
      def x(fun):
          def y(x):
              return fun(x + 1)
          return y
      @x
      def s(a):
          return a
      s(5) # 6
      ```

  * instancemethod, classmethod, staticmethod

    * instancemethod

      * 인스턴스의 변수에 접근 가능
      * 첫번째인자는 무조건 self가 들어가야 함
      * self는 인스턴스 자기 자신

    * classmethod

      * 클래스의 변수에 접근 가능
      * 첫번째인자는 무조건 cls가 들어가야 함
      * cls는 인스턴스의 클래스

    * staticmethod

      * 클래스 안에 있는 값에 상관없는 기능
      * 인자가 없어도 된다
      * 일반함수와 같다

      ```python
      class B:
          x = 1
          def xx(self, x): # instance method
              self.x = x
          @classmethod
          def yy(cls, x):
              cls.x = x # class method
          @staticmethod
          def zz(a, b):
              print(a+b) # static method
          @staticmethod
          def kk(): # static method
              print(x)
      B.yy(10) # classmethod : 인스턴스 생성하지 않고 호출할 수 있다.
      B.zz(3, 4) # staticmethod : 인스턴스 생성하지 않고 호출할 수 있다.
      b = B()
      b.xx(5) # instancemethod : 인스턴스를 생성한 후 호출할 수 있다.
      ```



# 210527

* 상속(inheritance)

  * parent가 가지고 있는 자원을 child에게 넘겨줌

  * python의 모든 객체는 object를 상속받는다(python3부터는 object 상속표시 생략가능)

    ```python
    class Parent:
        x = 'parent'
        @classmethod
        def output(cls):
            return cls.x
        @staticmethod
        def prn():
            return Parent.x
    # Parent를 Child에 상속
    class Child(Parent):
        x = 'child'
        def display(self):
            print(Parent.output())
            print(Child.output())
    cd.prn() # 'parent'
    ```

* mangling(Name Decoration)

  * 함수나 변수를 선언했을 때, 선언시 사용했던 이름을 컴파일러가 컴파일 단계에서 일정한 규칙을 가지고 변형

  * 클래스 내부에선 그렇지만 파일에선 그렇지 않다

    ```python
    class A:
        x = 1
        __y = 2 # private취급하기 위해 컴파일러가 새로운 이름 지어줌
    A._A__y # 2
    ```

* vars

  * 인스턴스 안에 정의된 값 출력

  * __ dict __ 와 같음

    ```python
    class A:
        x = 1
        __y = 2
    a = A()
    vars(a) # {}
    a.t = 3
    vars(a) # {'t':3}
    a.__dict__ # {'t':3}
    ```

* import

  * 파이썬 파일을 객체로 불러온다.

    ```python
    # gosu.py 내용
    a = 1
    _a = 2
    __a = 3
    
    def b():
        return 1
    
    def _b():
        return 2
    
    def __b():
        return 3
    ```

    

    ```python
    import gosu # gosu.py를 불러옴
    
    gosu.a # 1
    gosu._a # 2
    gosu.__a # 3
    ```

* as

  * alias의 줄임말로 별칭이라는 뜻

    ```python
    import gosu as g
    
    g.__name__ # gosu
    ```

* from

  * 어디에서 가져오는지

    ```python
    from gosu import a as f
    f # 1
    ```

* double underbar

  * 이름의 앞, 뒤에 언더바가 2개씩 붙은 것들은 python에서 미리 정해져 있는 기능
  * __ init __ , __ call __ , __ len __

* __ init __

  * 파이썬의 생성자(constructor)
  * 인스턴스화 할 때 자동으로 실행

* __ new __

  * 인스턴스화 할 때 자동으로 실행
  * new 에서 인스턴스를 리턴하고 init이 실행됨

* __ class __

  * type과 같다

  * 어떤 class인지 알려준다

    ```python
    class A:
        def __init__(self):
            self.a = 4
            print('__init__')
        def __del__(self):
            print('__del__')
    a = A()
    type(int) # type
    int.__class__ # type
    type(A) # type
    type(a) # __main__.A
    ```

* __ dir __

  * 어떤 method와 변수가 있는지 알려준다

  * dir()는 정렬해서 알려준다

    ```python
    dir(a)
    a.__dir__()
    ```

* __ iter __

  * iter 사용가능

  * iterable에서 iterator로 변환

    * iterable은 상속받으면 만들수있다

      ```python
      from collections.abc import Iterable
      ```

      

* __ next __

  * next 사용가능

* __ doc __

  * docstring

  * 함수나 클래스를 선언할 때, signature 바로 앞에 따옴표 세개를 이용해서 설명을 기재할 수 있다.

    ```python
    class A:
        '''클래스 설명서'''
        def __init__(self):
            '''init 설명서'''
            self.a = 10
            print('__init__')
    a = A()
    a.__doc__ # '클래스 설명서'
    a.__init__.__doc__ # 'init 설명서'
    ```

* inspect

  * 타인이 만든 코드 확인 가능

    ```python
    from PIL import Image
    import inspect
    
    im = Image.open('1.jpg')
    print(insect.getsource(im.rotate))
    ```

* __ getattr __

  * 정의되어 있지 않은 인스턴스 변수를 호출할 때 이 메소드를 호출한다.

    ```python
    class A:
        def __init(self, x):
            self.x = x
            self.y = 1
        def __getattr__(self, value):
            print('call')
    a = A(3)
    a.y # 1
    a.k # 'call'
    ```

  * 기존에 정의되어 있는 __ getattribute __ 에서 정의되어 있지 않은 인스턴스 변수일 때 __ getattr __을 호출하는 것 같다

* __ setattr __ , __ delattr __

  * 각각 인스턴스 변수를 설정, 삭제할 때 호출

    ```python
    class A:
        def __setattr__(self, name, value): # 새로운 변수를 할당할 때 실행됨
            print('__setattr__(%s)=%s called' % (name, value))
            
        def __delattr__(self, name): # del 할 때 실행
            print('__delattr__(%s) called' % name)
    a.x = 3 # __setattr__(x)=3 called
    del a.x # __delattr__(x) called
    ```

* decorator로 getter, setter처럼 사용하기

  ```python
  class Exam:
      def __init__(self):
          self.__writing_grade = 0
          self.__math_grade = 0
      @staticmethod
      def __check_grade(value):
          if not(0 <= value <= 100):
              raise ValueError('0~100')
      # method 이름이 같아야함, get set할 변수와 이름 같아야함
      @property
      def writing_grade(self): # getter처럼 사용
          return self.__writing_grade
      @writing_grade.setter
      def writing_grade(self, value): # setter처럼 사용
          # Exam._Exam__check_grade(value)
          self.__writing_grade = value
      @property
      def math_grade(self):
          return self.__math_grade
      @math_grade.setter
      def math_grade(self, value):
          # Exam._Exam__check_grade(value)
          self.__math_grade = value
  exam = Exam()
  exam.writing_grade = 97
  print(exam.writing_grade) # 97
  ```

# 210529

* 다중 상속

  * 여러 부모 클래스 갖는 것 가능

  * 상속체계순서(MRO, Method Resolution Order)가 명확하지 않으면 클래스 선언시 에러

    ```python
    class A:
        pass
    
    class B(A):
        pass
    
    class C(A, B): # 다중 상속은 가능하지만, B가 A를 상속받는데 C가 A, B(A상속받은)를 상속받는다고 해서 에러 -> 상속체계가 불명확하다
        pass
    ```

  * mro() (리스트 반환), __ mro __ (튜플 반환) 로 상속체계 순서를 알 수 있다.

    ```python
    class A:
        a = 1
    
    class B:
        b = 1
    
    class C(A, B):
        pass
    C.mro() # [__main__.C, __main__.A, __main__.B, object]
    C.__mro__ # (__main__.C, __main__.A, __main__.B, object)
    c = C()
    c.mro() # 불가, 클래스에서만 사용가능
    ```

  * 다이아몬드 상속구조

    * Base를 상속받는 A, Base를 상속받는 B 두 개를 동시에 상속받는 구조

      ```python
      class Base:
          pass
      class A(Base):
          a = 1
      class B(Base):
          b = 1
      class C(A, B):
          pass
      C.mro() # [__main__.C, __main__.A, __main__.B, __main__.Base, object]
      ```

    * super()를 사용하면, 각 생성자는 한 번씩만 호출되고, Base의 속성이 A나 B의 속성을 덮어씌우지 않게 된다

      ```python
      class Base:
          def __init__(self):
              self.x = 'Base'
              self.y = 'Base'
              print('Base')
      class A(Base):
          a = 1
          def __init__(self):
              super().__init__() # super()로변경
              self.x = 'A' 
              print('A')
      class B(Base):
          b = 1
          def __init__(self):
              super().__init__() # super()로변경
              self.y = 'B' 
              print('B')
      class C(A, B):
          def __init__(self):
              super().__init__() # python3에서 super(C, self)를 super()라 표현 가능
              print('C')
      c = C() # Base B A C
      issubclass(c, (A, int)) # True, 여러 개를 넣었을 때 하나라도 상속받으면 True
      isinstance(c, C) # True
      ```

* class 예제

  * 내부 변수들은 private처럼 취급하기 위해 __ 를 붙여줌

  * getter, setter를 이용해 변수 접근

  * 클래스명 : Sales
    -item:str
    -qty:int
    -cost:int

    Sales(self, item=None:str, qty=None:int, cost=None:int)
    +setItem(item str):None
    +setQty(qty int):void
    +setCost(cost int):None
    +getItem():str
    +getQty():int
    +getCost():int
    +toString():str
    +getPrice():int

    예시) 품목 : apple 수량 : 5 단가 : 1200원 금액 : 6000원

    apple 1200원짜리 5개 구입하면 6000원이 필요함

  * ```python
    
    ```

    ```python
    class Sales:
        def __init__(self, item=None, qty=None, cost=None):
            self.__item = item
            self.__qty = qty
            self.__cost = cost
        def setItem(self, item): # setter
            self.__item = item
        def setQty(self, qty):
            self.__qty = qty
        def setCost(self, cost):
            self.__cost = cost
            
        def getItem(self, item): # getter
            return self.__item
        def getQty(self, qty):
            return self.__qty
        def getCost(self, cost):
            return self.__cost
            
        def toString(self):
            return '{} {}원짜리 {}개를 구입하면 {}원이 필요함'.format(self.__item, self.__qty, self.__cost, self.getPrice())
        def getPrice(self):
            return self.__qty * self.__cost
    ```

    

# 210601

