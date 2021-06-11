# 210608

* pandas datatype

  * Series, DataFrame

* pandas 기본 함수

  ```python
  # 읽기
  data = pd.read_csv('pdsample/num.txt', header=None)
  data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
  # pandas의 DataFrame을 파일에 저장
  a.to_csv('pdsample/num2.csv', sep=',', header=False)
  # column(열) Series
  data[0]
  # index(행) Series
  data.iloc[0]
  # numpy로 변환
  data.values
  # 전반적인 데이터 정보 확인
  data.info()
  # 앞 5개
  data.head()
  # 뒤 5개
  data.tail()
  # random으로 10개씩 보여줌
  data.sample(10)
  # 데이터의 범위 확인
  data.index # RangeIndex(start=0, stop=244, step=1)
  # column명 확인
  data.columns # Index(['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'], dtype='object')
  ```

* pandas 데이터 값 꺼내기

  ```python
  data.iloc[3] # indexing, Series
  data.iloc[0:5] # slicing
  data.iloc[[0, 3, 5]] # fancy indexing
  
  # 둘 다 가능
  data.iloc[3][1]
  data.iloc[3, 1]
  
  data.loc[:, 'tip':'day']
  
  data['tip'] # dictionary 접근
  data.tip # property 접근
  data[['tip', 'day']] # fancy indexing
  
  data.at[0, 'tip'] # data.at[0:3, 'tip']처럼 slicing은 불가
  data.iat[0, 1] # data.at[0:3, 1]처럼 slicing은 불가
  
  data[data['tip'] >= 5] # boolean indexing
  ```

* rename

  ```python
  data2 = data.rename(columns={'size':'size_'})
  ```

* 데이터타입 변경

  ```python
  data['sex'] = data['sex'].astype('category')
  ```

* describe

  ```python
  # describe()은 기본적으로 숫자형 데이터만 분석해준다.
  # include 옵션을 사용하면 다른 분석할 데이터 타입을 지정할 수 있다.
  data.describe(include=['float64', 'object'])
  ```

* value_counts

  ```python
  # unique한 값과 그 개수를 알려준다
  data['day'].value_counts()
  ```

* where

  ```python
  a = np.array([1, 2, 3, 4])
  np.where(a>3, 0, 100) # array([100, 100, 100,   0])
  ```

* pandas에서 제공하는 통계 함수

  ```python
  data['tip'].count()
  data['tip'].mean()
  data['tip'].max()
  data['tip'].min()
  data['tip'].std()
  ```

  