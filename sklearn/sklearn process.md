# supervisor

* 정답이 있는 데이터셋에서 학습

* 학습예시

  ```python
  from sklearn.neighbors import KNeighborsClassifier
  
  knn = KNeighborsClassifier()
  X_train, X_test, y_train, y_test = train_test_split(breast_cancer.iloc[:,:-1], breast_cancer.target)
  # fit(적합)/train(학습)
  knn.fit(X_train, y_train) # 학습데이터, 정답데이터
  knn.score(X_test, y_test) # 0.9090909090909091
  ```

* regression을 classification으로 환원

  ```python
  df = pd.DataFrame()
  df['level'] = pd.cut(data_boston2.target, 10) # 총 target label의 개수 10개
  # df['level'] = pd.qcut(data_boston2.target, 10) # label에 들어가는 개수 균등하게 10개
  
  from sklearn.preprocessing import Binarizer
  
  custom_threshold = 0.4
  pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)
  
  binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
  custom_predict = binarizer.transform(pred_proba_1)
  get_clf_eval(y_test, custom_predict)
  #오차 행렬
  #[[98 20]
  # [10 51]]
  #정확도 : 0.8324, 정밀도: 0.7183, 재현율 : 0.8361
  ```

* 기계학습 관련 이론

  * 큰 수의 법칙 : 데이터가 많을수록 좋다
  * 중심 극한 정리 : 표본이 크면 정규분포에 가까워진다
  * No Free Lunch : 성능이 좋아지면 오래걸리고, 속도가 빠르면 성능이 낮아진다

* pandas dataframe 시각화 함수

  * pandas dataframe에서 호출(내부적으로는 matplotlib사용)

    ```python
    data_iris2.boxplot()
    data_iris2.hist()
    ```

  * class를 구분해서 데이터확인(seaborn)

    ```python
    import seaborn as sns
    
    sns.pairplot(data_iris2, hue='target')
    ```

* 모델저장

  ```python
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression
  import pickle
  import joblib
  import numpy as np
  
  iris = load_iris()
  X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
  lrmodel = LogisticRegression(max_iter=4000)
  lrmodel.fit(X_train, y_train)
  lrmodel.predict(X_test)
  
  # joblib
  joblib.dump(lrmodel, './model/iris_model.pkl')
  model_from_joblib = joblib.load('./model/iris_model.pkl')
  
  # pickle
  with open('./model/iris_pic_model.pickle', 'wb') as f:
      pickle.dump(lrmodel, f)
  with open('./model/iris_pic_model.pickle', 'rb') as f:
      model_pickle = pickle.load(f)
  model_pickle.score(X_test, y_test)
  ```

* 전처리

  * 중복제거

  * 불필요한 속성 제거

  * null값처리

    ```python
    import missingno
    import matplotlib.pyplot as plt
    # null값 찾기
    mpg[mpg.horsepower.isnull()]
    # matrix
    missingno.matrix(mpg)
    # bar
    missingno.bar(mpg)
    plt.ylim((.9, 1))
    
    # null 처리방법들
    # 해당 feature 삭제
    # 해당 row 삭제
    mpg.dropna()
    # 평균을 구해서 넣기 or 작은값 넣기
    titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
    titanic_df['Cabin'].fillna('N', inplace=True)
    titanic_df['Embarked'].fillna('N', inplace=True)
    
    ```

  * 데이터인코딩

    * 머신러닝에서는 숫자 데이터만 써야한다
    * 단, sklearn에서 y값에 대해서는 문자열을 사용해도 되도록 만들었다
    * x, y값 모두 encoding할 수 있고, y값의 경우 다른 패키지와 연동을 위해 label encoding하는 것이 좋다.

    ```python
    mpg.origin.value_counts() # usa 245, japan 79, europe 68
    # 데이터 타입을 category로
    mpg.origin = mpg.origin.astype('category')
    
    # encoding 방식
    # one hot encoding
    m = mpg.origin.str
    m.get_dummies()
    # pd.get_dummies(mpg.origin)
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    ohe.fit_transform(iris[['species']]).toarray() # array([[1., 0., 0.], ...
    ohe.inverse_transform([[1,0,0],[0,1,0],[0,0,1]]) # array([['setosa'], ['versicolor'], ['virginica']], dtype=object)
    
    # label encoding
    iris.species.unique() # array(['setosa', 'versicolor', 'virginica'], dtype=object)
    iris.species.map({'setosa':0, 'versicolor':1, 'virginica':2})
    from sklearn.preprocessing import LabelEncoder
    ll = LabelEncoder()
    ll.fit_transform(iris.species) # array([0, 0, 0, ... , 2, 2, 2])
    ll.inverse_transform([0,1,1,1,2,2,0]) # array(['setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'setosa'], dtype=object)
    
    # 1차원으로 평평하게
    np.ravel(iris[['species']])
    ```

    

  * train, test 데이터 분리(hold out)

    * 과적화방지를 위해 사용

    * train, validation 데이터 분리에도 사용

      ```python
      X_train, X_test, y_train, y_test = train_test_split(breast_cancer.iloc[:,:-1], breast_cancer.target)
      # 비율 유지하며 쪼갰는지 확인
      y_train.value_counts()
      y_test.value_counts()
      ```

  * 교차검증(cross validation)

    * cross_val_score

      * 내부적으로 StratifiedKFold사용

      * 불균형한 분포도를 가진 레이블 데이터 집합을 위한 KFold 방식

        ```python
        lr = LogisticRegression(max_iter=4000)
        cross_val_score(lr, iris.data, iris.target, cv=3)
        ```

    * cross_validate

  * feature scaling

    * StabdardScaler, MinMaxScaler, RobustScaler, Normalizer 등이 있다

    * 스케일의 범위를 비슷하게 만들어줘서 모델의 성능향상효과

    * 예시

      ```python
      mm = MinMaxScaler()
      # stratify는 데이터의 비율이 동일하게 나누는 옵션이다.
      X_train, X_test, y_train, y_test = train_test_split(breast_cancer.iloc[:,:-1], 
                                                          breast_cancer.target, 
                                                          stratify=breast_cancer.target)
      mm.fit(X_train[['worst area']])
      X_train['worst area'] = mm.transform(X_train[['worst area']])
      X_test['worst area'] = mm.transform(X_test[['worst area']])
      
      lrl = LogisticRegression(max_iter=4000)
      lrl.fit(X_train, y_train)
      lrl.score(X_test, y_test) # 0.958041958041958
      ```

      

* 알고리즘 생성

  * GridSearchCV

    * 좋은 알고리즘, 하이퍼파라미터 찾기

      ```python
      from sklearn.pipeline import Pipeline
      from sklearn.model_selection import GridSearchCV
      from sklearn.preprocessing import StandardScaler
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.linear_model import LogisticRegression
      
      pipe = Pipeline([('ss', StandardScaler()), ('clf', KNeighborsClassifier())])
      
      # pipeline을 통과하는 경우 mangling되어 이름+원래하이퍼파라미터명 으로 해줘야 한다
      grid = GridSearchCV(pipe, [
          {'clf': [KNeighborsClassifier()], 'clf__n_neighbors': [2, 3, 4, 5]},
          {'clf': [LogisticRegression()], 'clf__penalty': ['l2']}
      ])
      grid.fit(wine.iloc[:, :-1], wine.target)
      print(grid.best_score_) # 0.9831746031746033
      pd.DataFrame(grid.cv_results_)
      ```

* 테스트

  * predict

  * score

  * 분류 성능 평가 지표

    * 정확도(Accuracy) : True / 전체

    * 오차행렬(Confusion Matrix)

      ```python
      from sklearn.metrics import confusion_matrix
      
      confusion_matrix(y_test, fakepred)
      #array([[405,   0],
      #       [ 45,   0]], dtype=int64)
      #[[TN, FP],
      #[FN, TP]]
      ```

      

    * 정밀도(Precision) : TP /Positive (양성, 음성 비율이 적절할 때, 양성 데이터 예측 성능에 초점)

    * 재현율(Recall) : TP / FN + TP(양성 데이터 예측 성능에 초점, 양성을 음성으로 잘못 판단하면 절대 안되는 경우)

    * PPV(Positive Predictive Value) : TP / TP + FN

    * F1 스코어 : 2 * (정밀도 * 재현율) / (정밀도 + 재현률) (양성, 음성의 비율이 unbalance할 때)

    * 특이성 : TN / FP + TN(음성을 음성이라고 예측한 비율)

    * FPR : FP / TN + FP(음성을 양성으로 예측한 비율)

    * ROC AUC

      ```python
      from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
      
      def get_clf_eval(y_test, pred):
          confusion = confusion_matrix(y_test, pred) # 오차행렬
          accuracy = accuracy_score(y_test, pred)    # 정확도
          precision = precision_score(y_test, pred) # 정밀도
          recall = recall_score(y_test, pred)        # 재현율
                                      
          print("오차 행렬")
          print(confusion)
                                      
          print('정확도 : {0:.4f}, 정밀도: {1:.4f}, 재현율 : {2:.4f}'.format(accuracy, precision, recall))
      ```

    * precision과 recall 그래프

      ```python
      import matplotlib.pyplot as plt
      import matplotlib.ticker as ticker
      %matplotlib inline
      
      def precision_recall_curve_plot(y_test, pred_proba_c1):
          precisions, recalls, thresholds = precision_recall_curve(y_test, prd_proba_class1)
          
          # X축을 thresholds 값, Y축은 정밀도, 재현율 값으로 각각 Plot 수행
          plt.figure(figsize=(8,6))
          threshold_boundary = thresholds.shape[0] # (143, 0)에서 143 추출
          
           # thresholds는 143이고, precisions과 recalls는 144로 X축과 Y축 값의 개수 가 맞이 낳으므로
          # 이 precisions과 recalls 갓으로 그래프를 그리면 오류 발생
          # y 값을 [0:threshold_boundary]로 143개 추출해서 X축 개수와 맞춤
          plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
          
          plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')
          
          # threshold 값 X축의 Scaledmf 0.1 단위로 변경
          # xlim() : X축 범위를 지정하거나 반환
          start, end = plt.xlim() # X축 범위 반환
          plt.xticks(np.round(np.arange(start, end, 0.1), 2))
          
          
           # X축, Y축 label과 legend, grid 설정
          plt.xlabel('Threshold Value')
          plt.ylabel('Precision and Recall Value')
          plt.legend(); plt.grid()
          plt.show()
          
          
      precision_recall_curve_plot(y_test, lr_clf.predict_proba(X_test)[:, 1])
      ```

      ![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfEAAAFzCAYAAAAuSjCuAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd3zV1f3H8de5N3uTCSFh7z1lOQLugYp7V+uos61aq7b1p1ZtbbWt1o11tbUqarWiuFBwgMqSjcgSCJuETLJzfn98wzQJNyE333tv3s/H4z7u+t5v3omYT875nmGstYiIiEjw8bgdQERERJpHRVxERCRIqYiLiIgEKRVxERGRIKUiLiIiEqRUxEVERIJUmNsBmiopKcn26NHD7RjNUlpaSmxsrNsxmkXZ3RPM+ZXdHcruDn9mnz9//k5rbdrBrwddEc/IyGDevHlux2iWmTNnkpOT43aMZlF29wRzfmV3h7K7w5/ZjTHr63td3ekiIiJBSkVcREQkSKmIi4iIBCkVcRERkSClIi4iIhKkVMRFRESClIq4iIhIkFIRFxERCVIq4iIiIkHKb0XcGPO8MWa7MWZpA+8bY8zfjTGrjTGLjTHD/JVFREQkFPmzJf4icFIj758M9Ky7XQM85ccsIiIiIcdva6dbaz83xnRp5JAzgH9aay3wtTEmyRjTwVq7xV+Z6rX2M4hNg4x+rfplRUTEPYVlVWwpLKNP+wQAZq7cjj3omKykaHpmxFNba/ls1Y4fnaNzcgzd0uKorK5l1pqdLNlRjV25nf6ZCaTHR7XCdwHGqaF+OrlTxN+11g6o5713gQettV/WPf8EuN1a+6PdTYwx1+C01klLSxs+ZcqUFss4Zvbl5KWM5PveN7TYORtSUlJCXFyc37+OPyi7e4I5v7K7Q9kP7cWlFeyqsNw83Cm2V35YSs1B5fC4TmFc0i+SqlrL1R/t/tE5TusWzjm9IiiutNz06b73bxgSycj2LdtGHj9+/Hxr7YiDX3dzFzNTz2v1/kVhrZ0MTAbo3bu3bdFdYuZFkNmhA5mtsGuOdudxRzBnh+DOr+zuUPZD++O3nxMd5SUnZxwAb3Yv4OBGbWpcJNnJMdTWWt7qUfCjc2QkRJGZFE1VTS1v9SlkwYIFDBs2jC4psbSLjfD79wDuFvFcIHu/51nA5taPYWjgbwcREQlBuyurWbW9mBsn9Nz72pDspAaP93gMQzu1a/D9cK+HoZ3aUbjW2+hx/uDmFLN3gMvqRqmPBgpb/Xo4gDHgx0sKIiISWJZuKqLWwuCsRLejHDa/tcSNMa8AOUCqMSYXuBsIB7DWPg1MA04BVgO7gSv8leWQirfAui8gawSER7sWQ0RE/G/RRqdrfFBWw63vYOHP0ekXHuJ9C/h/NNmhRCbA6unO7Zg7YPydbicSERE/Gtk1mdtO7E1afKTbUQ6bm9fEA8Nlb0P+Onj9J1CY63YaERHxsyHZSY1eAw8mWnY1IRO6jIO49rB7p9tpRETEj0oqqpn7Qz7lVTVuR2kRKuJ7xKZAqYq4iEgom79+F+c+/RULNuxyO0qLUHf6HjGpsHMVrP/qx++l9oTY1NbPJCLSwqy1lFXVEB3uxZj6lusIbYs2FmAMDOwY/CPTQUV8n6RsWPoGvFDPcu/Zo+HKD1s/k4hIC7t36nJenP0DxkBcZBjxkWHERYXx+rVjSYwOp7i8irjIsJAt8ItzC+iWGkt8VLjbUVqEivgeR98G3XJ+PGd83vOwZobzeoj+oxaR0Fdba/F4DNfldCcmwovXYygur6akopqS8mqiwp2rq3e8uYRv1uWRmRRNRkIU7ROi6NU+nktHdwZg6aZCPMaQFBNOcmwEUeFeN7+tJrHWsnBjIUf3Cp2eVRXxPSJinSJ+sJ3fw4p3oHQHxKW3dioRkcNireUfX6zjq7V5PHvZCDISovj1SX0aPP6E/hnERHjZWlTO+rxS5qzLp8um2L1F/NdvLGb5lqK9x3dPi+X0wR35xXE9GzplwNhSWM7OkgoGh8D88D1UxA8lubtzn7daRVxEgkpJRTW3v7GY95Zs4aT+7amsriU6ovGW8xlDOnLGkI4HvFZdU7v38f2TBrCtsJyCsip2FFewYMMuisqrAKiptfzu7aWc2D+DwVlJrbZ+uK9S4yJ549oxZCfHuB2lxaiIH0rKfkW881h3s4iI36zPK2XVthISY8JJiAqntCq4l2Nevb2Ea/89n7U7Srjz5D5cc3S3Zl/nDvPum8g0rJG1wdftLOXDZVt5Zc4GADomRdO3QzzX5XRneOdktheV8/W6fFJiI0iOjSAlLoLkmIgDzu9PEWEeRnRJbpWv1VpUxA8lqRMYL+xa73YSEfGjn/1rPt9tLd77/Mwe4ZwKVFTXsGZ7KdnJ0UEz4KvWWq7993x2lVby7ytHMbZH61wD7pEexxe/Hs/CjQUs2VTI0k2FrNpWQlml05JfnFvIz1/59kef+89VTsZv1ubx2LflfLxryX6FPpKje6aRGHP4A9FembOBzikxjO2ua+Jth8cLnjCorXY7iYj4SVF5FSu3FXPJ6E6c0K89ReVV7Fr/HdZaznpyNss2O9eAo8O9pCdE0jEpmv9cPRqA95dsYVNBGe1inKKTEB1OUkw43dNafz/vPd3eHmN45PwhJMdGkJnUuvtBxEaGMa5HKuPq+cNhXI9Upt9yNHklleSVVpJXUkFeaSVdUmMBKCyrYktpLT8s3Ur+7sq944w/vvloEmPCeX/JFqav2M6J/TOY0Ce9SS342lrLH95bwelDMlXERURCidcY/nTWIAZnJ9G7fTwAM/O/xxjDm9eNZfqKbWwuKGNbUQXbiyuorN632td/v93Ex8u3HXC+rHbRfHn7BACueGEOCzcWEB8VTkJ0GPGR4fRuH889p/cHYMrcjRRXVBMfFUZCVDiJ0eFktYtu8nXbvJIKbnrlW/pnJjAuFgYE4Dzo6AgvPdLj6dHA8KIT+rcnYkcMOTk51NRaCnY7xb5TivOz2FpUzsfLt/Lmglz6dkjgN6f0YUy3FJ+K+bq8UoorqkNqUBuoiIuIEBsZxnkjs+t9Lyrcy2mDMhv87ORLh1NUXk3B7krySyspLKs64P0JfTPo2C6a4vJqisqqKC6vZtfuyr3vPz9r3QHd+ACjuyXz6jVjALjhPwuorqklLT6StLgo0uIj6d0+nuGd912bXrixgOv+PZ/80komDe0IJdub/DMINF6PISUukpS4fZuUXDGuK5eM7swHS7fy4Pvfcelzc8jpncaLVxwBwLaictLjI+u95LE4t27nsuzA++PmcKiI+6p0B1QUQ2S820lEpIV9sWoH2e1i9nbrNoUxhsRopwXdOeXHn98zNash7950JCUV1RSXV1NYVkVhWRXh+7Usq2tqWbezlLk/7CK/1Cn+k4Z2ZHjndtTWWn7+6rd8tGwb6QmRvHndWAZ0TGTmzDVN/j6CRbjXw8TBmRzXN4PpK7YRUzfavri8inEPfkpUuJce6XH0zoinZ0Ycx/RKo2dGPIs2FhId7qWHC5c5/ElF3Bfh0bDwZWeXs5+843YaEWlB1lp+8epCJvRJ5+FzB7f61w/zekiKiSApJoL6+gKeuXTE3seV1bXklVbsfb6poIylmwo5pncafz57UMBN6fKn6AgvEwcf2ENy7xn9+X5rMSu3FfPxim28Nm8jYR5Dz4x42idGMaBjQquNhG8tKuK+uPxdmH4PbF7odhIRaSF/nLaCWWt20jM9nvzSSoZ2CvxrpRFhHjok7huolp0cw8zbxruYKHDER4Vz8agDez12llQQ7nGK9vje6fTtkOBGNL8KrT9J/KX9QOhylLNVaXmh22lEpAVcn9ODhKhwPlmxDa/HMKZbituRpIWlxkXunZrWu308x/RKczlRy1NL3FfJ3Zz7/HWQOcTdLCLSbNuLykmIDicxJpz/XD0aay27K2uIjdSvQwk+aon7Krmrc7/xG3dziEiz1dQ6i6Bc8o9vsHWTkI0xKuAStFTEfdWuroi//2vYne9uFhE5JGstv5+6nGWb910Ce/aLtSzYUMAlozsHxcprIoeiIu6ryDg44mfO49Kd7mYRkUMqq6rh+VnrOPXvXwKwcmsxf/3oe07q354zhjQ871skmKiIN0W3HOe+qtTNFCLSBMf1Tae21nLTKwuIjwrjgUkD1AqXkKEi3hQRdQs5VKqIiwSLkV2SscCO4gr+eNbAA1YAEwl2Gs3RFBF1K/3kr4OEjo0fW4+osq3OZ4OQX7N7IyCx6T9PEV95PYav7jyWqPDG99IWCTYq4k0RXbcYxDs3NuvjowGCdHC737MfczuM/40fv4C0JdOWbOG7rcVMvfFIOtVtJKICLqFIRbwpUrrDRVOaPTp9xXcr6NunbwuHah1+zb7yPfj8IehxPGSP9M/XkDahsKyKu95eyjuLNtMuJpzrc7qreEtIUxFvql4nNvuj2wpm0ndITstlaUV+zd7nVNg8Ft6+Fn72BUQ0bQtGEYCv1+Zx65RFbC0q59bje/HTI7uqgEvI08A2cV9UApzxOOSthk/vczuNBKGaWstdby8l3Gt449ox3HRsTy3gIm2C/pVLYOiWAyOvhq+fhK7HQIdBzTiJgfj2oOlDbcbaHSW0T4wiJiKMZy8bQVp8pIq3tCn61y6B4/h7YfV0eOX85p9j/G/hmF+3XCZxxZbCMi569hvCPIZuabF7t+OctmQLhWVVpMRGkLurjD9/+B0Xj+rMXaf1a9Ze4CLBTkVcAkdELFwxDVZ91LzPfzMZVkxVEQ8BCzcUsG5nKUf2SCUmYt+vqRdn/cCcH/YNLD26Vxo/O7qbGxFFAoKKuASWhEwYfnnzPlu6Az69H0p2QFzobTnYlmzI3w3Ak5cMIyEqfO/r/7rqCPJLK8krqaSiupah2Ul4PLp8Im2XBrZJ6Og+wblfO9PVGHL4dhRXkBgdfkABB4gM89IhMZoBHRMZ3rmdCri0eWqJS+joMASi28H3H+wr6AC2xr1M0iy/O60ft57Q2+0YIgFPRVxCh8frFO+lbzi3OqOiMiDpbhh0Pnj1Tz6QWWv5am0e7ROi6JYW53YckYCn32gSWo67FzqNAWud57XVVM/6B/zvevjiYTjmDhh4jlPwJWBU19Ty/tKtTP58LUs2FXLJ6E7cf+ZAt2OJBDwVcQktSdlwxNUHvDS/vC85HXbDjD/AW9fUFfPbof9Z4NGwELe9OmcDT8xczcb8MrqlxvKHSQM5a5g2xBHxhYq4hD5jnKVde50M302FGX+EN6+Ezx+G8XdCn4kq5q0sv7SSdjHhGGNYua2Y9PgofndqP47vm6HBaiJNoCIubYfHA/3OcIr28rdg5oMw5TLIGOgU8y5HHfocEbHqij8Ma3eU8OwX63hzQS4vXj6SsT1SufPkvkSE6Y8okeZQEZe2x+OBAWdDvzNhyRvw2YPw6kW+fbbLUXD5u/7NF6Ienb6KRz75nnCvh7OHdSSrnbPRjQq4SPOpiEvb5fHC4POdgr7iHSja3Pjxqz6E3HnOoDmtz94ki3MLePST7zllQAfuOb0/afGRbkcSCQkq4iLeMBhw1qGP83hh3efOfvKxKf7PFULySivp0z6BP5498EcLuIhI86mIi/gqMcu5L9ygIl7HWovxoVdifO90cnql+XSsiPhORVzEV4nZzv2uHyBzqKtRAkFldS05D82guKKav503hOP6ZbB2RwlvfbuJjIQoOiRGsTKvhjVfruMnYzoT5tW1b5GWpiIu4qukTs7965fDF3+FkVfCwHOdEettUH5pJZsLyxnVNZnsZGeQ2qrtJTwxYzW1dt9xMRErOWVgezokRruUVCR0qYiL+ComGc5+DnaucrY8nfoL+OguGHwBjLgS0vu4nbBVFZdXAXDx6M70bh8PwIn92/P9/Sezs6SSrUXlTJ81jxOPHKkCLuInKuIiTTHwHOc+5w7Y+A3M/QfMfxHmTIbO42DET6Hv6RAW4WpMf1m9vYQVW4oY2z2FovJqAOKjDvw1Eub10D4xivaJURS0D2NgVqIbUUXaBBVxkeYwBjqNdm4nPQjf/gvmveCsBBebBsMuc/ZF39MFHwI25O3m/Ge+Iq+0kpFd2vHApIFMHJxJVpJa2SJuUREXOVyxqXDkzTD2F7DmE5j7HHz5N+e6ea8TndZ592MbX+ktwEdt7yqt5PIX5lBda3n2shEkRofTKyOexy7UAD8RN6mIi7QUjwd6Hu/cCjY63ewL/unsb96YDoPh6hkBu5yrtZbrX15AbkEZL181ipFdkt2OJCJ1VMRF/CEpG469y9ktbeU02PFd/ccV5jpd8d9/4GzSEoCMMdw0oQcFZVUq4CIBRkVcxJ/CIqD/mQ2/X1MNaz+D2Y8HXBH/flsxX6/N47IxXRjbI9XtOCJSD7+uvmCMOckYs9IYs9oYc0c97ycaY6YaYxYZY5YZY67wZx6RgOMNg9HXwobZsGm+22kAKKmo5oH3lnPKo1/wyPRVFNVNJRORwOO3Im6M8QJPACcD/YALjTH9DjrsBmC5tXYwkAP8xRgTmnNzRBoy9FKITICvnnA1hrWW/y3cxLF/mcmzX6zjnOFZTL/lGK11LhLA/NmdfgSw2lq7FsAY8ypwBrB8v2MsEG+cBZXjgHyg2o+ZRAJPVIIzJe2rx2H5O2A8kDUS+k6Evqe1WoxtRRX8+o3F9MqI5+lLhjO0U7tW+9oi0jz+LOIdgY37Pc8FRh10zOPAO8BmIB4431pb68dMIoHpqFshIg5qKqG6HNbMgA9uhw9uZ1h8Twi72FlEJqV7i3/pOevyOaJrMu0To3jj2rH0y0zA6wnsKW8i4jDW2kMf1ZwTG3MucKK19qq655cCR1hrb9rvmHOAccAtQHfgY2CwtbbooHNdA1wDkJaWNnzKlCl+yexvJSUlxMXFuR2jWZS99UXvziVtx9ckb5tF0u61AJTEdmZn6mh2pI2hNLbLYc0vr7WWV7+r5KP11fxiWCRD01v+b/pg/dmDsrtF2es3fvz4+dbaEQe/7s+WeC6Qvd/zLJwW9/6uAB60zl8Sq40x64A+wJz9D7LWTgYmA/Tu3dvm5OT4K7NfzZw5E2VvfcGcHS5x8g/pDt+9R9yKd4hbP4Uu61+Ddl3rutxPh47DnXnqPiqtqOYXry5k+vpt/HRcV35+al+/tL6D+Wev7O5Q9qbxZxGfC/Q0xnQFNgEXABcddMwG4FjgC2NMBtAbWOvHTCLBKSnbGcU++loo2e7MPV8xFb5+Cmb/HeI7QJ/TnKLeeZwz6r0B24rKufKluSzfXMTvz+jPZWO6tN73ISItym9F3FpbbYy5EfgQ8ALPW2uXGWOurXv/aeA+4EVjzBLAALdba3f6K5NISIhLd9ZlH345lBXAqo9gxTvw7b9h7rPgjQBPwyPKU6xlSlUNEbFewj418GkTv36XcXDBK43+oSAircOv/xdaa6cB0w567en9Hm8GTvBnBpGQFp0Eg85zbpW7nbXbN86BRsaHhgHVVbWEhTdjhml5gfPHwuxHncF4IuIq/SktEioiYuqukU+s921rLbNW5zG2ewpRh3P9u6IYZj4IvU9tc3uoiwQav67YJiKBY9bqPC557hveXbLl8E50yl+c6XD/uwFqa1omnIg0i4q4SIiz1vLhsq3c8d/FpMZFcEK/jMM7YVwanPIQbJrn+ipzIm2dutNFQtj6vFJ+/cZivlmXT4/0OP563hCiwltgy9MBZ8PS/8LHdzk7sY3/jXN9XkRalYq4SACx1vLu4i2UVdUwoU86qXGRlFdbdpZUkBgdTrjXt84zay3GGOIiw9hWVM59Zw7gwpHZhPn4+UMyBk5/DD5uB3Mmw7L/wnH3wuALmzRfXUQOj4q4SAB5b8kWbnrlWwD+d8M4UuMi+WZLNdfePx2A+MgwkmLDSYqO4MmLh5GdHMM3a/OYtSaPdjHhJMWEs3ZHKXN/yOc/V40mJS6ST2/NweOPZVRjU+DMJ+CIq2HabfC/62H+i05Xe+aQlv96IvIjKuIiAaKwrIp7py5nQMcEnrxoOOkJkQD0aOfl92f0Z1dpFbt2V1JY5txHRzjd4vM37OKxT1ex/wrKEwdnsruqhrjIMP8U8P1lDoGffgiLXoGP/w+eHQ8jfgrjf+vfrysiKuIibnpx1jpWbS/hgUkD+dMH35FXUsELl4+kU0rM3mM6xnnIaWRVtetzevCzo7tTVFfcw70espNjGjzeLzweGHox9DkVZvzBWXRm2Vt0yLoAao9WF7uIn6iIi7jovSVbyCutpKbWkldSwRXjujKgY2KTz+P1GNrFRtAuNsIPKZsgOglO+TMMuxSm3Ubv75+Ah/7jrCLXVGGRMOYmGHkleFpgMJ5ICFIRF3GJtZZV20s4eUB7vB7DM5eOoLomRHbibT8Qrnif5VN+T7+Y/OadY+cqeP82WPwaTHwU2g9o2YwiIUBFXMQlubvKKNhdRZ/2CXtfa7HR44HAGLZnHEO/5u7qZC0seR0+uBMmHwNjb4Jjbofw6BaNKRLMQug3hkjwKNhdyfnPfEWYxzCme4rbcQKTMc6a8DfOhUEXwJd/gyfHwJoZbicTCRgq4iKtyNYNIU+KieC8kdn89/qx9MqIdzlVgItJdqay/WQqGA/860x461oozXM7mYjr1J0u0kpWby/mtjcW88CZA+mXmcAvj+vldqTg0vVouG42fPGw0ypfMRWikxs+vtNoOPvZ1ssn4gIVcZFWkF9ayVlPzibM6yG/tNLtOMErPAom/A76n+WsFFddUf9xu9bBkilw7P9BUnbrZhRpRSriIk1UVTeC3NclUAGenLGakopqPvzl0fRU9/nhy+gHEx9p+P28NfDYMFg5DUb9rPVyibQyFXGRJqiqqSXnoZlsLixj4qBM/n7hUADue3c5UeEe0uOjSI+PJD0hkuzkGNLjo9hcUMY/v17P2cOyVMBbS0p3SOvjdLmriEsIUxEXaYIvV+9kU0EZ5wzPYmzdqPI9W31uKSynpnbf2qcXjerEHyYNpLSiGoBfHNfTlcxtVp/TnGvnu/OdwXEiIUhFXKQRNbWWr9bksbOkgjOHdmTqws0kRIXxwKQBRIY5q4gZY/jy9gnU1lryd1eyvaiCbcXlpMU5a593SIrmj5MGktWulZdCbev6nOoMglv1EQy+wO00In5xyCJujOkFPAVkWGsHGGMGAadba+/3ezoRF1hrWbChgKmLNvPu4i3sLKmgc0oMpw7qwEfLtzG6W/LeAr4/j8eQGhdJalwk/di3gEtcZBhnD89qzW9BADL6O/eFue7mEPEjX1rizwK3Ac8AWGsXG2P+A6iIS0h68P3veObztUSEeTi2TzqnD85kfJ90wjyGX53Qi2P7ZrgdUXyxZ732Gs0GkNDlSxGPsdbOMeaA7Qyr/ZRHxBUb8nZTVF5F3w4JTBycSa+MeE7on0F8VPgBx10+rqtLCaXJjHEKeUPT0ERCgC9zZHYaY7oDFsAYcw6wxa+pRFrId1uL+HZ7Nd9tLaK4vKrB416es55JT86iuraWAR0TOXt41o8KuAQhb6Ra4hLSfGmJ3wBMBvoYYzYB64BL/JpKpAVYa7njzSUs3FjBowu+ACAxOpwx3VJ4+tLhAExbsgWvxzBnXT6928fXe61bgliYWuIS2g5ZxK21a4HjjDGxgMdaW+z/WCLNV15VQ3WtJS4yjKcuGcbb02eT3bMvubvK2LSrjKSYfS3sP76/go35ZQBcMrqTW5HFX7yRUKMiLqHLl9Hp/3fQcwCstb/3UyaRZivYXcllz88hIyGKyZcOp0NiNH1TvOQMyqz3+HduOJLcXWVsKSxjRBfNJQ45YRFQre50CV2+dKeX7vc4CjgNWOGfOCKH54OlW1mcW8jjF3XjoMGY9WoXG0G72AgGZiW2Qjppdd4ItcQlpPnSnf6X/Z8bYx4G3vFbIhEfFZVXsXhjIau3F+8dNb5gwy4AjtM0MIG67vSGBzSKBLvmrNgWA3Rr6SAivvhmbR5vLshl4cYCVm0vwVpnJtGkoVnUWMuUec7CHlHhGqAmaGCbhDxfrokvoW56GeAF0gBdDxe/qq21zFi5nXnrd7FwQwEPTBpAt7Q41u0s5aPl2xiancSpAzMZ2imJwVlJJNYNVvvDpIH8kFd6iLNLm6EpZhLifGmJn7bf42pgm7VWi72IX81YuZ0rX5pHmMfQPzOBwjKnS/Ts4VmcPzK7wevdF43SCHPZT1gEVO52O4WI3zRYxI0xe4bqHjylLMEYg7U233+xpK0rrawBYOpNR9K3w751yJuyh7eI0xLf5XYKEb9prCU+H6cbvb4mj0XXxcWPkqLDCfcawr2HHmEu0iBNMZMQ12ARt9ZqkWhpFUtyC0mNj6BDYvTe147qmcoLlx9B+/1eE2kybyRUqTtdQpdPo9ONMe2AnjjzxAGw1n7ur1DSdpRV1jDx8S8BGNs9hTOHdCQ1PoIJfTI4smeqy+kk6HUYBEvfgM3fQuZQt9OItLhDXmA0xlwFfA58CNxbd3+Pf2NJWxHmNfzu1L6cPjiT3F1l/PrNxVz7rwVsKSxzO5qEguFXQGQifPFXt5OI+IUvLfFfACOBr621440xfXCKuchhC/MYrjrKGV5hrWXBhl0YYw7oWhdptqgEGHUNfP4w7FgJab3dTiTSonwZ6lturS0HMMZEWmu/A/R/ghy2rYXlnP3UbL7bWgQ46/IP75zMsE7tXE4mIWXUdRAeDV/+ze0kIi3Ol5Z4rjEmCXgb+NgYswvY7N9YEsp2llSwOLeA+99bwfaiCsqrat2OJKEsNsXpVv/maSjMPfC9QefBsMvcySXSAhqbJ/4r4DVr7aS6l+4xxswAEoEPWiOcBL/SimpWby9hcHYSADe/tpC3vt0EQFxkGM9fPpIhde+J+M24X8DO7w8cqV64ET74DfQ5DWK0g50Ep8Za4h2B2caYdcArwOvW2s9aJ5YEq/V5pXy+aieLNxawKLeA1dtLAFhyz4nERoYxoU86fTvEMzgriQEdE4mNbM7y/SJNFJ8Bl7xx4GvblsNTY+Grx+HY/6v/cyIBrrF54jcbY24BjgYuAO4yxizCKehvWWsPXslNhE9WbOf37y4nJTaCQVmJnDKwA4OzkwirW7Rl4uD69/UWaXUZ/aD/mSn0/a4AACAASURBVPDNMzDmRrXGJSg12gyy1lrgM+AzY8yNwHHAg8DTOLuZiRzgzKEdOb5fBlnton3az1vEVcfcDsveVmtcgpavi70MxGmNnw/kAb/xZygJPpXVtTw5czV92idw0oD2bscR8U16X+g/Cb5+ylkQpvuxMOYGZ39bkSDQ4BQzY0xPY8xdxpjlwH+A3cAJ1tpR1tpHWi2hBLzyqhquf3k+j0xfRXF5ldtxRJpmwu8gcxgUb4WPfgv/u0HrrUvQaKwl/iHO9e/zrbVLWimPBJntReXcMmURX67eyX1n9OfcEdluRxJpmpTucMV7YC189meY+Qco2oQ38xq3k4kcUmMD27RLmTRq6aZCzn36K6pqavnzOYM4TwVcgpkxkHM7JGXDOzcxdNs6GDkEEju6nUykQdqcWZqkYHclyzc7K6z17ZDAJaM7Mf2WY1TAJXQMuQgufoOo8h3wj2NhqzoiJXCpiEujrLWs2FLEu2sqOeep2Qy772Mu/sfXlFRU4/UYfntqP7qkxrodU6RldR/Pt0P/CMYDz58EWxa7nUikXlppQ36ksroWr8fg9Rge/mglT8xYA8CAjjXcOL4H4/ukExvhdTmliH+VxnWBn34AjwyE1dOdbU1FAkxjy64uAWx9b+FMIde/6BBUXF7FFS/M5c5T+jC8czInD+hAp+QYIvJWM+mko9yOJ9K64jLqHtT3q1DEfY21xE9rtRQSEIrKq/jJ83NYkltIWaWzKcmAjokM6JjIzJlrXU4n4oa6+eJWm/RIYGpsdPr61gwi7lq2uZBfvrqQH/JKefyiYRzZM9XtSCLuM3XDhtQQlwDV2GIvxcaYonpuxcaYIl9Obow5yRiz0hiz2hhzRwPH5BhjFhpjlhljtMGKC5ZuKuTMJ2ZRWFbFC5cfoRXXRPYwaolLYGusJR5/OCc2xniBJ4DjgVxgrjHmHWvt8v2OSQKeBE6y1m4wxqQfzteU5umfmcAvj+vFRUd0ol1shNtxRAKH8UBEHJTlu51EpF4+j06vK7BRe55bazcc4iNHAKuttWvrPv8qcAawfL9jLgL+u+dc1trtvuaRw5dXUkFpRQ2dUmK4YXwPt+OIBB5jILkb5K1xO4lIvYyzUVkjBxhzOvAXIBPYDnQGVlhr+x/ic+fgtLCvqnt+KTDKWnvjfsc8AoQD/YF44FFr7T/rOdc1wDUAaWlpw6dMmeLzNxhISkpKiIuLczvGXq+vrOSDH6r4W04MCZGNb/gQaNmbIpizQ3DnD4Xs/Zb9mfjiNXwz+hm3I/ksFH7uwcif2cePHz/fWjviR29Yaxu9AYuAFODbuufjgck+fO5c4B/7Pb8UeOygYx4HvgZigVRgFdCrsfP26tXLBqsZM2a4HWGvkvIqO/DuD+y1/5rn0/GBlL2pgjm7tcGdPySyf3Kftfe0s7a60tU8TRESP/cg5M/swDxbT030ZcW2KmttHuAxxnistTOAIT58LhfYfy3OLGBzPcd8YK0ttdbuBD4HBvtwbjlMr8/bSFF5NVcdpSXyRRqV3B1sDeStdjuJyI/4UsQLjDFxOAX2ZWPMo0C1D5+bC/Q0xnQ1xkTg7Ef+zkHH/A84yhgTZoyJAUYBK3yPL81RU2t5ftYPDOuUxPDO7dyOIxLYOo+BsGh4+3qoLHU7jcgBfCniZ+DsJX4z8AGwBph4qA9Za6uBG3G2NF0BTLHWLjPGXGuMubbumBV151wMzMHpfl/anG9EfDdnXT4b8nerFS7ii3Zd4NwXYMtCeOOnUONLG0akdfgyOj0d2GKtLQdeMsZEAxlA3qE+aK2dBkw76LWnD3r+EPCQz4nlsI3pnsLK+08izKP9b0R80vtkOPUv8O7NMO1WOO2RfXPIRVzky2/x14H9VzqoqXtNglhkmBevR7+ERHw24qdw1K0w/0X44mG304gAvhXxMGtt5Z4ndY+1IkiQyi+t5PTHv+SrNYfsSBGRg024CwZdAJ/eD/NecDuNiE9FfEfdXHEAjDFnADv9F0n8ZcZ325n42Jcs21xEu9hwt+OIBB9j4PTHoMdx8O4v4eO7obbG7VTShvlyTfxanFHpT+BsA5ALXObXVNKiVm0r5tFPVvHu4i30SI/jtWtG06d9gtuxRIJTWARc+Cq8/2uY9Qjs/B7OmgyRh7VStUizHLKIW2vXAKPrppkZa22x/2PJ4SqvcloHUeFevlqbx8fLt3Hzcb24NqcbkWFel9OJBDlvOJz6V0jrCx/cDs+dCBe9Ckmd3E4mbcwhi7gxJgP4A5BprT3ZGNMPGGOtfc7v6cRnxeVVLNtcxNJNhSzOLeSLVTv41Ym9uXhUZ84ZnsXEQZna3ESkJRkDo66BlO7w+hXw7AQ4/2XoNMrtZNKG+NKd/iLwAvDbuuffA68BKuIuKamoZummQsI8hhFdkimrrGHI7z+mptZZB79DYhTjeqTSt4PTZR4TEUaM6reIf/Q4Fq6aDq+cDy+dBic9CMOvAE3hlFbgSxFPtdZOMcbcCc4iLsYYjeRwyYa83Rz/t8+oqK7lmF5pvPTTI4iO8HL3xH5kJ8cwsGMiqXGRbscUaVvSesFVnziLwbx3Cyx6xelu7zDI7WQS4nwp4qXGmBScQW0YY0YDhX5NJQ1an19KRXUtd0/sx8TBmXtfv2xMF/dCiQjEJMOlb8GiV+Gj38HkY+CIa2D8byAq0e10EqJ86e+5BWfN8+7GmFnAP4Gb/JpKGrRqWwkAOb3T1eIWCTTGwJAL4aZ5zuIw3zwDj4+EJW/AIbZ9FmmOQxZxa+0C4BhgLPAz9u39LS6Ysy6fwVmJdE2NdTuKiDQkup2zTOvVn0JCR3jzSnhpIuxY6XYyCTENdqcbY7zAeUBH4P26zUtOAyYD0cDQ1onYdhXsruSTFdv5cNlWhnZqx3U53XnqkmHs2l3ldjQR8UXHYc6gt/kvwif3wlPjYOyNcPRtEKE/xOXwNXZN/Dmc/cDnAI8ZY9YDo4E7rbVvt0a4tmz/AWztE6L2bhlqjCFZU8VEgofHCyOvhL6nw/S74cu/Od3rJz0IfU7VRipyWBor4iOAQdbaWmNMFM5Sqz2stVtbJ1rbNm99PhXVtfzjshFM6JOOR5uViAS3uDQ480kYeqkzgv21i6HniXDynyC5q9vpJEg1dk280lpbC1C3Den3KuCtZ/X2EuKjwsjpnaYCLhJKOo+Bn30OJzwA62fBk6Phsz9DdYXbySQINVbE+xhjFtfdluz3fIkxZnFrBWwryiprmPdDPv/4Yi1VNbXcdmJvpv38KMK8WjBCJOR4w51r4zfOdfYqn/EAPDkGVn/idjIJMo11p/dttRRt1PLNRfz7m/Us3FDAym3Fe1dcG9s9lX6ZCWQnx7icUET8KiETzn3R6WKfdhv8+yzoPwkm/h2itEmRHFqDRdxau741g7RF+aWVTF20mSHZSVzXpztDspMYnJ1EWrzmf4u0KT2Oheu/gll/h88ehJLtcMmbEB7tdjIJcL6s2CYtpLrW8u2GXSzYUIC1livGdWXR/52ga94iAmGRcMxtziC3N6+C1y+H8//tdL2LNEBFvBW8NPsHpi7azMKNu6n+aDYAQ7KTuOqobi4nE5GAM/AcKC90RrC/fR1MmqzNVKRBKuKtYHNBGbXWcmynMM4YN4hhndrRPjHK7VgiEqhGXgnlBfDJ7yEqCU55SPPJpV6Nrdi2hLpNT+pjrdX2PI2w1vLq3I2cMqADd57ijBGcOXMmOQM7uJxMRILCkbdA2S6Y/RhExsGxd6uQy4801hI/re7+hrr7f9XdXwzs9luiEFBeVcOtry/ivcVb2LW7kutzergdSUSCjTFw/H1QUeys8lawAU5/HCI0a0X2OeTodGPMOGvtuP3euqNuN7Pf+ztcsHruy3W8t3gLt5/Uh58dreveItJMxsBpj0BSZ6drfecquOA/kJTtdjIJEL6Mlog1xhy554kxZiyglfsb8a+v1nNUz1Suy+mukecicniMgaNugYteg10/wOQcWD/b7VQSIHwp4lcCTxhjfjDG/AA8CfzUr6mCXH5pJQM6JrodQ0RCSa8T4apPIDrJ2dZ03vNuJ5IAcMjR6dba+cBgY0wCYKy1hf6PFdx+e2pf+mdqtSURaWFpvZxC/uZV8O7NsG25Rq63cYcs4saYSOBsoAsQZur+sVhrdU28HlU1tfxkbBe3Y4hIqIpOcrrWP7gD5kyGIRdCx+FupxKX+NKd/j/gDKAaKN3vJgf5ak0eZzw+i8LdVW5HEZFQ5vFCzp1gvPDdNLfTiIt8Wewly1p7kt+ThICpizfz/bZiwrzq2hIRP4tJhk5jYOU0OPYut9OIS3xpic82xgz0e5IgVl5Vw82vLeQ/32zgpAHtiY3UQngi0gr6nALbl0P+OreTiEt8qTZHApcbY9YBFYABrFZs22fqos289e0mfj6hB784rpfbcUSkreh9Cnz4G5h+D2QO2fty9oa18OW3DX8uvT/0OsH/+cTvfCniJ/s9RZAL8xq6pcVy5ZHd8GpeuIi0luSuTpf68redW53uAGsP8dmRV8OJf4CwCH8mFD/zZYrZnpXb0gHt2lGPSUOzmDQ0y+0YItIWXT4NaioOeOnzzz/n6KOPrv/42hqY+Uf46nHYuhjOfQkStKdDsDrkNXFjzOnGmFXAOuAz4AfgfT/nChqbC8r474Jct2OISFvl8UB49AG3Wm/kj17be4uMgxMfgHOeh61LYfIxWgEuiPnSnX4fMBqYbq0daowZD1zo31iB7835ubw2byNz1uVjDBzZI5X0BHVUiEiQGHA2pPeDVy92VoA74QEY9TMtHBNkfBmdXmWtzQM8xhiPtXYGMORQHwpl24vLufX1RWwrKufW43sx49YcFXARCT7pfeGaGdDzBPjgdvjvNVBT7XYqaQJfWuIFxpg44HPgZWPMdpyFX9qsrYXlAPz2lL6c0L+9y2lERA5DVCKc/zLMeAC+eBgGnw89jnM7lfjIl5b4GTj7h98MfACsASb6M1QgWrOjhO+2FpFfWklWuxgevWAIg7OT3I4lInL4PB4Yfb3zeMdKd7NIk/gyOn3PEqu1wEv+jRO4fvX6Ir7dUADA3N8exxlDOrqcSESkBcWmQEyKiniQ8aUlLsBdp/Xb+/jpz9a4mERExE9Se8HO791OIU2gIu6D1+ZuIKtdNC9cMZKJgzO5aFQntyOJiLQ8FfGgo0W+G1FZXctzX67jTx98B8APD57K+N7pLqcSEfGTtN6w4CUozXO61yXg+bLYyzhjzMfGmO+NMWuNMeuMMYda0C8k5JdW8uSM1QAc1TPV5TQiIn6WWrf3w05dFw8WvnSnPwf8FWcjlJHAiLr7kNc+MYo3rx8LwImaSiYioW5vEVeXerDwpTu90Frb5pZZXbOjhN0VNfTPTGD1AycT5tXwAREJcYnZEBYNO1TEg4UvlWmGMeYhY8wYY8ywPTe/J3PZ81+u46J/fI0xqICLSNvg8UBqD3WnBxFfWuKj6u5H7PeaBSa0fBz3VVTX8NaCTbyzcDNH9UrFaB1hEWlLMofBoldh41zIbhNXToOaL4u9jG+NIIGgorqGY//yGbm7yhjQMYFfn9jH7UgiIq3ruHtg3Wfw6kXOuuqJ2mY5kPkyOj3RGPNXY8y8uttfjDGJrRGutRWVVVNUVsWtx/di6o1H0iU11u1IIiKtKyYZLnwVqsrglQuhsvTQnxHX+HKx93mgGDiv7lYEvODPUG5Ji49k8T0nctOxPdWNLiJtV3pfOOc52LoE3r4OamvdTiQN8KWId7fW3m2tXVt3uxfo5u9gIiLiol4nwgn3wfL/wcw/up1GGuBLES8zxhy554kxZhxQ5svJjTEnGWNWGmNWG2PuaOS4kcaYGmPMOb6c1182F5Rxw8sLWLBhl5sxREQCw5gbYcgl8PmfYeovobrC7URyEF9Gp18HvFR3HdwA+cDlh/qQMcYLPAEcD+QCc40x71hrl9dz3J+AD5sWveUV7K7ivSVbmDi4g9tRRETcZwxMfBRiU2HWI073+nn/hETt4hgoDtkSt9YutNYOBgYBA621Q621i3w49xHA6rou+ErgVZy9yQ92E/AmsL0Juf2iuu66T5hH88JFRADwhsHx98J5/4Id38EzR8O6L9xOJXWMtbb+N4y5xFr7b2PMLfW9b639a6MndrrGT7LWXlX3/FJglLX2xv2O6Qj8B2fO+XPAu9baN+o51zXANQBpaWnDp0yZ4sv31mSrd9Vw/zfl3Do8koFpLb83TElJCXFxcS1+3tag7O4J5vzK7g5/ZY8p3Uj/ZQ8Ss3sza7pfTm7W6U5rvQXp516/8ePHz7fWjjj49cYq1Z75VfHN/Jr1/Zc9+C+GR4DbrbU1jY0Gt9ZOBiYD9O7d2+bk5DQzUuOi1ubBN18zfOgQxvZo+Q1PZs6cib+y+5uyuyeY8yu7O/ya/bgz4e3r6LHieXpEFcAZT0BETIudXj/3pmmwiFtrn6m7v7eZ584Fsvd7ngVsPuiYEcCrdQU8FTjFGFNtrX27mV+z2Qp3V2GADolRREd4W/vLi4gEh8h4p2t91qMw/W5n05Txd7qdqs3yZbGXPxtjEowx4caYT4wxO40xl/hw7rlAT2NMV2NMBHAB8M7+B1hru1pru1hruwBvANe3dgF/f8kWzn16NkPu+4gvV+/kqzuPZWindq0ZQUQkuBgDR/4Sep4I857XqHUX+TKC6wRrbRFwGk7ruhdw26E+ZK2tBm7EGXW+AphirV1mjLnWGHPtYWRuUde9vIC5P+wiMzGaXbsr3Y4jIhI8Rv0MSrfDslbvPJU6vozeCq+7PwV4xVqb7+tqZtbaacC0g157uoFjL/fppC3s1IEdGN0tmdHdUli9vcSNCCIiwanbeEjpCXOegcHnu52mTfKliE81xnyHs8DL9caYNKDcv7FazxMX79tVtWdGc8fwiYi0QR6P0xqf9ivInQdZPxo8LX7myzzxO4AxwAhrbRVQSv3zvUVEpK0ZfAFEJsA3z7idpE1qsIgbYybU3Z8FjAfOqHt8EjC2deKJiEhAi4yHIRfDsregeJvbadqcxlrix9TdT6zndpqfc4mISLA44mqorYb5IbnBZUBrbJ743XX3V7ReHBERCTop3aHn8TD3OTjyFgiLcDtRm+HLPPE/GGOS9nvezhhzv39jiYhIUNkz3Wy5ppu1Jl/miZ9srS3Y88RauwtnupmIiIij2wRI7gaLXnU7SZviSxH3GmMi9zwxxkQDkY0cLyIibY3HA12Ogk3zoYGNtaTl+VLE/w18Yoy50hjzU+Bj4CX/xhIRkaDTcRiUF0D+WreTtBmHXOzFWvtnY8xi4Dicncnus9Z+6PdkIiISXDLrFs/a/K0z2E38ztdNs1cA1dba6caYGGNMvLW22J/BREQkyKT3hbAo2LQABp7jdpo2wZfR6Vfj7DC2ZzmejoCGH4qIyIG84U5rfMVUqNBeFK3Bl2viNwDjgCIAa+0qIN2foUREJEgdexcUboTp97idpE3wpYhXWGv37tFpjAkDNPRQRER+rPNYGH0dzH0W1n7mdpqQ50sR/8wY8xsg2hhzPPA6MNW/sUREJGhNuAuSu8P/boQKDZ/yJ1+K+O3ADmAJ8DOc/cF/589QIiISxCJi4MynoCgXPlK58KdGR6cbYzzAYmvtAODZ1okkIiJBr9MoGHMjzP47FG+FEx6A1B5upwo5jbbErbW1wCJjTKdWyiMiIqHi2Lvh+Pvgh1nw5Gj48LdQXuh2qpDiS3d6B2CZMeYTY8w7e27+DiYiIkHOGwbjfg4/XwBDLoSvnoC/D4P5L0JtjdvpQoIvi73c6/cUIiISuuLS4fTHYMSV8MEdMPUXMPcfcNKfoMs4t9MFtQZb4saYKGPML4FzgT7ALGvtZ3turZZQRERCQ+YQuOJ9OOcF2L0LXjwFpvwEdq13O1nQaqw7/SVgBM6o9JOBv7RKIhERCV3GwICz4Ma5kPMb+P5DeHwEvH87lGx3O13Qaaw7vZ+1diCAMeY5YE7rRBIRkZAXEQM5t8PQS+CzP8GcZ2HBP+na4RQYNRii27mdMCg01hKv2vPAWlvdCllERKStSewIp//daZn3PoXOG96ARwfD5w9r/XUfNFbEBxtjiupuxcCgPY+NMUWtFVBERNqAlO5wznPMHfEodB4Hn94Hfx8CXz8N1RVupwtYDRZxa63XWptQd4u31obt9zihNUOKiEjbUBrXBS58Ba6cDml94IPbnWlpC/6paWn18GWeuIiISOvKHgmXvwuX/Q/iM+Cdm2DGA26nCjgq4iIiEri65cBVn8DA82D2Y7DrB5cDBRYVcRERCWzGwHH3gCcMPr7b7TQBRUVcREQCX2JHGPdLWP42rJ/tdpqAoSIuIiLBYexNkNDRWbq1ttbtNAFBRVxERIJDRIzTrb5lEax8z+00AUFFXEREgkf/syAmFZa84XaSgKAiLiIiwcMbBv3PdNZc14puKuIiIhJk+p8F1WXw/QduJ3GdiriIiASXTmMgvgMs/a/bSVynIi4iIsHF44H+k2DVR84gtzZMRVxERILPkTdDXDq8chGU7HA7jWtUxEVEJPjEpcMFL8PunTDlUqiudDuRK1TERUQkOGUOhTOegA1fwfu3gbVuJ2p1YW4HEBERabaB58C2ZfDlXyFjABxxtduJWpWKuIiIBLcJd8H25fD+7VC8FY75NYRFup2qVag7XUREgpvHA2c/B4POhy8ehsk5sHmh26lahYq4iIgEv8g4mPQUXDQFdufDsxPg0/tDfsCbiriIiISOXifCDV/DoPPg84dCvlWuIi4iIqEluh1MehoufA1259W1yh+Aku1QunPfLQRa6RrYJiIioan3SdDpa3j/Dvj8z85tfxkD4bov3cnWQlTERUQkdEW3g7OegaEXw46V+15f+T6s+9yZW26Me/kOk4q4iIiEvq5HO7c9qspgzSdQUQxRCe7lOky6Ji4iIm1PTIpzvzvP3RyHSUVcRETanj1FvCzf3RyHSUVcRETantg0537bcndzHCa/FnFjzEnGmJXGmNXGmDvqef9iY8ziuttsY8xgf+YREREBoMNg5zb9Hme6WZDyWxE3xniBJ4CTgX7AhcaYfgcdtg44xlo7CLgPmOyvPCIiInt5w+DMp6C8EKb9yu00zebPlvgRwGpr7VprbSXwKnDG/gdYa2dba3fVPf0ayPJjHhERkX0y+kPO7bDsLecWhPxZxDsCG/d7nlv3WkOuBN73Yx4REZEDjbsZOgyB926Fkh1up2kyY/20ibox5lzgRGvtVXXPLwWOsNbeVM+x44EngSOttT8a72+MuQa4BiAtLW34lClT/JLZ30pKSoiLi3M7RrMou3uCOb+yu0PZmya2ZD3D59/C1vbH8n3v65t9Hn9mHz9+/Hxr7YgfvWGt9csNGAN8uN/zO4E76zluELAG6OXLeXv16mWD1YwZM9yO0GzK7p5gzq/s7lD2ZphyubUP9bK2trbZp/BndmCeracm+rM7fS7Q0xjT1RgTAVwAvLP/AcaYTsB/gUuttd/7MYuIiEjDuk+Akq2wfYXbSZrEb8uuWmurjTE3Ah8CXuB5a+0yY8y1de8/DfwfkAI8aZy1a6ttfd0FIiIi/tR9vHO/5lPIOHgiVeDy69rp1tppwLSDXnt6v8dXAVf5M4OIiMghJWZBai9YOwPG3uh2Gp9pxTYRERFwutR/mAXVFW4n8ZmKuIiICEDnsVBdBluXup3EZyriIiIiAFkjnfvcOe7maAIVcREREYCETEjIgty5bifxmYq4iIjIHlkjYKOKuIiISPDJGgmFG2DeC+CnFU1bkoq4iIjIHkMvhi5Hwbu/hH+fDYW5bidqlIq4iIjIHtHt4LJ34JSHYcNX8OQYWPDPgG2Vq4iLiIjsz+OBI66G62ZD+0Hwzk3w8jlQuMntZD+iIi4iIlKf5K7wk6lw8kOwfjY8ORoW/CugWuUq4iIiIg3xeGDUNXDdrLpW+Y3w+cNup9pLRVxERORQkrs5rfK0vrBpnttp9lIRFxER8YXHA7GpUF7kdpK9VMRFRER8FZkAFSriIiIiwScyXi1xERGRoBSVABWFbqfYS0VcRETEV94IqKlyO8VeKuIiIiK+8nihtsbtFHupiIuIiPjKEwa11W6n2EtFXERExFfGC1YtcRERkeDj8Tr3AdKlriIuIiLiq8gE5748MEaoq4iLiIj4Ki7duS/d4W6OOmFuB2gJVVVV5ObmUl5e7naURiUmJrJixQq3Y/xIVFQUWVlZhIeHux1FRCSwxaY59yXbIa23u1kIkSKem5tLfHw8Xbp0wRjjdpwGFRcXEx8f73aMA1hrycvLIzc3l65du7odR0QksO0p4gHSEg+J7vTy8nJSUlICuoAHKmMMKSkpAd+LISISEAKsOz0kijigAn4Y9LMTEfFReLRzXx0YDZ+QKeKhaN68efz85z9v8P3NmzdzzjnntGIiEREJJCFxTTxY1NQ0bV7hiBEjGDFiRIPvZ2Zm8sYbbxxuLBERCVJqif9/e3cfXVV15nH8+xAMoWF4m1BFREItJUHICxBITIpBBaEqoA2Ndni11hcWSIdlBcc1Y0aXM8qgdaFYakETqyvCoFhFrRQVmFAQVBBQk4gSEFCpiQiBLin4zB9nB28uAQPk5t6dPp+1sjh3n3Pu/d2zSHb2vif7aSJVVVWkpKQwceJE0tLSKCgo4NChQyQnJ3P33XeTl5fH0qVLWb58OTk5OfTv35+xY8dSW1sLwIYNG7joootIT09n0KBBHDhwgJUrV3LllVcCsGrVKjIyMsjIyCAzM5MDBw5QVVVF3759geC+gMmTJ9OvXz8yMzN54403ACguLuaaa65hxIgR9OrVi9tvvz06F8gYY0yTa5Ej8cLfrT2u7cq0rozPSeZvh48y6Yn1x+0vGHAeYwd2p+bgYW556u16+xbdlNOo162oqGDhwoXk5uZy/fXX8+ijjwLBn3CVlZVRVVXFQ1RsaAAAD1BJREFUhAkTWLFiBYmJidx///08+OCDzJo1i8LCQhYtWkRWVhb79++nbdu29Z57zpw5zJs3j9zcXGpra0lISKi3f968eQBs2bKF8vJyhg8fTmVlJQCbNm1i48aNtGnTht69ezNt2jS6d+/eqPdkjDGmAarRTgDYSLxJde/endzcXADGjRtHWVkZAIWFhQCsX7+e999/n9zcXDIyMigpKWHHjh1UVFTQtWtXsrKyAGjfvj2tW9f//So3N5cZM2Ywd+5c9u3bd9z+srIyxo8fD0BKSgo9evQ41olfeumldOjQgYSEBPr06cOOHTsidxGMMaYla90WWifEzN3pLXIkfrKRc9v4uJPu75wY3+iRd7jwu7zrHicmJh5rGzZsGKWlpfWO27x583feIT5r1iyuuOIKXn75ZbKzs1mxYkW90bie5LfCNm3aHNuOi4vjyJHYqcBjjDFeadUKOvaAL6uinQSwkXiT2rlzJ2vXBlP5paWl5OXl1duflZXFmjVr2LZtGwCHDh2isrKSlJQU9uzZw4YNG4BgUZjwjvajjz6iX79+zJw5k4EDB1JeXl5v/5AhQ3j66acBqKysZOfOnfTuHf3VhIwxpsXp3NM68ZYoNTWVkpIS0tLSqKmp4ZZbbqm3PykpieLiYq677jrS0tLIzs6mvLyc+Ph4Fi1axLRp00hPT2fYsGHHLb7y0EMP0bdvX9LT02nbti0jR46st3/KlCkcPXqUfv36UVhYSHFxcb0RuDHGmCbSqSfUbI+Jz8Vb5HR6tLRq1Yr58+fXa6uqqqr3+JJLLjk24g6VlZXFunXr6rXl5+eTn58PwMMPP3zcOcnJyWzduhUIbp4rLi4+7phJkyYxadKkY4+XLVvWiHdijDHmhDr3hL8fDD4Xr1vBLUpsJG6MMcacik6uzkTN9ujmwDrxJhM6KjbGGNOCdUoO/v3SOnFjjDHGL516ABITN7dZJ26MMcacitZtoH03m043xhhjvNS5J+x5B76ujWoM68SNMcaYU9V/AlRvg4XDoObjqMWwTjyGFRcXM3XqVACKioqYM2dOlBMZY4wBIO1nMO5Z2L8HHhsK216LSgzrxCNAVfnmm2+iHcMYY0wkXXAJ3Lgy+Hz86QK673yu2ReAsU68iVRVVZGamsqUKVPo378/99xzD1lZWaSlpXHXXXcdO+7JJ58kLS2N9PT0YwVLXnzxRQYPHkxmZiaXXXYZn3/+ebTehjHGmFPRuSfc8GdIHcUFH5fAs7+Aw4ea7eVb3optr8yCz7Y07XOe0w9G3vedh1VUVPDEE08wZswYlixZwvr161FVRo0axerVq0lISODee+9lzZo1JCUlUVNTA0BeXh7r1q1DRFiwYAGzZ8/mgQceaNr3YIwxJjLiE2FsMR/94VYu2PoH+KISfr4Y2p8b8ZdueZ14FPXo0YPs7Gxuu+02li9fTmZmJgC1tbV8+OGH1NTUUFBQQFJSEgCdO3cGYNeuXRQWFvLpp59y+PBhevbsGbX3YIwx5jSI8Mn5P+WCnFFQ9hto075ZXrbldeKNGDFHSl3JUVXljjvu4Kabbqq3f/bs2Q2WHJ02bRozZsxg1KhRrFy5kqKiouaIa4wxpqn1GgY/vAy+o7x0U7HPxCPg8ssv5/HHH6e2Nvj7wd27d7N3717y8/NZvHgx1dXVAMem07/66iu6desGQElJSXRCG2OMaRrN1IFDSxyJx4Dhw4fzwQcfkJOTA0C7du146qmnSE1N5c477+Tiiy8mLi6OzMxMiouLKSoqYuzYsXTr1o3s7Gy2b4/+KkDGGGNin3XiTSS8AMr06dOZPn16vWMOHDjAxIkTmThxYr320aNHM3r06OOeM7SMqE2xG2OMCWfT6cYYY4ynrBM3xhhjPBXRTlxERohIhYhsE5FZDewXEZnr9m8Wkf6RzGOMMca0JBHrxEUkDpgHjAT6ANeJSJ+ww0YCvdzXjcBvT/f1tJmXumtJ7NoZY4yfIjkSHwRsU9WPVfUw8AwQfvfWaOBJDawDOopI11N9oYSEBKqrq60zOg2qSnV1NQkJCdGOYowx5hRJpDo+ESkARqjqDe7xeGCwqk4NOWYZcJ+qlrnHrwEzVfWtsOe6kWCkTpcuXQYsXrw4/LVITEwkLi4uIu+lqahqg4u9RNvRo0c5ePDgSX8Jqq2tpV27ds2Yqun4nB38zm/Zo8OyR0cksw8dOvRtVR0Y3h7JPzFrqLcK7yUacwyq+hjwGEDv3r01Pz//jMNFw8qVK7Hszc/n7OB3fsseHZY9OqKRPZLT6buA7iGPzwP2nMYxxhhjjGlAJDvxDUAvEekpIvHAtcALYce8AExwd6lnA1+p6qcRzGSMMca0GBGbTlfVIyIyFXgViAMeV9X3RORmt38+8DLwE2AbcAiYHKk8xhhjTEsTsRvbIkVEDgAV0c5xmpKAL6Id4jRZ9ujxOb9ljw7LHh2RzN5DVbuEN/q4dnpFQ3fo+UBE3rLszc/n7OB3fsseHZY9OqKR3ZZdNcYYYzxlnbgxxhjjKR878ceiHeAMWPbo8Dk7+J3fskeHZY+OZs/u3Y1txhhjjAn4OBI3xhhjDDHWiZ9J6VIR6SgiS0SkXEQ+EJGcGMueIiJrReRrEbktbF+sZ/8Xd703i8hfRCTdo+yjXe5NIvKWiOT5kj3kuCwROerqEdS1xXR2EckXka/cdd8kIv/hS3Z3TL7L/Z6IrAppj+nsIvLrkGu+1f2/6exJ9g4i8qKIvOuu++SQfbGevZOILHU/a9aLSN9my66qMfFFsCDMR8APgHjgXaBP2DE/AV4hWHM9G3gzZF8JcIPbjgc6xlj27wNZwL3AbWH7Yj37RUAntz3Ss+vejm8/NkoDyn3JHnLc6wQLIxX4kh3IB5ad4PxYz94ReB843z3+vi/Zw46/Cnjdl+zAvwH3u+0uQA0Q70n2/wHuctspwGvNdd1jaSR+2qVLRaQ9MARYCKCqh1V1XyxlV9W9qroB+HtouyfZ/6KqX7qH6wjWuPcle6267x4gEVdgx4fszjTgWWBvXYNH2Y/jSfafA8+p6k6XcS94kz3UdUApeJNdgX8SESH45bsGOOJJ9j7Aay5fOZAsImc3R/ZY6sS7AZ+EPN7l2hpzzA+AvwJPiMhGEVkgIomRDNvIXI3hW/ZfEMyGgCfZReRqESkHXgKud80xn11EugFXA/PDzo357E6Omxp9RUQudG0+ZP8R0ElEVorI2yIywbX7kB0AEfkeMILgF0DwI/sjQCpBEawtwHRV/QY/sr8LXAMgIoOAHgSDnYhnj6VO/ExKl7YG+gO/VdVM4CBwws8YI6BRJVVPwJvsIjKUoBOf6Zq8yK6qS1U1BRgD3OOafcj+EDBTVY+GtfuQ/R2CZSLTgYeB5127D9lbAwOAK4DLgX8XkR/hR/Y6VwFrVLXGPfYh++XAJuBcIAN4xI1kfch+H8EvfpsIZs82Akdohuyx1ImfSenSXcAuVX3TtS8huHDN5UxKqnqRXUTSgAXAaFWtDjk35rPXUdXVwAUikoQf2QcCz4hIFVAAPCoiY/Agu6ruV9Vat/0ycJZH130X8CdVPaiqXwCrgXT8yF7nWtxUesi5sZ59MsHHGKqq24DtBJ8vx3x29/99sqpmABMIPtPfTjNkj6VO/LRLl6rqZ8AnItLbHXcpwY0pzaUx2RvkQ3YROR94DhivqpV17Z5k/6H7jA0J/pohHqj2Ibuq9lTVZFVNJvjmn6Kqz/uQXUTOCbnugwh+1nhx3YE/Aj8WkdZuWnow8IEn2RGRDsDFBO8D8ON7FdjpciEiZwO9gY99yO7uQI93D28AVruOPfLZT/eOuEh8Edx9XklwJ+Cdru1m4Ga3LcA8t38LMDDk3AzgLWAzwdRdpxjLfg7Bb2X7gX1uu70n2RcAXxJMdW0C3vLous8E3nO51wJ5vmQPO7aY+nenx3R2YKq77u8S3Ax5kS/Z3eNfE/yw3Qr8yrPsk4BnGjg3prMTTKMvJ/jZvhUY51H2HOBDoJxgwNOpubLbim3GGGOMp2JpOt0YY4wxp8A6cWOMMcZT1okbY4wxnrJO3BhjjPGUdeLGGGOMp6wTNyZGiMg/y7cVqD4Tkd1ue5+INPnfxYpIkYRV1GvEObUnaC+WkCprrm2SiJSGtSWJyF9FpM0JnmeSiDxyKpmM+UdmnbgxMUJVq1U1Q4NVn+YDv3HbGcA333W+iLSOdMZT9BwwzC2YUqcAeEFVv45SJmNaFOvEjfFDnIj8XoI6y8tFpC2AK9LxXxLUvJ4uIgNEZJUr3PGqiHR1x90qIu9LUO/4mZDn7eOe42MRubWuUURmSFCPequI/Co8jFs18RH3nC8RlNqtR1X3EyxZelVI87VAqYhcJSJvuqIQK9wKXeGvUW90HzoLIEHd7A3u/fznKVxHY1oU68SN8UMvYJ6qXkiw4t9PQ/Z1VNWLgbkExUYKVHUA8DhB/XoIii5kqmoawUpTdVIICk8MAu4SkbNEZADBOtaDgWzglyKSGZbnaoJlMfsBvySoOd+QUoKOGxE5l6BC2BtAGZCtQVGIZ4DbG3shRGS4ux6DCGYpBojIkMaeb0xLEmvTb8aYhm1X1U1u+20gOWTfIvdvb6Av8Ge3bHkc8Knbtxl4WkSe59uKYgAvuantr0VkL3A2kAcsVdWDACLyHPBjgspMdYYApRpUWNsjIq+fIPcygsIt7YGfAUtU9aiInAcscjMF8QTFIhpruPuqy9OOoFNffQrPYUyLYJ24MX4I/Qz5KNA25PFB968A76lqTgPnX0HQ8Y4iKK1ZV987/Hlb03DpxYZ855rNqvo3EfkTwcj9WuBf3a6HgQdV9QURyQeKGjj9CG620BVTqSswIcB/q+rvGpnTmBbLptONaTkqgC4ikgPgpsYvFJFWQHdVfYNg2rojwej1RFYDY0TkeyKSSNAB/18Dx1wrInFuND30JM9XCswgGOWvc20dgN1ue+IJzqsiqOsNMBo4y22/ClwvIu3c++wmIsd9Jm/MPwIbiRvTQqjqYXcj2FxXjrI18BBB9aWnXJsQ3PW+z025N/Q874hIMbDeNS1Q1Y1hhy0FLiGoOFUJrDpJtOVACbBQv624VAT8r4jsJujYezZw3u+BP4rIeuA13IyDqi4XkVRgrXsPtcA4YO9JMhjTIlkVM2OMMcZTNp1ujDHGeMo6cWOMMcZT1okbY4wxnrJO3BhjjPGUdeLGGGOMp6wTN8YYYzxlnbgxxhjjKevEjTHGGE/9PwUiI2JNceLLAAAAAElFTkSuQmCC)

    * ROC 곡선

      ```python
      def roc_curve_plot(y_test , pred_proba_c1):
          # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
          fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
      
          # ROC Curve를 plot 곡선으로 그림. 
          plt.plot(fprs , tprs, label='ROC')
          # 가운데 대각선 직선을 그림. 
          plt.plot([0, 1], [0, 1], 'k--', label='Random')
          
          # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
          start, end = plt.xlim()
          plt.xticks(np.round(np.arange(start, end, 0.1),2))
          plt.xlim(0,1); plt.ylim(0,1)
          plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
          plt.legend()
          plt.show()
          
      roc_curve_plot(y_test, lr_clf.predict_proba(X_test)[:, 1] )
      
      # 그래프 설명
      # 일반적으로 곡선 자체는 RPR과 TRP의 변화값을 보든데 이용하고
      # 분류의 성능 지표로 사용되는 것은 ROC 곡선 면적에 기반한 AUC 값으로 결정
      # AUC(Area Under Curve) 값은 ROC 곡선 밑의 면적을 구한 것으로
      # 일반적으로 1에 가까울수록 좋음
      # AUC 수치가 커지려면 FPR이 작은 상태에서 얼마나 큰 TPR을 얻을 수 있느냐가 관건
      # 가운데 직선에서 멀어지고 왼족 상단 모서리 쪽으로 가파르게 곡선이 이동할 수록
      # 직사각형에 가까운 곡선이 되어 면적이 1에 가까워지는
      # 좋은 ROC AUC 성능 수치를 얻게 됨
      # 가운데 직선은 랜덤 수준의 이진 분류 AUC 값으로 0.5
      # 따라서 보통의 분류는 0.5이상의 AUC 값을 가짐
      ```

      ![img](data:image/png;base64,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t25l/fr1TJ06FYC4uDiio6OZM2cOO3bsIC4ujltvvdXKXyVHOkZQSPz9/Tl9+jQLFy7koYcesjqOUh7p5SXxJBw+59TnDKtckrEdwx3evnHjxmzbtg2Azz//nKZNm9KmTRsAgoODmTx5Mi1btmTAgAG8+eabjB49mpo1awIQEBBA//79nZrfGXSPwIXWr1/P888/D0DNmjXZs2ePFgGlPFh6ejorV66kU6dOgG1PP/u5PrfddhtJSUmcO3eOuLg4jzgXSPcIXCApKYlRo0YxefJkqlWrxvDhwylbtiwBAfp2K3U98vPN3ZkuXrxIVFQU+/fvp379+tx3332Abd5+brN0PGmmk+4RONmKFSuoXbs2kydPJjo6mri4OMqWLWt1LKXUdbgyRnDgwAEuX76cOUYQHh7Opk2brtp23759FC9enBIlShAeHs5vv/1mReT8McZ41E9gpRrGXZ0/f96ULVvWhIaGmp9++snqOEp5hYSEBKsjmJCQkMzbmzdvNjfddJO5fPmySU5ONtWrVzffffedMcaY5ORk06FDBzNp0iRjjDFbt241t912m9m1a5cxxpj09HTzzjvvuDxvTu8ZsMnk8rmqewRO8N1335Genk7x4sVZsWIFsbGxNG3a1OpYSikXqFu3LpGRkcybN4+goCBiYmL497//TWhoKHXq1KFhw4ZER0cDEBERwfvvv0+3bt2oVasWtWvX5siRIxb/Bn+n3Uevw5EjR4iOjmbRokXMnj2bnj17Wh1JKa+zY8cOatWqZXUMj5LTe3at7qO6R1AAxhhmzZpFWFgY33zzDRMmTNAmcUopj6XTWArg2WefZdq0aTRr1owZM2YQGhpqdSSllCowLQQOytokrnv37kRERNCvXz/8/HSnSinl2fRTzAE7duygefPmjBo1CoC7776b/v37axFQSnkF/SS7htTUVF577TWioqLYuXMndevWtTqSUko5nR4aykV8fDw9evQgNjaWRx55hA8++IAKFSpYHUsppZxO9whyERAQwNmzZ1m0aBFffvmlFgGlfNi12lBfj1mzZmWec2AlLQRZ/Pjjj5m9wkNDQ9m9ezddunSxOJVSymq5taH2FnpoCDh//jwvvPACU6dOpXr16rzwwgvaJE4pN9WyZcu/LfvnP/9J//79SU5O5v777//b+ieffJInn3ySkydP8vDDD1+1bvXq1fl6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)

      