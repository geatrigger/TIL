# classification 알고리즘

* LogisticRegression

* KNeighborsClassifier

* DecisionTreeClassifier

  ```python
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import accuracy_score
  
  # 예제 반복 시 마다 동일한 예측 결과 도출을 위해 random_state 설정
  dt_clf = DecisionTreeClassifier(random_state=156)
  dt_clf.fit(X_train , y_train)
  pred = dt_clf.predict(X_test)
  accuracy = accuracy_score(y_test , pred)
  print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy))
  
  # DecisionTreeClassifier의 하이퍼 파라미터 추출
  print('DecisionTreeClassifier 기본 하이퍼 파라미터:\n', dt_clf.get_params())
  ```

* RandomForestClassifier

  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  import pandas as pd
  import warnings
  warnings.filterwarnings('ignore')
  
  rf_clf1 = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=8, min_samples_split=8)
  rf_clf1.fit(X_train, y_train)
  pred = rf_clf1.predict(X_test)
  print('예측 정확도 : {0: .4f}'.format(accuracy_score(y_test, pred)))
  ```

# 앙상블 학습(Ensemble Learning)

* 여러 개의 모델을 조합해서 정확도를 높이는 방법

* VotingClassifier

  * voting

    ```python
    import pandas as pd
    
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    cancer = load_breast_cancer()
    
    data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    data_df.head(3)
    
    # 개별 모델은 로지스틱 회귀와 KNN 임. 
    lr_clf = LogisticRegression()
    knn_clf = KNeighborsClassifier(n_neighbors=8)
    
    # 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기 
    vo_clf = VotingClassifier( estimators=[('LR',lr_clf),('KNN',knn_clf)] , voting='soft' )
    
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                        test_size=0.2 , random_state= 156)
    
    # VotingClassifier 학습/예측/평가. 
    vo_clf.fit(X_train , y_train)
    pred = vo_clf.predict(X_test)
    print('Voting 분류기 정확도: {0:.4f}'.format(accuracy_score(y_test , pred)))
    
    # 개별 모델의 학습/예측/평가.
    classifiers = [lr_clf, knn_clf]
    for classifier in classifiers:
        classifier.fit(X_train , y_train)
        pred = classifier.predict(X_test)
        class_name= classifier.__class__.__name__
        print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test , pred)))
    ```

    

* Random forest

  * bagging

* XGBoost(eXtra Gradient Boost)

  * boost

