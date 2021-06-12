# git flow

* git flow 사용이유

  * 거대하고 복잡한 소프트웨어 개발 작업을 체계적으로 관리하기 위해서
  *  Vincent Driessen의 "A successful git branching model"이라는 글에서 제안한 브랜치 모델을 쉽게 사용할 수 있도록 몇개의 명령으로 구현해 놓은 git의 확장
  
* git flow에서 각 브랜치의 역할

  * master
    * 배포될 안정적인 버전의 소스코드
    * 배포 버전이 tag 형태로 관리
  * develop
    * 여러 개발자들이 develop 브랜치를 기준으로 feature 브랜치를 생성해서 작업을 한 다음 다시 develop 브랜치로 내용을 병합한다.
  * feature
    * 새로운 기능 개발을 위해 생성
  * release
    * 새로운 릴리즈를 생성
    * develop 브랜치를 기반으로 생성되고 master 브랜치와 병합 후 develop 브랜치와 병합된다.
    * 브랜치명이 master에 병합되고 나서 tag명이 된다.
  * hotfix
    * 긴급하게 수행되어야 할 버그 수정 반영
    * master 브랜치를 기반으로 생성

* 기타 git flow 특성

  * 다른 브랜치에서 git flow명령을 사용해도 똑같이 작동한다(hotfix를 하면 어디에 있든 무조건 master기반으로 생성)
  * 같은 version이름을 가진 브랜치 생성불가(hotfix, release)

* git-flow 설치(git 이미 설치)

  ```shell
  # ubuntu
  apt-get install git-flow
  # window는 git for window설치시 자동 설치
  ```

* git flow init

  ```shell
  git flow init
  ```

* feature

  ```shell
  git flow feature start crawl # feature/crawl라는 브랜치가 develop에서 새로 생성
  # 작업
  git flow feature publish crawl # origin에 브랜치 push
  git flow feature pull origin crawl # origin에서 crawl브랜치 가져옴
  
  git flow feature finish crawl # develop 브랜치로 checkout 한 다음, feature branch의 변경사항을 merge, feature branch를 삭제
  ```

* release

  ```shell
  git flow release start v0.0 # develop브랜치를 기반으로 release/v0.0 브랜치 생성
  git flow release publish v0.0
  git flow release trach v0.0
  git flow release finish v0.0
  #1) release 브랜치를 master 브랜치에 병합(merge) 하고
  #2) release 버전을 태그로 생성한다. 이 때, git flow init 에서 명시한 Version tag prefix 문자열이 release버전 앞에 추가되어 태그로 생성된다. 
  #3) release 브랜치를 develop 브랜치에 병합한다.
  #4) release 브랜치를 삭제한다.
  git push --tags # master브랜치를 tag와 함께 push
```
  
* hotfix

  ```shell
  git flow hotfix start v0.01 # master브랜치 기반으로 hotfix/v0.01 브랜치 생성
  git flow hotfix finish v0.01
  ```

  

