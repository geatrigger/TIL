# git switch restore

* https://blog.outsider.ne.kr/1505
* 기존의 checkout은 하나의 명령어가 가진 기능이 너무 많았다
* checkout
  * Switch branches or restore working tree files
* switch
  * Switch branches
  * HEAD에서 브랜치 새로 만들기
  ```bash
  git switch -c step1
  ```
  * 그냥 브랜치 전환
  ```bash
  git switch step1
  ```
  * 특정 커밋으로 HEAD 옮김
  ```bash
  git switch --detach HEAD~1
  ```
* restore
  * Restore working tree files
  * 수정한 파일 복원
  ```bash
  # git checkout .
  # git reset --hard
  git restore .
  # git reset .
  git restore --staged .
  ```

  