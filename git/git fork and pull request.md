# git fork

* github에서 fork 버튼을 누른다.
* fork한 repository를 local에서 받은 후 수정하고 push한다.
* github의 fork한 repository에서 pull request 버튼을 누른다.
  * 만약 owner가 pull하기 전에 변경사항을 push해두었다면 자동으로 pull request가 업데이트 된다.
  * 각각의 branch에 대해서 pull request할 수 있다.

# git for update

* 원본에서 update가 생겼을 때 자신의 fork repository를 같이 update하기 위해 사용하는 방법

  ```bash
  # add the original repository as remote repository called "upstream"
  git remote add upstream https://github.com/OWNER/REPOSITORY.git
  
  # fetch all changes from the upstream repository
  git fetch upstream
  
  # switch to the master branch of your fork
  git checkout master
  
  # merge changes from the upstream repository into your fork
  git merge upstream/master
  ```

* 기존에 작업했던 branch의 출발 지점을 바꾸고 싶을 때(master 브랜치를 업데이트 시키고 나서)

  ```shell
  # switch to your feature branch
  git checkout my-feature-branch
  # update your feature branch from the master branch
  git rebase master
  ```

  

