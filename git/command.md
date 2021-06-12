# git command

> 기본 명령어를 정리합니다.

### init

.git 폴더를 생성해주는 명령어, 처음 한 번만 실행합니다.

### add

working directory에 있는 파일을 staging area에 올리는 명령어

- ![Git - Git 기초](command.assets/areas.png)
- 기본 사용법
  - file name에 .을 입력하면 모든 파일, 폴더를 한번에 올림

```bash
git add <file name>
```

### commit

staging area에 있는 파일들을 하나의 commit으로 저장하는 명령어

- 기본 사용법

  - `-m` : 커밋 메시지를 작성하기 위한 옵션

    ```bash
    git commit -m "message"
    ```

### remote

원격저장소를 관리하기 위한 명령어



- add : 원격 저장소를 추가

  ```bash
  git remote add origin <URL>
  ```

- remove : 원격 저장소를 제거

  ```bash
  git remote remove <remote name>
  ```

### push

로컬에 저장되어 있는 커밋들을 원격 저장소에 업로드 하는 명령어

- 기본 사용방법

  ```bash
  git push origin master
  ```

### status

git의 현재 상태를 확인하는 명령어

- 기본 명령어

  ```bash
  git status
  ```


### config

git의 설정을 알 수 있는 명령어

```shell
git config --global --list # global로 저장된 config 보여줌
git config --global user.name "name" # set name
git config --gloabl user.email test@example.com # set email
```

### log

git의 log 파악

```shell
git log --graph --oneline
```

### reset

현재 커밋이후 변경사항이 있으면 삭제하는 명령어

```shell
git reset --hard # 바뀐 내용도 지움
git reset # add한 것만 취소(바뀐 내용은 유지)
git reset HEAD~1 # 가장 최근 commit 지움
```

### diff

수정된 파일이 있는지 확인

```shell
git diff
```

### checkout

브랜치 이동, 생성

```shell
git checkout develop
git checkout -b develop # 새 브랜치 생성
```

### merge

현재브랜치에서 다른 브랜치를 가져와 합침

```shell
git checkout develop
git merge test # develop에 test를 merge함
```

