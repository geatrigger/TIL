# git rename

* Dockerfile을 실수로 DockerFile로 적어 대소문자를 바꾸려고 하는데 깃이 인식을 못함

  ```bash
  git mv -f crawler/DockerFile crawler/Dockerfile
  ```

* 디렉토리 이름이 잘못되었을 경우 바로 바꿀 순 없고, 중간에 임시로 다른 이름으로 바꿨다 바꿔야 한다.

  ```shell
  git mv kafka tmp
  git mv tmp Kafka
  ```

  