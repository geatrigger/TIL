# anaconda

* anaconda 사용이유 : python 버전 관리 및 패키지 충돌방지하기 위해 가상환경 생성
* ipykernel 사용이유 : conda의 가상환경을 jupyter에서 관리하기 위해 사용

* ipykernel 만들어서 가상환경에 연결

  * 여러 명이서 같은 커널을 사용하면 충돌나기 때문에 각자의 가상환경에 커널을 각자 만들어 실행해야 한다

  ```shell
  python -m ipykernel install --user --name conda_name --display-name kernel_name
  ```

* jupyter 실행

  ```shell
  jupyter-notebook --ip=0.0.0.0 --no-browser --port=8890
  ```

  