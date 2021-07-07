# Hypervisor

* https://dora-guide.com/%ED%95%98%EC%9D%B4%ED%8D%BC%EB%B0%94%EC%9D%B4%EC%A0%80/
* 호스트 컴퓨터에서 다수의 운영 체제를 동시에 실행하기 위한 논리적 플랫폼
* 호스트 : 하나 이상의 가상머신을 실행하는 컴퓨터
* 게스트 : 각 가상머신
* 가상화를 통해 프로그래머는 시스템의 안정성을 해치지 않고 배포 및 디버깅이 가능해짐
* Type1 : native, Hardware->Hypervisor->OS
  * 게스트 운영 체제 중 하나의 문제가 다른 게스트 운영 체제에 영향을 미치지 않는다
  * Xen, Microsoft Hyper-V, VMware의 ESX/ESXi
* Type2 : hosted, Hardware->Host OS->Hypervisor->OS
  * 호스트 운영 체제에 생기는 문제는 전체 시스템에 영향을 준다
  * VMware Workstation, VMware Player, VirtualBox

# Container

* https://cloud.google.com/containers/?hl=ko
* 가상화를 통해 프로그래머는 시스템의 안정성을 해치지 않고 배포 및 디버깅이 가능
* OS 커널을 공유하여 하이퍼바이저보다 가벼워지고 이식성이 뛰어남
* Docker