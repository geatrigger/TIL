# check if openssh-server is installed
dpkg : 
grep : 
```
dpkg -l | grep openssh
```

# apt-get install
```
sudo apt-get update
sudo apt-get install openssh-server
```

# ssh start
```
sudo service ssh start
```

# check service status
```
service --status-all | grep +
```

# check service port
```
sudo netstat -antp
```

# ssh config
```
sudo vim /etc/ssh/sshd_config
// option 바꿈
sudo service ssh restart
```

# authenticate by SSH key
<https://arsviator.blogspot.com/2015/04/ssh-ssh-key.html/>
client에서 OpenSSH 깔려있는 상태에서 시작
```
// client
ssh-keygen
ssh-copy-id test@foo.bar.com
// 혹은 server의 authorized_keys에 직접 붙여넣기
```

# dns
// 윈도우에서 예전 주소로 접속 될때
```
ipconfig/flushdns
```
운영체제                            로컬 캐시 갱신 명령                                  수동 mapping 파일
 MS Windows                   ipconfig /flushdns                                     {SystemDir}\drivers\etc\hosts
 Linux                                  /etc/init.d/nscd restart                              /etc/hosts
 OS X Mountain Lion       sudo killall -HUP mDNSResponder        /etc/hosts
 Mac OS X v10.6              dscacheutil -flushcache                            /etc/hosts
 Mac OS X Yosemite        lookupd –flushcache                                /etc/hosts
 Mac OS X El Capitan      sudo dscacheutil -flushcache
                                             sudo killall -HUP mDNSResponder        /etc/hosts
