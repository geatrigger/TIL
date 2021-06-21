# add user

```shell
useradd gckim
passwd gckim
mkdir /home/gckim
sudo usermod -aG sudo gckim # make gckim sudoer(super user)
grep -Po '^sudo.+:\K.*$' /etc/group # search sudoers
cat /etc/passwd # 모든 사용자 ID/password:UID:GID:설명:홈디렉터리:쉘
```

