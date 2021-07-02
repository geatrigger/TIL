# 210523
* remote access
```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install xrdp
```

* set root password
```shell
sudo su
passwd # set password
su -l pi
su
```

* docker
```shell
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo docker run hello-world
```

# 210525
* font
```shell
sudo apt install fonts-nanum
```

* korean input
```shell
# sudo apt install nabi
# sudo apt install im-config
# # go to menu>Preferences>Input Method select hangul
# sudo reboot
# sudo apt-get purge --auto-remove nabi
# sudo apt-get purge --auto-remove im-config
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install ibus ibus-hangul
# go to menu>Preferences>Input Method select ibus
# go to menu>Raspberry Pi Configuration>Localisation
# Locale : language : en, Coutry : GB(United Kingdom), Character Set : UTF-8
# Keyboard : Layout : English(UK)
# Country : GB Britain(UK)
```
히히 한글 발사

* booting by ssd
```shell
# boot from sd card
# update os and firmware
sudo apt update
sudo apt full-upgrade
sudo rpi-update
# reboot
# install bootloader
sudo rpi-eeprom-update -d -a
sudo raspi-config
# boot options(advanced option) > boot ROM version > latest > no > boot order > USB boot > not reboot
# see bootloader setting
vcgencmd bootloader_config
# Accessories > SD card copier > check from, to > start
# shut down and remove microSD card
```

* test sd card speed

  ```shell
  # raspbian method
  sudo apt update
  sudo apt install agnostics
  # Accessories > Raspberry Pi Diagnostics
  
  # dd
  # alert : it will be cached, so in second test change file content
  dd if=/dev/zero of=./speedTestFile bs=20M count=5 oflag=direct
  dd if=./speedTestFile of=/dev/zero bs=20M count=5 oflag=dsync
  ```

  * samsung T5 result

    ```shell
    # rpdiags
    prepare-file;0;0;299251;584
    seq-write;0;0;291271;568
    rand-4k-write;0;0;42918;10729
    rand-4k-read;40730;10182;0;0
    Sequential write speed 291271 KB/sec (target 10000) - PASS
    Random write speed 10729 IOPS (target 500) - PASS
    Random read speed 10182 IOPS (target 1500) - PASS
    # dd
    5+0 records in
    5+0 records out
    104857600 bytes (105 MB, 100 MiB) copied, 0.516366 s, 203 MB/s
    5+0 records in
    5+0 records out
    104857600 bytes (105 MB, 100 MiB) copied, 0.546984 s, 192 MB/s
    ```

    