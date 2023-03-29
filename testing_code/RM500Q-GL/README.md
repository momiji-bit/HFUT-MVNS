# 移远RM500Q-GL Linux驱动安装

## 1. USB驱动安装

```shell
# 安装必要软件
apt install make
apt install make-guile
apt install gcc
# 拨号上网
apt install udhcpc
apt install net-tools
systemctl stop ModemManager
systemctl disable ModemManager
```

