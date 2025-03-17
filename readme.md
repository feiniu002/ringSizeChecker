


1. 构建镜像
```
docker build -t ring-size-checker:latest .
```

2. 推到镜像托管
```
docker tag xxxxxxxxx ccr.ccs.tencentyun.com/feiniuxuelianghua/ring-size-checker:latest
docker push ccr.ccs.tencentyun.com/feiniuxuelianghua/ring-size-checker:latest
```

3. 云服务器拉取镜像
```
docker pull ccr.ccs.tencentyun.com/feiniuxuelianghua/ring-size-checker:latest
```

4. 部署到云服务器后，运行以下命令运行容器：
```
docker run -d -p 5000:5000 --name ring-size-app xxxxxxxx
```

5. 访问以下地址进行测试：
```
http://119.91.139.199:5000/
```